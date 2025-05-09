import metaworld
import random
import os
import PIL
import argparse
import wandb

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.data import Dataset, DataLoader

import mujoco
import cv2
from shutil import rmtree
# import concurrent.futures

## Env variables
os.environ["MUJOCO_GL"] = "egl"

## Helper classes
class Config:
    def __init__(self, **kwargs):        
        ## Environment
        self.env_name = kwargs.get("env_name", "box-close-v2")
        self.env_idx = kwargs.get("env_idx", 0)
        self.seed = kwargs.get("seed", 42)
        self.gamma = kwargs.get("gamma", 0.99)
        self.num_workers = kwargs.get("num_workers", 16)
        self.initial_env_delay = kwargs.get("initial_env_delay", 0)
        ## Architecture
        # Actor
        self.actor_dim1 = kwargs.get("actor_dim1", 256)
        self.actor_dim2 = kwargs.get("actor_dim2", 256)  
        # Encoders - common
        self.embed_dim = kwargs.get("embed_dim", 64)
        # StateActionEncoder
        self.sae_dim1 = kwargs.get("sae_dim1", 256)
        self.sae_dim2 = kwargs.get("sae_dim2", 256)
        # GoalEncoder
        self.ge_dim1 = kwargs.get("ge_dim1", 256)
        self.ge_dim2 = kwargs.get("ge_dim2", 256)
        
        ## Training
        self.num_actors = kwargs.get("num_actors", 1)
        self.device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.replay_buffer_size = kwargs.get("replay_buffer_size", 1000000) # 1000000

        self.actor_batch_size = kwargs.get("actor_batch_size", 2**16) # 16
        self.actor_micro_batch_size = kwargs.get("actor_micro_batch_size", 2**10) # 10
        assert self.actor_batch_size % self.actor_micro_batch_size == 0, "Batch size must be divisible by micro batch size"
        self.each_actor_descent_steps = self.actor_batch_size // self.actor_micro_batch_size
        self.critic_batch_size = kwargs.get("critic_batch_size", 2**10) 
        self.critic_micro_batch_size = kwargs.get("critic_micro_batch_size", 2**8)
        assert self.critic_batch_size % self.critic_micro_batch_size == 0, "Batch size must be divisible by micro batch size"
        self.each_critic_descent_steps = self.critic_batch_size // self.critic_micro_batch_size

        self.max_episode_length = kwargs.get("max_episode_length", 150)
        self.max_log_std = kwargs.get("max_log_std", 5)
        self.min_log_std = kwargs.get("min_log_std", -6.5)
        self.mostly_deterministic_actor = kwargs.get("mostly_deterministic_actor", False)
        self.max_entropy_coeff = kwargs.get("max_entropy_coeff", 0)  
        self.num_transitions_before_training = kwargs.get("num_transitions_before_training", 10000) # 10000
        self.num_steps_per_rollout = kwargs.get("num_steps_per_rollout", 1)
        self.num_steps_per_update = kwargs.get("num_steps_per_update", 1)
        self.num_critic_updates_per_actor_update = kwargs.get("num_critic_updates_per_actor_update", 1)
        self.actor_lr = kwargs.get("actor_lr", 3e-4)
        self.critic_lr = kwargs.get("critic_lr", 3e-4)
        self.num_train_steps = kwargs.get("num_train_steps", 1000000)
        self.save_interval = kwargs.get("save_interval", 100000)

        self.video_out_path = kwargs.get("video_out_path", "out/goal_trajs/run1")
        self.video_save_interval = kwargs.get("video_save_interval", 50)
        self.max_videos_to_save = kwargs.get("max_videos_to_save", 10)
        self.checkpoint_dir = kwargs.get("checkpoint_dir", "out/checkpoints/run1")
        self.checkpoint_save_interval = kwargs.get("checkpoint_save_interval", 1000)
        self.max_checkpoints_to_save = kwargs.get("max_checkpoints_to_save", 3)
        self.overwrite_checkpoints = kwargs.get("overwrite_checkpoints", True)
        
        ## Wandb args
        self.wandb_dir = kwargs.get("wandb_dir", "out/wandb/run1")
        self.wandb_project = kwargs.get("wandb_project", "sgcrl_debug")
        self.wandb_entity = kwargs.get("wandb_entity", None)

class EnvSpecs:
    def __init__(
        self, 
        obs_low, 
        obs_high, 
        obs_dim, 
        action_low, 
        action_high,
        action_dim,
        goal_state,
    ):
        # Core inputs
        self.obs_low = obs_low
        self.obs_high = obs_high
        self.obs_dim = obs_dim
        self.action_low = action_low
        self.action_high = action_high
        self.action_dim = action_dim
        self.goal_state = goal_state

# Actor class
class Actor(nn.Module):
    def __init__(
        self,
        config,
        env_specs,
    ):
        super(Actor, self).__init__()
        # Save some helpful values
        self.register_buffer(
            "action_mean",
            (env_specs.action_low + env_specs.action_high) / 2,
        )
        self.register_buffer(
            "action_scale",
            (env_specs.action_high - env_specs.action_low) / 2,
        )
        self.max_log_std = config.max_log_std
        self.min_log_std = config.min_log_std
        self.mostly_deterministic_actor = config.mostly_deterministic_actor
        
        in_dim =  2 * env_specs.obs_dim 
        out_dim = env_specs.action_dim    
        
        # Actor network
        self.fc_1 = nn.Linear(in_dim, config.actor_dim1)
        self.relu_1 = nn.ReLU()
        self.fc_2 = nn.Linear(config.actor_dim1, config.actor_dim2)
        self.relu_2 = nn.ReLU()
        self.fc_out = nn.Linear(config.actor_dim2, out_dim)
        self.tanh_out = nn.Tanh()    
        self.fc_log_std = nn.Linear(config.actor_dim2, out_dim)

    def forward(self, state, goal):
        x1 = torch.cat((state, goal), dim=-1)
        
        # Pass through the network
        x2 = self.relu_1(self.fc_1(x1))
        x3 = self.relu_2(self.fc_2(x2))
        x = self.fc_out(x3)
        log_std = torch.clamp(self.fc_log_std(x3), self.min_log_std, self.max_log_std)
        
        if self.mostly_deterministic_actor:
            with torch.no_grad():
                std = torch.ones_like(log_std) * 1e-3 
                log_std = torch.ones_like(log_std) * -10
        else:
            std = torch.exp(log_std)
        
        distribution = Normal(x, std)
        sampled = distribution.rsample()
        output = self.tanh_out(sampled)
        
        # Scale the output to the action space
        num_extra_dims = len(x.shape) - 1
        action = self.action_mean.view([1] * num_extra_dims + [-1]) + output * self.action_scale.view([1] * num_extra_dims + [-1])

        # Calculate the log probability of the action
        log_prob = distribution.log_prob(sampled)
        # Subtract the log determinant of the Jacobian for the tanh transformation
        tanh_correction = torch.log(
            self.action_scale.view([1] * num_extra_dims + [-1]) * (1 - output**2) + 1e-6
        )
        log_prob = torch.sum(log_prob - tanh_correction, dim=-1, keepdim=True)
        
        with torch.no_grad():
            upper = self.tanh_out(x + std)
            lower = self.tanh_out(x - std)
            ratio = (upper - lower) / 2     # the scale is already factored in

        return action, log_prob, ratio

# Critic classes
class StateActionEncoder(nn.Module):
    def __init__(self, config, env_specs):
        super(StateActionEncoder, self).__init__()
        
        in_dim = env_specs.obs_dim + env_specs.action_dim 
        out_dim = config.embed_dim
        
        # StateActionEncoder network
        self.fc_1 = nn.Linear(in_dim, config.sae_dim1)
        self.relu_1 = nn.ReLU()
        self.fc_2 = nn.Linear(config.sae_dim1, config.sae_dim2)
        self.relu_2 = nn.ReLU()
        self.fc_out = nn.Linear(config.sae_dim2, out_dim)
        
    def forward(self, state, action):
        # Both state and action should be either (dim,) or (batch_size, dim)
        x1 = torch.cat((state, action), dim=-1)
        
        # Pass through the network
        x2 = self.relu_1(self.fc_1(x1))
        x3 = self.relu_2(self.fc_2(x2))
        
        h = self.fc_out(x3)

        return h

class GoalEncoder(nn.Module):
    def __init__(self, config, env_specs):
        super(GoalEncoder, self).__init__()
        in_dim = env_specs.obs_dim 
        out_dim = config.embed_dim
        
        # GoalEncoder network
        self.fc_1 = nn.Linear(in_dim, config.ge_dim1)
        self.relu_1 = nn.ReLU()
        self.fc_2 = nn.Linear(config.ge_dim1, config.ge_dim2)
        self.relu_2 = nn.ReLU()
        self.fc_out = nn.Linear(config.ge_dim2, out_dim)
        
    def forward(self, state):
        # State should be either (obs_dim,) or (batch_size, obs_dim)
        x1 = state 
        
        # Pass through the network
        x2 = self.relu_1(self.fc_1(x1))
        x3 = self.relu_2(self.fc_2(x2))
        
        h = self.fc_out(x3)

        return h

class ReplayBuffer:
    def __init__(
        self, 
        config,
        env_specs,
    ):
        self.max_size = config.replay_buffer_size
        self.gamma = config.gamma
        
        self.ptr = 0
        self.size = 0
        
        self.state = torch.empty((self.max_size, env_specs.obs_dim), dtype=torch.float32)
        self.action = torch.empty((self.max_size, env_specs.action_dim), dtype=torch.float32)
        self.done = torch.empty((self.max_size, 1), dtype=torch.float32)

    def get_state_dict(self):
        state_dict = {
            "state": self.state,
            "action": self.action,
            "done": self.done,
            "size": self.size,
        }
        return state_dict
    
    def load_state_dict(self, state_dict):
        self.state = state_dict["state"]
        self.action = state_dict["action"]
        self.done = state_dict["done"]
        self.size = state_dict.get("size", self.max_size)

    def add(self, state, action, done):
        # If you're calling this function, make sure to insert sequential states
        # sequentially. This helps us when we want to perform sampling of future states.
        # Ensure that we move all tensors to the cpu - we don't want to expend GPU memory
        self.state[self.ptr] = state.cpu()
        self.action[self.ptr] = action.cpu()
        self.done[self.ptr] = done.cpu()
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        
    def is_same_trajectory(self, idx1, idx2):
        if idx1 == idx2:
            # We don't want the same state twice
            return False 
        elif idx1 < idx2:
            return self.done[idx1:idx2].sum() == 0
        else:
            return self.done[idx1:self.size].sum() + self.done[:idx2].sum() == 0
    
    def sample_single_entry_for_actor(self):
        n_attempts = 0
        while True:
            idx1 = np.random.randint(0, self.size)
            offset = np.random.geometric(1-self.gamma)
            idx2 = (idx1 + offset) % self.size
            if self.is_same_trajectory(idx1, idx2):
                break
            else:
                n_attempts += 1
            if n_attempts > 100:
                assert False, f"Tried too many times: {idx1}"
        return self.state[idx1], self.state[idx2]
                
    def sample_for_actor(self, batch_size, device='cpu'):
        # Select a random batch of indices
        idx = np.random.choice(self.size, batch_size, replace=False)
        # Sample the states and actions
        states = []
        goals = []

        for _ in range(batch_size):
            state, goal = self.sample_single_entry_for_actor()
            states.append(state)
            goals.append(goal)

        states = torch.stack(states)
        goals = torch.stack(goals)

        return states.to(device), goals.to(device)
    
    def sample_single_entry_for_critic(self):
        # Sample a random index
        n_attempts = 0
        while True:
            idx1 = np.random.randint(0, self.size)
            offset = np.random.geometric(1-self.gamma)
            idx2 = (idx1 + offset) % self.size
            if self.is_same_trajectory(idx1, idx2):
                break
            else:
                n_attempts += 1
            if n_attempts > 100:
                assert False, f"Tried too many times: {idx1}"
        
        return self.state[idx1], self.action[idx1], self.state[idx2]

    def sample_for_critic(self, batch_size, device='cpu'):
        # Do the following `batch_size` times
        # 1) Sample s,a from an index i randomly
        # 2) Sample s+ randomly as (i + geom(1-gamma)) % buffer_size randomly, and make
        #    sure that s+ is from the same trajectory
        # Just returning these is fine because we will use the other trajectories as the negative goals
        states = []
        actions = []
        goals = []
        
        for _ in range(batch_size):
            state, action, goal = self.sample_single_entry_for_critic()
            states.append(state)
            actions.append(action)
            goals.append(goal)
        
        return torch.stack(states).to(device), torch.stack(actions).to(device), torch.stack(goals).to(device)

## Code starts here
def parse_args_and_config():
    parser = argparse.ArgumentParser(description="Train SGCRL Agent")

    # Environment args
    parser.add_argument('--env_name', type=str, default="box-close-v2", help='Metaworld environment name')
    parser.add_argument('--env_idx', type=int, default=0, help='Task index within the environment class')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--num_workers', type=int, default=16, help='Number of workers for parallel sampling')
    parser.add_argument('--initial_env_delay', type=int, default=0, help='Number of steps to delay before starting to collect data')
    
    # Architecture args
    parser.add_argument('--actor_dim1', type=int, default=256, help='Actor hidden layer 1 size')
    parser.add_argument('--actor_dim2', type=int, default=256, help='Actor hidden layer 2 size')
    parser.add_argument('--embed_dim', type=int, default=64, help='Common embedding dimension for critic encoders')
    parser.add_argument('--sae_dim1', type=int, default=256, help='StateActionEncoder hidden layer 1 size')
    parser.add_argument('--sae_dim2', type=int, default=256, help='StateActionEncoder hidden layer 2 size')
    parser.add_argument('--ge_dim1', type=int, default=256, help='GoalEncoder hidden layer 1 size')
    parser.add_argument('--ge_dim2', type=int, default=256, help='GoalEncoder hidden layer 2 size')

    # Training args
    parser.add_argument('--num_actors', type=int, default=1, help='Number of actors in the ensemble')
    # device is handled automatically
    parser.add_argument('--replay_buffer_size', type=int, default=1_000_000, help='Size of the replay buffer')
    parser.add_argument('--actor_batch_size', type=int, default=2**14, help='Total batch size for actor update')
    parser.add_argument('--actor_micro_batch_size', type=int, default=2**8, help='Micro batch size for actor update')
    parser.add_argument('--critic_batch_size', type=int, default=2**10, help='Total batch size for critic update')
    parser.add_argument('--critic_micro_batch_size', type=int, default=2**10, help='Micro batch size for critic update')
    parser.add_argument('--max_episode_length', type=int, default=150, help='Maximum steps per episode')
    parser.add_argument('--max_log_std', type=float, default=5.0, help='Maximum log standard deviation for actor')
    parser.add_argument('--min_log_std', type=float, default=-7.0, help='Minimum log standard deviation for actor')
    parser.add_argument('--mostly_deterministic_actor', action='store_true', help='Make actor mostly deterministic (small std dev)')
    parser.add_argument('--max_entropy_coeff', type=float, default=0.0, help='Coefficient for entropy maximization term')
    parser.add_argument('--num_transitions_before_training', type=int, default=10000, help='Number of transitions to collect before starting training')
    parser.add_argument('--num_steps_per_rollout', type=int, default=1, help='Number of environment steps per rollout')
    parser.add_argument('--num_steps_per_update', type=int, default=1, help='Frequency of updates (controls actor/critic update relative frequency)')
    parser.add_argument('--num_critic_updates_per_actor_update', type=float, default=1.0, help='Ratio of critic updates to actor updates')
    parser.add_argument('--actor_lr', type=float, default=3e-4, help='Actor learning rate')
    parser.add_argument('--critic_lr', type=float, default=3e-4, help='Critic learning rate')
    parser.add_argument('--num_train_steps', type=int, default=1000000, help='Total number of training steps')
    parser.add_argument('--save_interval', type=int, default=100000, help='Frequency of saving checkpoints (deprecated, use checkpoint_save_interval)') # Mark as deprecated maybe

    # Video args
    parser.add_argument('--video_out_path', type=str, default="out/goal_trajs/run1", help='Directory to save trajectory videos')
    parser.add_argument('--video_save_interval', type=int, default=30, help='Frequency (in steps) to save videos')
    parser.add_argument('--max_videos_to_save', type=int, default=10, help='Maximum number of videos to keep')

    # Checkpoint args
    parser.add_argument('--checkpoint_dir', type=str, default="out/checkpoints/run1", help='Directory to save checkpoints')
    parser.add_argument('--checkpoint_save_interval', type=int, default=1000, help='Frequency (in steps) to save checkpoints')
    parser.add_argument('--max_checkpoints_to_save', type=int, default=3, help='Maximum number of checkpoints to keep')
    parser.add_argument('--overwrite_checkpoints', action='store_true', help='Overwrite existing checkpoints in the directory on start')

    # Wandb args
    parser.add_argument('--wandb_dir', type=str, default="out/wandb/run1", help='Directory for wandb offline logs')
    parser.add_argument('--wandb_project', type=str, default="sgcrl_debug", help='Wandb project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='Wandb entity (username or team)')

    args = parser.parse_args()
    config_kwargs = vars(args)
    config_kwargs['device'] = "cuda" if torch.cuda.is_available() else "cpu"

    config = Config(**config_kwargs)

    # Create necessary directories
    os.makedirs(config.video_out_path, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.wandb_dir, exist_ok=True)

    return config

def get_environment(idx=0, initial_env_delay=0):    
    test_class = metaworld.envs.mujoco.env_dict.ALL_V2_ENVIRONMENTS['box-close-v2']
    
    class EnvClass(test_class):
        #####
        # Sawyer Box Close - the observations are 39D, structured as follows:
        # 18D for current obs + 18D for prev obs + 3D for goal
        # Each 18D has - 3D for gripper pos, 1D for gripper separation, (3D pos + 4D quat) for box, and then 7 x 0s (unused for Box Close)
        # Also it seems that the goal pos is hidden in this observation, so we'll have to get it manually.
        #####
        def __init__(self, *args, **kwargs):
            self.keep_prev = kwargs.pop("keep_prev", True)
            self.initial_env_delay = kwargs.pop("initial_env_delay", 0)
            super().__init__(*args, **kwargs)
            self._set_task_called = True
            self._partially_observable = False
            self._freeze_rand_vec = False
            self._saved_state_rand_vec = None
            if self.keep_prev:
                self.obs_selection_indices = np.array(
                    [
                        0, 1, 2,            # gripper pos
                        3,                  # gripper separation
                        4, 5, 6,            # box pos
                        7, 8, 9, 10,        # box quat
                        18, 19, 20,         # prev gripper pos
                        21,                 # prev gripper separation
                        22, 23, 24,         # prev box pos
                        25, 26, 27, 28,     # prev box quat
                    ]
                )
            else:
                self.obs_selection_indices = np.array(
                    [
                        0, 1, 2,            # gripper pos
                        3,                  # gripper separation
                        4, 5, 6,            # box pos
                        7, 8, 9, 10,        # box quat
                    ]
                )
        
        def get_obs_low_high_and_dim(self):
            obs_space = self.observation_space
            low = obs_space.low
            high = obs_space.high
            
            low = low[self.obs_selection_indices]
            high = high[self.obs_selection_indices]
            dim = low.shape[0]
            return low, high, dim
        
        def get_action_low_high_and_dim(self):
            action_space = self.action_space
            low = action_space.low
            high = action_space.high
            dim = low.shape[0]
            return low, high, dim
        
        def get_goal_coords(self):
            return self._goal
        
        def get_goal_obs(self):
            goal_coords = self.get_goal_coords()        # 3D
            
            ideal_gripper_pos = goal_coords + np.array([0.0, 0.0, 0.03])
            ideal_gripper_separation = 0.4
            ideal_lid_pos = goal_coords
            ideal_lid_quat = np.array([0.707, 0, 0, 0.707])
            ideal_state = np.concatenate(
                [
                    ideal_gripper_pos,
                    [ideal_gripper_separation],
                    ideal_lid_pos,
                    ideal_lid_quat,
                ]
            )
            if self.keep_prev:
                ideal_state = np.concatenate(
                    [
                        ideal_state,
                        ideal_gripper_pos,
                        [ideal_gripper_separation],
                        ideal_lid_pos,
                        ideal_lid_quat,
                    ]
                )
            return ideal_state
        
        ## Helpers that allow artificially setting a state -- useful for getting a goal render
        def _set_hand_to_position(
            self, 
            hand_pos,
            steps=50
        ):
            
            mocap_id = self.model.body_mocapid[self.data.body("mocap").id]
            for _ in range(steps):
                self.data.mocap_pos[mocap_id][:] = hand_pos
                self.do_simulation([-1, 1], self.frame_skip)
            self.init_tcp = self.tcp_center
            
        def set_state_of_everything(
            self,
            gripper_pos,
            lid_pos,
            lid_angle,
            gripper_separation=0.4,
            steps=100,
        ):
            # These offsets were found by hand. I don't know why the setting of state isn't precise, but I can at least
            # swap this hard-coding for a binary search later.
            gripper_offset = np.array([0.0, 0.0, 0.005])
            lid_offset = np.array([0.0, 0.0, -0.075])
            
            self._set_hand_to_position(gripper_pos + gripper_offset, steps)  # This involves running the simulation, so move lid later
            _sep = self._get_obs()[3]
            while abs(gripper_separation - _sep) > 0.01:
                if gripper_separation < _sep:
                    # Expand
                    action = np.array([0.0, 0.0, 0.0, 1.0])
                else:
                    # Contract
                    action = np.array([0.0, 0.0, 0.0, -1.0])
                self.step(action)
                _sep = self._get_obs()[3]
            
            self.obj_init_angle = lid_angle
            self.obj_init_pos = lid_pos + lid_offset
            self._set_obj_xyz(lid_pos + lid_offset)
            return self._get_obs()[self.obs_selection_indices]
        
        ## Wrappers around the original env functions
        def reset_wrapper(self, apply_delay=True):
            obs, _ = self.reset()
            if apply_delay:
                # Take no action for the initial delay
                for _ in range(self.initial_env_delay):
                    self.step(np.zeros(self.action_space.shape))
            self._goal = self._target_pos.copy()
            return obs[self.obs_selection_indices]
        
        def step_wrapper(self, action):
            obs, reward, done, truncate, info = self.step(action)
            done = done or truncate
            return obs[self.obs_selection_indices], reward, done, info

    env = EnvClass(render_mode="rgb_array", camera_name="corner2", initial_env_delay=initial_env_delay)
    return env

def get_env_specs(
    env,
    config,
):
    # ! This changes the state of the environment, so be careful with it
    # We want to always return to the original state of the environment, so we need to reset it
    env.reset_wrapper(apply_delay=False) 
    
    # The obs low/high are numpy arrays
    obs_low, obs_high, obs_dim = env.get_obs_low_high_and_dim()
    obs_low, obs_high = torch.from_numpy(obs_low), torch.from_numpy(obs_high)
    
    # The action low/high are floats
    action_low, action_high, action_dim = env.get_action_low_high_and_dim()
    action_low, action_high = torch.from_numpy(action_low), torch.from_numpy(action_high)
    
    goal_obs = env.get_goal_obs()
    goal_obs = torch.from_numpy(goal_obs)

    return EnvSpecs(
        obs_low=obs_low,
        obs_high=obs_high,
        obs_dim=obs_dim,
        action_low=action_low,
        action_high=action_high,
        action_dim=action_dim,
        goal_state=goal_obs,
    )

@torch.no_grad()
def roll_out_trajectory(
    env,
    policy,
    env_specs,
    config,
    replay_buffer,
    video_out_path=None,
    is_random_policy=False,
):
    device = config.device
    frames = []
    
    # Roll out a trajectory and add it to the replay buffer
    obs = env.reset_wrapper()
    obs = torch.FloatTensor(obs).to(device)
    goal_state = env_specs.goal_state.to(device=device, dtype=torch.float32)
    done = False
    
    discounted_reward_sum = 0.
    gamma = config.gamma 
    mean_ratio = 0.
    n = 0
    success = 0

    for t in range(config.max_episode_length):
        if is_random_policy:
            action = torch.FloatTensor(env.action_space.sample()).to(device)
            neg_log_prob = torch.tensor(0.)
            ratio = torch.tensor(0.)
        else:
            action, neg_log_prob, ratio = policy(obs, goal_state)

        mean_ratio += ratio.mean().item()

        n += 1
        next_obs, reward, done, info = env.step_wrapper(action.cpu().numpy().flatten())
        next_obs = torch.FloatTensor(next_obs).to(device)
        done = torch.FloatTensor([done]).to(device)

        if info["success"]:
            success = 1.
        
        if video_out_path is not None:
            frame = env.render()
            frame = PIL.Image.fromarray(frame)
            frame = frame.rotate(180)
            frames.append(np.array(frame))

        # Add the transition to the replay buffer
        if t == config.max_episode_length - 1:
            done = torch.FloatTensor([True]).to(device)
        replay_buffer.add(obs, action, done) # Removed img
        
        discounted_reward_sum += reward * (gamma ** t)
        
        obs = next_obs
        if done:
            break

    if video_out_path is not None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use appropriate codec for your format
        fps = 30  # Frames per second
        frame_width, frame_height = frames[0].shape[1], frames[0].shape[0]
        out = cv2.VideoWriter(video_out_path, fourcc, fps, (frame_width, frame_height))
        for frame in frames:
            out.write(frame)
        out.release()

        maybe_clean_video_dir(config)
            
    return discounted_reward_sum, mean_ratio / n, success
        
def calculate_actor_loss(
    policy,
    state_action_encoder,
    goal_encoder,
    config,
    replay_buffer,
    batch,
):
    device = config.device
    
    # Sample a batch of states and actions from the replay buffer
    states, goals = batch
    
    # Get the goal state
    actions, log_probs, _ = policy(states, goals)    
    state_action_encodings = state_action_encoder(states, actions)
    goal_encodings = goal_encoder(goals).detach()

    # Both encodings are of shape (batch_size, embed_dim)
    actor_term_1 = -torch.multiply(state_action_encodings, goal_encodings).sum(dim=-1) # (batch_size,)
    actor_term_2 = config.max_entropy_coeff * log_probs
    
    # Combine critic objective with negative log probabilities (entropy maximization)
    actor_loss = (actor_term_1 + actor_term_2).mean()

    return actor_loss

def calculate_critic_loss(
    state_action_encoder,
    goal_encoder,
    config,
    replay_buffer,
    batch
):
    device = config.device
    
    # Sample a batch of states and actions from the replay buffer
    states, actions, goals = batch
        
    # Combine positive and negative goals so that we can do one forward pass
    state_action_encodings = state_action_encoder(states, actions) # (batch_size, embed_dim)
    all_goal_embeddings = goal_encoder(goals) # (batch_size, embed_dim)
    
    # Get similarities between every pair of (state, action), goal
    similarities = torch.matmul(state_action_encodings, all_goal_embeddings.T) # (batch_size, batch_size)
    
    ## Two terms in the critic loss
    # First: logsoftmax of the first critic value - average the values on the diagonal
    # Effectively, this means that the positive goal for one state-action pair is the negative goal for other state-action pairs
    log_softmax = F.log_softmax(similarities, dim=-1) # (batch_size, batch_size)
    critic_term_1 = torch.mean(torch.diagonal(log_softmax)) # scalar
    # Second: regularization term = -0.01 * (logsumexp of all critic values)**2
    critic_term_2 = -0.01 * (torch.logsumexp(similarities, dim=-1) ** 2) # (batch_size,)
    critic_term_2 = torch.mean(critic_term_2) # scalar
    
    # Combine the two terms
    critic_loss = -(critic_term_1 + critic_term_2)

    return critic_loss

def maybe_clean_video_dir(config):
    video_out_dir = config.video_out_path
    videos = os.listdir(video_out_dir)
    videos = [v for v in videos if v.endswith(".mp4")]
    videos = sorted(videos, key=lambda x: int(x.split("_")[-1].split(".")[0]))
    if len(videos) > config.max_videos_to_save:
        for video in videos[:-config.max_videos_to_save]:
            os.remove(os.path.join(video_out_dir, video))

def save_checkpoint(
    policies,
    state_action_encoder,
    goal_encoder,
    replay_buffer,
    config,
    step,
    episodes,
):
    checkpoint_dir = config.checkpoint_dir
    
    existing_checkpoints = os.listdir(checkpoint_dir)
    existing_checkpoints = [int(c.split("_")[-1].split(".")[0]) for c in existing_checkpoints]
    existing_checkpoints = sorted(existing_checkpoints)
        
    # Save the checkpoint
    out_dir = os.path.join(checkpoint_dir, f"checkpoint_{step}")
    os.makedirs(out_dir, exist_ok=True)
    for i, policy in enumerate(policies):
        torch.save(policy.state_dict(), os.path.join(out_dir, f"policy_{i}.pth"))
    torch.save(state_action_encoder.state_dict(), os.path.join(out_dir, "state_action_encoder.pth"))
    torch.save(goal_encoder.state_dict(), os.path.join(out_dir, "goal_encoder.pth"))
    torch.save(replay_buffer.get_state_dict(), os.path.join(out_dir, "replay_buffer.pth"))
    with open(os.path.join(out_dir, "episodes.txt"), "w") as f:
        f.write(f"{episodes}\n")

    if len(existing_checkpoints) >= config.max_checkpoints_to_save:
        for checkpoint in existing_checkpoints[:(-config.max_checkpoints_to_save+1)]:
            rmtree(os.path.join(checkpoint_dir, f"checkpoint_{checkpoint}"))

def load_latest_checkpoint_if_exists(
    policies,
    state_action_encoder,
    goal_encoder,
    replay_buffer,
    config,
):
    checkpoint_dir = config.checkpoint_dir
    existing_checkpoints = os.listdir(checkpoint_dir)
    existing_checkpoints = [int(c.split("_")[-1].split(".")[0]) for c in existing_checkpoints]
    if config.overwrite_checkpoints:
        for checkpoint in existing_checkpoints:
            rmtree(os.path.join(checkpoint_dir, f"checkpoint_{checkpoint}"))
            existing_checkpoints.remove(checkpoint)
        # Also remove old videos
        videos = os.listdir(config.video_out_path)
        videos = [v for v in videos if v.endswith(".mp4")]
        videos = sorted(videos, key=lambda x: int(x.split("_")[-1].split(".")[0]))
        for video in videos:
            os.remove(os.path.join(config.video_out_path, video))
        return 0, 0
    if len(existing_checkpoints) == 0:
        return 0, 0
    latest_checkpoint = max(existing_checkpoints)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{latest_checkpoint}")

    # Load the checkpoint
    if not config.overwrite_checkpoints:
        print(f"Loading checkpoint {latest_checkpoint}")
        for i, policy in enumerate(policies):
            policy.load_state_dict(torch.load(os.path.join(checkpoint_path, f"policy_{i}.pth")))
        state_action_encoder.load_state_dict(torch.load(os.path.join(checkpoint_path, "state_action_encoder.pth")))
        goal_encoder.load_state_dict(torch.load(os.path.join(checkpoint_path, "goal_encoder.pth")))
        replay_buffer.load_state_dict(torch.load(os.path.join(checkpoint_path, "replay_buffer.pth")))
        with open(os.path.join(checkpoint_path, "episodes.txt"), "r") as f:
            episodes = int(f.read())
        print(f"Loaded checkpoint {latest_checkpoint} with {episodes} episodes")
    else:
        latest_checkpoint, episodes = 0, 0

    return latest_checkpoint, episodes

def train(
    policies,
    state_action_encoder,
    goal_encoder,
    env,
    config,
    env_specs,
    replay_buffer,
):
    env.reset_wrapper()
    for policy in policies:
        policy.to(config.device)
    state_action_encoder.to(config.device)
    goal_encoder.to(config.device)
    
    # Optimizers
    actor_optimizers = [optim.Adam(policy.parameters(), lr=config.actor_lr) for policy in policies]
    critic_optimizer = optim.Adam(
        list(state_action_encoder.parameters()) + list(goal_encoder.parameters()),
        lr=config.critic_lr,
    )
    
    for policy in policies:
        policy.train()
    state_action_encoder.train()
    goal_encoder.train()
    
    last_critic_loss, last_actor_loss = -1, -1
    # Initialize metrics dictionary for logging
    metrics_to_log = {}

    latest_checkpoint, episodes = load_latest_checkpoint_if_exists(policies, state_action_encoder, goal_encoder, replay_buffer, config)
    
    # Training loop
    bar = tqdm(range(latest_checkpoint, config.num_train_steps))

    if config.num_critic_updates_per_actor_update >= 1:
        num_steps_per_critic_update = config.num_steps_per_update
        num_steps_per_actor_update = int(num_steps_per_critic_update * config.num_critic_updates_per_actor_update)
    else:
        num_steps_per_actor_update = config.num_steps_per_update
        num_steps_per_critic_update = int(num_steps_per_actor_update / config.num_critic_updates_per_actor_update)

    for step in bar:
        # Roll out one trajectory
        should_rollout = (
            replay_buffer.size < config.num_transitions_before_training or 
            step % config.num_steps_per_rollout == 0
        )
        should_save_video = (
            should_rollout and
            step >= config.num_steps_per_rollout and
            replay_buffer.size >= config.num_transitions_before_training and
            (1+step) // config.video_save_interval != ((1+step-config.num_steps_per_rollout) // config.video_save_interval)
        )
        if should_save_video:
            video_out_path = f"{config.video_out_path}/video_{step+1}.mp4"
        else:
            video_out_path = None

        if should_rollout:
            episodes += 1
            # Choose an actor at random
            index = np.random.randint(0, config.num_actors)
            policy = policies[index]
            policy.eval()
            discounted_reward_sum, mean_std, success = roll_out_trajectory(
                env, 
                policy, 
                env_specs, 
                config, 
                replay_buffer, 
                video_out_path,
                is_random_policy=(replay_buffer.size < config.num_transitions_before_training)
            )
            policy.train()
            # Update metrics for logging
            metrics_to_log['rollout/discounted_reward'] = discounted_reward_sum
            metrics_to_log['rollout/mean_std_ratio'] = mean_std
            metrics_to_log['rollout/episodes'] = episodes
            metrics_to_log['rollout/success'] = success

        if replay_buffer.size < config.num_transitions_before_training:
            bar.set_description(
                f"[{step}, {episodes}] r: {discounted_reward_sum:.2f} s: {mean_std:.3f} [not training yet]"
            )
            continue
        
        # Is it time to train the critic?
        if step % num_steps_per_critic_update == 0:
            for _ in range(config.each_critic_descent_steps):
                batch = replay_buffer.sample_for_critic(config.critic_micro_batch_size, device=config.device)
                critic_optimizer.zero_grad()
                critic_loss = calculate_critic_loss(state_action_encoder, goal_encoder, config, replay_buffer, batch)
                critic_loss.backward()
                critic_optimizer.step()
            last_critic_loss = critic_loss.item()
            # Update metrics for logging
            metrics_to_log['train/critic_loss'] = last_critic_loss
        
        # Is it time to train the actor?
        if step % num_steps_per_actor_update == 0:
            # Choose a random policy
            index = np.random.randint(0, config.num_actors)
            policy = policies[index]
            for _ in range(config.each_actor_descent_steps):
                batch = replay_buffer.sample_for_actor(config.actor_micro_batch_size, device=config.device)
                actor_optimizers[index].zero_grad()
                actor_loss = calculate_actor_loss(policy, state_action_encoder, goal_encoder, config, replay_buffer, batch)
                actor_loss.backward()
                actor_optimizers[index].step()
            last_actor_loss = actor_loss.item()
            # Update metrics for logging
            metrics_to_log['train/actor_loss'] = last_actor_loss
        
        # Log accumulated metrics to wandb at the end of the step
        if metrics_to_log: # Only log if something was updated
            wandb.log(metrics_to_log, step=step)
        
        bar.set_description(
            f"[{step}, {episodes}] r: {discounted_reward_sum:.2f}, s: {mean_std:.3f}, al: {last_actor_loss:.2f}, cl: {last_critic_loss:.2f}"
        )

        if (step+1) % config.checkpoint_save_interval == 0:
            save_checkpoint(policies, state_action_encoder, goal_encoder, replay_buffer, config, step+1, episodes)

def main():
    print("[+] Parsing Arguments and Config")
    config = parse_args_and_config()

    os.environ["WANDB_MODE"] = "offline"
    wandb.init(
        dir=config.wandb_dir,
        project=config.wandb_project,
        entity=config.wandb_entity,
        config=vars(config),
        name=f"{config.env_name}-idx{config.env_idx}-seed{config.seed}" 
    )

    print(f"[*] Using device: {config.device}")
    print(f"[*] Using environment: {config.env_name} (Task Index: {config.env_idx})")
    print(f"[*] Random Seed: {config.seed}")

    print("[+] Getting environment")
    env = get_environment(config.env_idx, initial_env_delay=config.initial_env_delay)
    env.seed(config.seed)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    print("[+] Getting environment specs")
    specs = get_env_specs(env, config) # get_env_specs changes the state of the environment    
    
    print("[+] Getting networks and replay buffer")
    policies = [Actor(config, specs) for _ in range(config.num_actors)]
    state_action_encoder = StateActionEncoder(config, specs)
    goal_encoder = GoalEncoder(config, specs)
    replay_buffer = ReplayBuffer(config, specs)
    
    print("[+] Training")
    train(
        policies,
        state_action_encoder,
        goal_encoder,
        env,
        config,
        specs,
        replay_buffer,
    )
    
    print("[!] Training finished") 

if __name__ == "__main__":
    main()
    wandb.finish()