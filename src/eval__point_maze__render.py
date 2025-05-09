import random
import os
import json
import PIL
import argparse
import wandb

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import perf_counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.data import Dataset, DataLoader

import cv2
from shutil import rmtree
import gymnasium as gym
import gymnasium_robotics

from transformers import CLIPProcessor, CLIPModel

import logging
logging.getLogger('numpy').setLevel(logging.WARNING)

## Env variables
# For MuJoCo mujoco-3.3.0?
os.environ['MUJOCO_GL'] = 'egl'
# For Pyglet-based envs
os.environ['PYOPENGL_PLATFORM'] = 'egl'
# Optionally select GPU
os.environ['MUJOCO_EGL_DEVICE_ID'] = '0'

## Helper classes
class Config:
    def __init__(self, **kwargs):        
        ## Environment
        self.env_name = kwargs.get("env_name", "u_maze")
        self.seed_min = kwargs.get("seed_min", 0)
        self.seed_max = kwargs.get("seed_max", 1000)
        self.gamma = kwargs.get("gamma", 0.99)

        ## Architecture
        # Image embedding
        self.img_embed_dim = kwargs.get("img_embed_dim", 512)
        self.num_frames = kwargs.get("num_frames", 3) # Number of frames to stack for state
        self.frame_delta = kwargs.get("frame_delta", 5) # Number of frames to skip between consecutive frames
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
        # CLIP
        self.clip_checkpoint = kwargs.get("clip_checkpoint", "openai/clip-vit-base-patch32")
        
        ## Evaluation
        self.device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.max_episode_length = kwargs.get("max_episode_length", 500)
        self.max_log_std = kwargs.get("max_log_std", 5.0) 
        self.min_log_std = kwargs.get("min_log_std", -5.0)
        self.num_eval_episodes = kwargs.get("num_eval_episodes", 100)
        
        self.txt_out_path = kwargs.get("txt_out_path", "out/evals/pointmaze_run1.txt")
        self.video_out_path = kwargs.get("video_out_path", "eval_trajs/pointmaze_run1")
        self.checkpoint_in_dir = kwargs.get("checkpoint_in_dir", "out/checkpoints/run1")

class EnvSpecs:
    def __init__(
        self, 
        obs_width,
        obs_height,
        action_low, 
        action_high,
        action_dim,
        goal_state,
        goal_cell,
    ):
        # Core inputs
        self.obs_width = obs_width
        self.obs_height = obs_height
        self.action_low = action_low
        self.action_high = action_high
        self.action_dim = action_dim
        self.goal_state = goal_state
        self.goal_cell = goal_cell

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

        in_dim = config.img_embed_dim * config.num_frames + config.img_embed_dim # state (N frames) + goal (1 frame)
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
        x1 = torch.cat((state, goal), dim=-1) # (B, in_dim)
        
        # Pass through the network
        x2 = self.relu_1(self.fc_1(x1))
        x3 = self.relu_2(self.fc_2(x2))
        x = self.fc_out(x3)
        log_std = torch.clamp(self.fc_log_std(x3), self.min_log_std, self.max_log_std)
        
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
            ratio = (upper - lower) / 2    

        return action, log_prob, ratio

# Critic classes
class StateActionEncoder(nn.Module):
    def __init__(self, config, env_specs):
        super(StateActionEncoder, self).__init__()
        
        in_dim = config.img_embed_dim * config.num_frames + env_specs.action_dim # state (N frames) + action
        out_dim = config.embed_dim
        
        # StateActionEncoder network
        self.fc_1 = nn.Linear(in_dim, config.sae_dim1)
        self.relu_1 = nn.ReLU()
        self.fc_2 = nn.Linear(config.sae_dim1, config.sae_dim2)
        self.relu_2 = nn.ReLU()
        self.fc_out = nn.Linear(config.sae_dim2, out_dim)
        
    def forward(self, state, action):
        x1 = torch.cat((state, action), dim=-1) # (B, in_dim)
        # Pass through the network
        x2 = self.relu_1(self.fc_1(x1))
        x3 = self.relu_2(self.fc_2(x2))
        
        h = self.fc_out(x3)

        return h

class GoalEncoder(nn.Module):
    def __init__(self, config, env_specs):
        super(GoalEncoder, self).__init__()
        
        in_dim = config.img_embed_dim
        out_dim = config.embed_dim
        # GoalEncoder network
        self.fc_1 = nn.Linear(in_dim, config.ge_dim1)
        self.relu_1 = nn.ReLU()
        self.fc_2 = nn.Linear(config.ge_dim1, config.ge_dim2)
        self.relu_2 = nn.ReLU()
        self.fc_out = nn.Linear(config.ge_dim2, out_dim)
        
    def forward(self, state):
        x1 = state # (B, in_dim)
        
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
        self.num_frames = config.num_frames # Store for use in sampling
        self.frame_delta = config.frame_delta
        self.ptr = 0
        self.size = 0
        
        self.state = torch.empty((self.max_size, config.img_embed_dim), dtype=torch.float32)
        self.action = torch.empty((self.max_size, env_specs.action_dim), dtype=torch.float32)
        self.done = torch.empty((self.max_size, 1), dtype=torch.float32)

    def get_state_dict(self):
        state_dict = {
            "state": self.state,
            "action": self.action,
            "done": self.done,
            "ptr": self.ptr, # Need ptr to continue adding correctly
            "size": self.size,
        }
        return state_dict
    
    def load_state_dict(self, state_dict):
        self.state = state_dict["state"]
        self.action = state_dict["action"]
        self.done = state_dict["done"]
        self.ptr = state_dict.get("ptr", 0) # Load ptr
        self.size = state_dict.get("size", self.max_size) # Load size

    def add(self, state, action, done):
        # Stores single-frame embedding
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
    
    def _get_multiframe_state(self, index):
        frames = []
        target_embedding = self.state[index] # Use this for padding

        for i in range(self.num_frames):
            current_idx = (index - i * self.frame_delta + self.size) % self.size
            if i > 0: # It's OK if the done flag is set for the last frame
                # If the idx is negative but we haven't even filled the buffer yet: our trajectory starts here
                # If the done flag is set here, this frame and everything before it is part of a different trajectory
                if (index - i * self.frame_delta < 0 and self.size < self.max_size):
                    break
                # If the done flag is set for idx, idx+1, ..., idx+frame_delta-1, this frame and everything before it is part of a different trajectory
                if current_idx + self.frame_delta <= self.size:
                    any_done = self.done[current_idx:current_idx + self.frame_delta].sum() > 0
                else:
                    to_idx = (current_idx + self.frame_delta) % self.size
                    any_done = self.done[current_idx:self.size].sum() + self.done[:to_idx].sum() > 0
                if any_done:
                    break
            frames.insert(0, self.state[current_idx])

        # Concatenate frames: order should be [oldest, ..., newest]
        # Pad left with the target embedding if we don't have enough frames
        while len(frames) < self.num_frames:
            frames.insert(0, target_embedding)
        concatenated_state = torch.cat(frames, dim=-1)
        return concatenated_state

    def sample_single_entry_indices(self):
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
        
        return idx1, idx2
                
    def sample_for_actor(self, batch_size, device='cpu'):
        # Construct multi-frame states and single-frame goals
        states = []
        goals = []
        for _ in range(batch_size):
            idx1, idx2 = self.sample_single_entry_indices()
            multi_frame_state = self._get_multiframe_state(idx1)
            goal_state = self.state[idx2] # Goal is single frame
            states.append(multi_frame_state)
            goals.append(goal_state)

        return torch.stack(states).to(device), torch.stack(goals).to(device)
    
    def sample_for_critic(self, batch_size, device='cpu'):
        # Construct multi-frame states and single-frame goals
        states = []
        actions = []
        goals = []
        for _ in range(batch_size):
            idx1, idx2 = self.sample_single_entry_indices()
            multi_frame_state = self._get_multiframe_state(idx1)
            action = self.action[idx1]
            goal_state = self.state[idx2] # Goal is single frame
            states.append(multi_frame_state)
            actions.append(action)
            goals.append(goal_state)

        return torch.stack(states).to(device), torch.stack(actions).to(device), torch.stack(goals).to(device)

## Code starts here
def parse_args_and_config():
    parser = argparse.ArgumentParser(description="Evaluate SGCRL Agent")

    # Environment args
    parser.add_argument('--env_name', type=str, default="u_maze", help='PointMaze environment name')
    parser.add_argument('--seed_min', type=int, default=42, help='Minimum random seed')
    parser.add_argument('--seed_max', type=int, default=10000, help='Maximum random seed')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    
    # Architecture args
    parser.add_argument('--img_embed_dim', type=int, default=512, help='Image embedding dimension')
    parser.add_argument('--num_frames', type=int, default=3, help='Number of consecutive frames to use for state representation')
    parser.add_argument('--frame_delta', type=int, default=5, help='Number of frames to skip between consecutive frames')
    parser.add_argument('--actor_dim1', type=int, default=256, help='Actor hidden layer 1 size')
    parser.add_argument('--actor_dim2', type=int, default=256, help='Actor hidden layer 2 size')
    parser.add_argument('--embed_dim', type=int, default=64, help='Common embedding dimension for critic encoders')
    parser.add_argument('--sae_dim1', type=int, default=256, help='StateActionEncoder hidden layer 1 size')
    parser.add_argument('--sae_dim2', type=int, default=256, help='StateActionEncoder hidden layer 2 size')
    parser.add_argument('--ge_dim1', type=int, default=256, help='GoalEncoder hidden layer 1 size')
    parser.add_argument('--ge_dim2', type=int, default=256, help='GoalEncoder hidden layer 2 size')
    parser.add_argument("--clip_checkpoint", type=str, default="openai/clip-vit-base-patch32")

    # Evaluation args
    parser.add_argument('--max_episode_length', type=int, default=500, help='Maximum steps per episode')
    parser.add_argument('--max_log_std', type=float, default=5.0, help='Maximum log standard deviation for actor')
    parser.add_argument('--min_log_std', type=float, default=-5.0, help='Minimum log standard deviation for actor')
    parser.add_argument('--num_eval_episodes', type=int, default=100, help='Number of episodes to evaluate')
    
    parser.add_argument('--txt_out_path', type=str, default="out/evals/pointmaze_run1.txt", help='Directory to save eval info')
    parser.add_argument('--tag', type=str, default=None, help='Tag to add to the output file')
    parser.add_argument('--video_out_path', type=str, default="out/eval_trajs/pointmaze_run1", help='Directory to save trajectory videos')
    parser.add_argument('--checkpoint_in_dir', type=str, default="out/checkpoints/pointmaze_run1", help='Directory to save checkpoints')

    # Special args
    parser.add_argument('--no_delta', action='store_true', help='Whether to use delta')
    

    args = parser.parse_args()
    if args.no_delta:
        args.frame_delta = 1
    if args.tag is not None:
        args.txt_out_path = f"eval_out/{args.tag}/metrics_{args.env_name}.txt"
        args.video_out_path = f"eval_out/{args.tag}/video_{args.env_name}"

        os.makedirs(args.video_out_path, exist_ok=True)

    # Create Config object from parsed args
    config_kwargs = vars(args)

    # Handle dependent defaults
    config_kwargs['device'] = "cuda" if torch.cuda.is_available() else "cpu"

    config = Config(**config_kwargs)

    # Create necessary directories
    os.makedirs(config.video_out_path, exist_ok=True)

    return config

def get_u_maze():
    u_maze = [
        [1, 1, 1, 1, 1],
        [1, "g", 0, 0, 1],
        [1, 1, 1, 0, 1],
        [1, "r", 0, 0, 1],
        [1, 1, 1, 1, 1]
    ]
    env = gym.make("PointMaze_UMaze-v3", maze_map=u_maze, render_mode='rgb_array')
    env.unwrapped.model.vis.global_.fovy = 60.0
    goal_kwargs = {
        "goal_cell": np.array([1, 1.3], dtype=float),
        "reset_cell": np.array([0.8, 0.85], dtype=float),
    }
    return env, goal_kwargs

def make_medium_maze():
    medium_maze = [
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, "r", 0, 1, 1, 0, 0, 1],
        [1, 0, 0, 1, 0, 0, 0, 1],
        [1, 1, 0, 0, 0, 1, 1, 1],
        [1, 0, 0, 1, 0, 0, 0, 1],
        [1, 0, 1, 0, 0, 1, 0, 1],
        [1, 0, 0, 0, 1, "g", 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1]
    ]
    env = gym.make("PointMaze_UMaze-v3", maze_map=medium_maze, render_mode='rgb_array')
    env.unwrapped.model.vis.global_.fovy = 60.0
    goal_kwargs = {
        "goal_cell": np.array([6, 5.3], dtype=float),
        "reset_cell": np.array([5.9, 4.9], dtype=float),
    }
    return env, goal_kwargs

def make_large_maze():
    large_maze = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, "r", 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
        [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
        [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
        [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
        [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, "g", 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ]
    env = gym.make("PointMaze_UMaze-v3", maze_map=large_maze, render_mode='rgb_array')
    env.unwrapped.model.vis.global_.fovy = 60.0
    goal_kwargs = {
        "goal_cell": np.array([7, 9.7], dtype=float),
        "reset_cell": np.array([6.9, 10.2], dtype=float),
    }
    return env, goal_kwargs

def get_maze(maze_name):
    # Dense, because it's easier to interpret rewards (we don't train on them anyway)
    if maze_name == "u_maze":
        return get_u_maze()
    elif maze_name == "medium_maze":
        return make_medium_maze()
    elif maze_name == "large_maze":
        return make_large_maze()
    else:
        raise ValueError(f"Unknown maze name: {maze_name}")

def get_env_specs(
    config,
):
    env, goal_kwargs = get_maze(config.env_name)

    # Get action space
    action_space = env.action_space
    action_low = torch.from_numpy(action_space.low)
    action_high = torch.from_numpy(action_space.high)
    action_dim = action_low.shape[0]
    # Get goal obs (render)
    env.reset(options=goal_kwargs)
    render = env.render()
    assert len(render.shape) == 3 and render.shape[2] == 3, "Goal render must be an RGB image"
    height, width = render.shape[0], render.shape[1]
    goal_obs = encode_image(render, config).cpu()

    return env, EnvSpecs(
        action_low=action_low,
        action_high=action_high,
        action_dim=action_dim,
        obs_height=height,
        obs_width=width,
        goal_state=goal_obs,
        goal_cell=goal_kwargs["goal_cell"],
    )

def encode_image(
    image, 
    config,
):
    image = PIL.Image.fromarray(image)
    image = image.resize((224, 224), resample=PIL.Image.Resampling.BICUBIC)
    inputs = config.processor(
        images=image, 
        return_tensors="pt",
    ).to(config.device)
    embeddings = config.model.get_image_features(**inputs)[0]
    return embeddings

def load_checkpoint(
    checkpoint_path,
    policy,
    config,
):
    # Load the checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    policy.load_state_dict(torch.load(os.path.join(checkpoint_path, f"policy_0.pth")))
    print(f"Loaded checkpoint from {checkpoint_path}")

@torch.no_grad()
def roll_out_trajectory(
    env,
    policy,
    env_specs,
    config,
    video_out_path=None,
    success_video_out_path=None,
):
    device = config.device
    frames = []
    
    # Roll out a trajectory and add it to the replay buffer
    env.reset(options={"goal_cell": env_specs.goal_cell})
    obs = env.render()
    obs = encode_image(obs, config).to(device=device, dtype=torch.float32)
    goal_state = env_specs.goal_state.to(device=device, dtype=torch.float32)
    
    done = False
    discounted_reward_sum = 0.
    gamma = config.gamma 
    n = 0
    success = 0

    # Need frames with indices 0, delta, 2*delta, ..., (num_frames-1)*delta
    # That is, we need to keep 1 + (num_frames-1) * frame_delta frames around
    # Keeping more around is fine, but keeping less will cause errors
    num_frames = config.frame_delta * (config.num_frames - 1) + 1
    last_k_obs = [obs for _ in range(num_frames)]

    for t in range(config.max_episode_length):
        # Oldest to newest, stride = frame_delta
        relevant_frames = last_k_obs[::config.frame_delta]
        concat_obs = torch.cat(relevant_frames, dim=-1)
        action, neg_log_prob, ratio = policy(concat_obs, goal_state)

        n += 1
        _, reward, done, truncate, info = env.step(action.cpu().numpy().flatten())
        done = done or truncate
        frame = env.render()
        next_obs = encode_image(frame, config).to(device=device, dtype=torch.float32)
        done = torch.FloatTensor([done]).to(device)

        if info.get("success", False):
            success = 1.
        
        if video_out_path is not None:
            frames.append(frame)

        # Add the transition to the replay buffer
        if t == config.max_episode_length - 1:
            done = torch.FloatTensor([True]).to(device)

        # Update the last k observations
        last_k_obs.pop(0)
        last_k_obs.append(next_obs)        
        discounted_reward_sum += reward * (gamma ** t)
        
        if done:
            break

    if success_video_out_path is not None and success:
        video_out_path = success_video_out_path
    if video_out_path is not None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use appropriate codec for your format
        fps = 30  # Frames per second
        frame_width, frame_height = frames[0].shape[1], frames[0].shape[0]
        out = cv2.VideoWriter(video_out_path, fourcc, fps, (frame_width, frame_height))
        for frame in frames:
            out.write(frame)
        out.release()
            
    return discounted_reward_sum, success

@torch.no_grad()
def eval_policy(
    policy,
    env,
    env_specs,
    config,
):
    env.reset(options={"goal_cell": env_specs.goal_cell})
    policy.eval()
    policy.to(config.device)

    load_checkpoint(config.checkpoint_in_dir, policy, config)

    rewards = []
    successes = []

    cur_seed = config.seed_min
    for i in tqdm(range(config.num_eval_episodes), desc="Evaluating policy"):
        torch.manual_seed(cur_seed)
        np.random.seed(cur_seed)

        video_out_path = f"{config.video_out_path}/video_{i+1}.mp4"
        success_video_out_path = f"{config.video_out_path}/success_video_{i+1}.mp4"
        discounted_reward_sum, success = roll_out_trajectory(
            env,
            policy,
            env_specs,
            config,
            video_out_path,
            success_video_out_path,
        )
        rewards.append(discounted_reward_sum)
        successes.append(success)

        cur_seed += 1
        if cur_seed > config.seed_max:
            cur_seed = config.seed_min

    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    mean_success = np.mean(successes)
    std_success = np.std(successes)

    with open(config.txt_out_path, "w+") as f:
        f.write(f"Mean reward: {mean_reward:.2f}\nStd reward: {std_reward:.2f}\nMean success: {mean_success:.2f}\nStd success: {std_success:.2f}\n")
        f.write("-"*10 + "\nIndividual:")
        for i in range(len(rewards)):
            f.write(f"\nEpisode {i+1}, r: {rewards[i]:.2f}, s: {successes[i]}")
        f.write("\n" + "-"*10 + "\n")

def main():
    print("[+] Parsing Arguments and Config")
    config = parse_args_and_config()

    config.processor = CLIPProcessor.from_pretrained(config.clip_checkpoint)
    config.model = CLIPModel.from_pretrained(config.clip_checkpoint, device_map=config.device)

    print(f"[*] Using device: {config.device}")
    print(f"[*] Using environment: {config.env_name}")

    print("[+] Getting environment")
    env, specs = get_env_specs(config)
    
    print("[+] Getting networks...")
    policy = Actor(config, specs)
    state_action_encoder = StateActionEncoder(config, specs)
    goal_encoder = GoalEncoder(config, specs)
    
    print("[+] Evaluating policy...")
    eval_policy(policy, env, specs, config)

if __name__ == "__main__":
    main()  
