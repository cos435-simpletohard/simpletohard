# Simple-to-hard generalization in RL

This repository contains the code we wrote to 
- reimplement the single goal-conditioned contrastive RL paper [1] for the [sawyer box](https://github.com/Farama-Foundation/Metaworld) environment. 
- implement a version that acts on renders of the scene, and not coordinates---and extend our setting to include the [point maze](https://robotics.farama.org/envs/maze/point_maze/) environment.
- test whether policies trained on renders of a certain maze can work on other mazes (answer: no).

The code of the original paper can be found [here](https://github.com/graliuce/sgcrl). Our implementation is from scratch, though we took goal coordinates from the original.

We are grateful to the staff of COS 435 (Ben and the TAs) for their help.

## Team members
- Adithya Bhaskar
- Katie Heller
- Laura Hwa

## Environment

You might want to create a new conda environment with Python 3.10 before running this code.
Having done so, you can run
```
pip install -r requirements.txt
```
to install most of the requirements. You will have to install the Metaworld package from its [github repository](https://github.com/Farama-Foundation/Metaworld), though (there are instructions in the linked repository).
Finally, you might have to apply the patch below to get training to work.

### A hacky patch
In the point maze environments, we pass in floating-point coordinates while resetting the environment.
We found that this was needed, as otherwise the point and goal would be rendered on top of each other in the desired final state (and this would mess with the CLIP embeddings). You will therefore see something like
```
goal_kwargs = {
    "goal_cell": np.array([7, 9.7], dtype=float),
    "reset_cell": np.array([6.9, 10.2], dtype=float),
}
```
in our code.
Point Maze itself does not care that the coordinates be integers (it renders the items anyway), but one of its subclasses has an annoying check that the coordinates be integers. If you run our code out-of-the box, you might get an error in `.../gymnasium-robotics/envs/maze/maze_v4.py:318`. You can fix this error by commenting out these two pairs of lines in that file (lines 317-320 and 334-339)
```
                ...
                # assert (
                #     self.maze.maze_map[options["goal_cell"][0]][options["goal_cell"][1]]
                #     != 1
                # ), f"Goal can't be placed in a wall cell, {options['goal_cell']}"
                ...
                # assert (
                #     self.maze.maze_map[options["reset_cell"][0]][
                #         options["reset_cell"][1]
                #     ]
                #     != 1
                # ), f"Reset can't be placed in a wall cell, {options['reset_cell']}"
                ...
```
Optionally, we found that not adding noise to the initial positions led to better training due to consistent start states (lines 328 and 349 of the same file):
```
        ...
            # self.goal = self.add_xy_position_noise(goal)
            self.goal = goal
        ...
        # self.reset_pos = self.add_xy_position_noise(reset_pos)
        self.reset_pos = reset_pos
        ...
```
It would be preferable to subclass `Maze`, but since `PointMazeEnv` derives from it, that becomes a bit messy. Therefore, we chose this simpler, if hacky, fix.

## Repository structure
The code for each `(environment, input_type)` pair is provided in the `src/` directory in its own file. 
We wrote each such setting in a single large file, though these files themselves should be quite readable.
There is also an evaluation script for easy evaluation of point maze checkpoints on other mazes.

We provide selected checkpoints of learned policies under `data/checkpoints/`, and corresponding videos in `data/videos/`. 
Unfortunately, we cannot release the entire set of checkpoints and videos over training due to size considerations. 
We also had to remove the checkpoint of the replay buffer from each checkpoint directory since it took up over a gigabyte of storage.

Within `scripts/`, we provide some bash scripts that should make running training and evaluation easier. Please refer to them for a examples of how to call the python files.

## TL; DR 
Goal-conditioned policies don't fit a reward; they fit environment dynamics. 
In principle, if the dynamics of two environments are similar, a goal-conditioned policy learned on one should obtain nontrivial performance on the other.
We test this hypothesis on the Point Maze environment with the SGCRL algorithm (after reproducing SGCRL on Sawyer Box).
To unify the representations of various mazes, we rely on CLIP embeddings of scene renders rather than ball/goal coordinates.
We find that policies learned on one maze do not transfer to other mazes: there is room for future work here.

## References

[1] Liu, G., Tang, M., & Eysenbach, B. (2025). *A Single Goal is All You Need: Skills and Exploration Emerge from Contrastive RL without Rewards, Demonstrations, or Subgoals*. In International Conference on Learning Representations (ICLR) 2025. Retrieved from https://openreview.net/forum?id=xCkgX4Xfu0 on May 8, 2025.

## TODO checklist

- [X] Write a skeleton of the README.
- [X] Copy code over from our files.
- [X] Clean the code to remove unused variables, etc., and add comments.
- [X] Add a `requirements.txt` file.
- [X] Make sure that everything runs; do one quick dryrun to confirm.
- [X] Make the respository public.
- [ ] Remove this checklist and upload the report to our Ed post + submit to gradescope.