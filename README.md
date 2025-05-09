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

This section should contain instructions to set a pip environment up once we fill out `requirements.txt`.

## Repository structure
The code for each `(environment, input_type)` pair is provided in the `src/` directory in its own file. 
We wrote each such setting in a single large file, though these files themselves should be quite readable.
The evaluation scripts have been branched off into their own files.

We provide selected checkpoints of successfully learned policies under `data/checkpoints/`, and corresponding videos in `data/videos/`. Unfortunately, we cannot release the entire set of checkpoints and videos over training due to size considerations. 

Within `run_scripts/`, we provide some bash scripts that should make running training and evaluation easier. Please refer to them for a examples of how to call the python files.

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
- [ ] Copy code over from our files.
- [ ] Clean the code to remove unused variables, etc., and add comments.
- [ ] Add a `requirements.txt` file.
- [ ] Write the report and upload it.
- [ ] Make the respository public.
- [ ] Remove this checklist.