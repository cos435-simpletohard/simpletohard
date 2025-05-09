#!/bin/bash -l

MAZE_TYPE="large" # "u", "medium", or "large"
EP_LENGTH=400 # 300 is enough for "u"; 400 for "medium" and "large"

N_ACTORS=1
ENV_NAME="${MAZE_TYPE}_maze" 
TAG="point_maze__render__${MAZE_TYPE}" 
VIDEO_OUT_PATH="data/videos/${TAG}__${N_ACTORS}"
CHECKPOINT_DIR="data/checkpoints/${TAG}__${N_ACTORS}"
WANDB_DIR="data/wandb/${TAG}__${N_ACTORS}"
SEED=42

ACTOR_MICRO_BATCH_SIZE=256
ACTOR_NUM_MICRO_STEPS=64
ACTOR_BATCH_SIZE=$(($ACTOR_MICRO_BATCH_SIZE * $ACTOR_NUM_MICRO_STEPS))  

CRITIC_MICRO_BATCH_SIZE=256
CRITIC_NUM_MICRO_STEPS=64 
CRITIC_BATCH_SIZE=$(($CRITIC_MICRO_BATCH_SIZE * $CRITIC_NUM_MICRO_STEPS))

cmd="python src/train__point_maze__render.py \
    --num_actors ${N_ACTORS} \
    --video_out_path ${VIDEO_OUT_PATH} \
    --checkpoint_dir ${CHECKPOINT_DIR} \
    --wandb_dir ${WANDB_DIR} \
    --actor_batch_size ${ACTOR_BATCH_SIZE} \
    --actor_micro_batch_size ${ACTOR_MICRO_BATCH_SIZE} \
    --critic_batch_size ${CRITIC_BATCH_SIZE} \
    --critic_micro_batch_size ${CRITIC_MICRO_BATCH_SIZE} \
    --seed ${SEED} \
    --min_log_std -5 \
    --env_name ${ENV_NAME} \
    --max_episode_length ${EP_LENGTH}"

echo $cmd
$cmd