#!/bin/bash -l

N_ACTORS=1
TAG="sawyer_box__coordinates"
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

cmd="python src/train__sawyer_box__coordinates.py \
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
    --initial_env_delay 30"

echo $cmd
$cmd