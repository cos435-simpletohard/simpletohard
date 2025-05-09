#!/bin/bash -l

EVAL_ENV_TYPE="u" # "u", "medium", or "large"
EVAL_ENV_NAME="${EVAL_ENV_TYPE}_maze"
CHECKPOINT_DIR="data/checkpoints/point_maze__render__u__20k" # Can be different from the eval env type

OUT_DIR="${CHECKPOINT_DIR}/eval"
OUT_VIDEO_DIR="${OUT_VIDEO_DIR}/videos"
OUT_RESULTS_FILE="${OUT_DIR}/results.txt"

mkdir -p ${OUT_VIDEO_DIR}

cmd="python src/eval__point_maze__render.py \
    --checkpoint_in_dir ${CHECKPOINT_DIR} \
    --video_out_path ${OUT_VIDEO_DIR} \
    --txt_out_path ${OUT_RESULTS_FILE} \
    --tag ${EVAL_ENV_TYPE}"

echo $cmd
$cmd