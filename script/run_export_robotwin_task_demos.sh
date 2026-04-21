#!/usr/bin/bash

set -euo pipefail
set -x

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)

CONDA_BIN=${CONDA_BIN:-"conda"}
ENV_NAME=${ENV_NAME:-"lingbot"}
DATASET_ROOT=${DATASET_ROOT:-"${ROOT_DIR}/data/robotwin-clean-and-aug-lerobot/lerobot_robotwin_eef_aug_500"}
MODEL_PATH=${MODEL_PATH:-"${ROOT_DIR}/models/lingbot-va-posttrain-robotwin"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"${ROOT_DIR}/eval_out/robotwin_task_demos/$(date +%Y%m%d_%H%M%S)"}
# 当前 12 fps 导出下，6 chunks 通常会得到约 3.7 秒的小 demo。
NUM_CHUNKS=${NUM_CHUNKS:-"6"}
MAX_TASKS=${MAX_TASKS:-"0"}
SEGMENT_INDEX=${SEGMENT_INDEX:-"0"}
ATTN_MODE=${ATTN_MODE:-"flashattn"}
GUIDANCE_SCALE=${GUIDANCE_SCALE:-"1"}
ACTION_GUIDANCE_SCALE=${ACTION_GUIDANCE_SCALE:-"1"}
NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS:-"0"}
ACTION_NUM_INFERENCE_STEPS=${ACTION_NUM_INFERENCE_STEPS:-"0"}
TASK_NAMES=${TASK_NAMES:-""}

export TOKENIZERS_PARALLELISM=false

ARGS=(
    "${SCRIPT_DIR}/export_robotwin_task_demos.py"
    --dataset-root "${DATASET_ROOT}"
    --model-path "${MODEL_PATH}"
    --output-root "${OUTPUT_ROOT}"
    --num-chunks "${NUM_CHUNKS}"
    --max-tasks "${MAX_TASKS}"
    --segment-index "${SEGMENT_INDEX}"
    --attn-mode "${ATTN_MODE}"
    --guidance-scale "${GUIDANCE_SCALE}"
    --action-guidance-scale "${ACTION_GUIDANCE_SCALE}"
    --num-inference-steps "${NUM_INFERENCE_STEPS}"
    --action-num-inference-steps "${ACTION_NUM_INFERENCE_STEPS}"
)

if [ -n "${TASK_NAMES}" ]; then
    ARGS+=(--task-names "${TASK_NAMES}")
fi

# SSH 环境下直接固定到 lingbot 环境运行，避免缺包和 CUDA 版本错位。
"${CONDA_BIN}" run -n "${ENV_NAME}" python "${ARGS[@]}" "$@"
