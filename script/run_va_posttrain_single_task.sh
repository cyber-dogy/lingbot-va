#!/usr/bin/bash

set -euo pipefail
set -x

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
DATA_ROOT="${ROOT_DIR}/data/robotwin-clean-and-aug-lerobot/lerobot_robotwin_eef_aug_500"

TASK_NAME=${TASK_NAME:-""}
if [ -z "${TASK_NAME}" ]; then
    echo "TASK_NAME is required, for example: TASK_NAME=beat_block_hammer-aloha-agilex_randomized_500-1000"
    exit 1
fi

TASK_PATH="${DATA_ROOT}/${TASK_NAME}"
if [ ! -d "${TASK_PATH}" ]; then
    echo "Task repo not found: ${TASK_PATH}"
    exit 1
fi

export NGPU=${NGPU:-"1"}
export CONFIG_NAME=${CONFIG_NAME:-"robotwin_train"}
export ENABLE_WANDB=${ENABLE_WANDB:-"1"}
export TASK_NAME
export DATASET_PATH="${ROOT_DIR}/data/robotwin-clean-and-aug-lerobot"
export EMPTY_EMB_PATH="${ROOT_DIR}/data/robotwin-clean-and-aug-lerobot/empty_emb.pt"
export LOAD_WORKER=${LOAD_WORKER:-"4"}
export NUM_STEPS=${NUM_STEPS:-"2000"}
export SAVE_INTERVAL=${SAVE_INTERVAL:-"200"}
export SAVE_ROOT=${SAVE_ROOT:-"${ROOT_DIR}/train_out/single_task/${TASK_NAME}"}
export WANDB_TEAM_NAME=${WANDB_TEAM_NAME:-"cyber-dogy"}
export WANDB_PROJECT=${WANDB_PROJECT:-"lingbot-va-single-task"}

# 单任务实验默认直接复用当前配置里的官方权重初始化，不额外改 checkpoint 目录。
bash "${SCRIPT_DIR}/run_va_posttrain.sh" "$@"
