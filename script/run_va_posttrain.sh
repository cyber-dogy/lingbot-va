#!/usr/bin/bash

set -euo pipefail
set -x

umask 007
 
NGPU=${NGPU:-"1"}
MASTER_PORT=${MASTER_PORT:-"29501"}
PORT=${PORT:-"1106"}
LOG_RANK=${LOG_RANK:-"0"}
TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-"http://localhost:29510"}
CONFIG_NAME=${CONFIG_NAME:-"robotwin_train"} # robotwin_train, libero_train
ENABLE_WANDB=${ENABLE_WANDB:-"0"}
TASK_NAME=${TASK_NAME:-""}
SAVE_ROOT=${SAVE_ROOT:-""}
DATASET_PATH=${DATASET_PATH:-""}
EMPTY_EMB_PATH=${EMPTY_EMB_PATH:-""}
LOAD_WORKER=${LOAD_WORKER:-""}
NUM_STEPS=${NUM_STEPS:-""}
SAVE_INTERVAL=${SAVE_INTERVAL:-""}
RESUME_FROM=${RESUME_FROM:-""}
PRETRAINED_MODEL_PATH=${PRETRAINED_MODEL_PATH:-""}
OPTIMIZER_TYPE=${OPTIMIZER_TYPE:-""}
SMOKE_MODE=${SMOKE_MODE:-""}

## node setting
num_gpu=${NGPU}
master_port=${MASTER_PORT}
log_rank=${LOG_RANK}
torchft_lighthouse=${TORCHFT_LIGHTHOUSE}
config_name=${CONFIG_NAME}

## cmd setting
export TOKENIZERS_PARALLELISM=false
train_args=(
    --config-name "${config_name}"
)

if [ -n "${SAVE_ROOT}" ]; then
    train_args+=(--save-root "${SAVE_ROOT}")
fi
if [ -n "${TASK_NAME}" ]; then
    train_args+=(--dataset-repo-name "${TASK_NAME}")
fi
if [ -n "${DATASET_PATH}" ]; then
    train_args+=(--dataset-path "${DATASET_PATH}")
fi
if [ -n "${EMPTY_EMB_PATH}" ]; then
    train_args+=(--empty-emb-path "${EMPTY_EMB_PATH}")
fi
if [ -n "${LOAD_WORKER}" ]; then
    train_args+=(--load-worker "${LOAD_WORKER}")
fi
if [ -n "${NUM_STEPS}" ]; then
    train_args+=(--num-steps "${NUM_STEPS}")
fi
if [ -n "${SAVE_INTERVAL}" ]; then
    train_args+=(--save-interval "${SAVE_INTERVAL}")
fi
if [ -n "${RESUME_FROM}" ]; then
    train_args+=(--resume-from "${RESUME_FROM}")
fi
if [ -n "${PRETRAINED_MODEL_PATH}" ]; then
    train_args+=(--pretrained-model-path "${PRETRAINED_MODEL_PATH}")
fi
if [ -n "${OPTIMIZER_TYPE}" ]; then
    train_args+=(--optimizer-type "${OPTIMIZER_TYPE}")
fi
if [ -n "${SMOKE_MODE}" ]; then
    train_args+=(--smoke-mode "${SMOKE_MODE}")
fi

if [ "${ENABLE_WANDB}" = "1" ]; then
    export WANDB_BASE_URL=${WANDB_BASE_URL:-"https://api.wandb.ai"}
    export WANDB_TEAM_NAME=${WANDB_TEAM_NAME:-"cyber-dogy"}
    export WANDB_PROJECT=${WANDB_PROJECT:-"lingbot-va"}
    train_args+=(--enable-wandb true)
else
    train_args+=(--enable-wandb false)
fi

if [ $# -ne 0 ]; then
    train_args+=("$@")
fi

PYTORCH_ALLOC_CONF="expandable_segments:True" TORCHFT_LIGHTHOUSE=${torchft_lighthouse} \
python -m torch.distributed.run \
    --nproc_per_node=${num_gpu} \
    --local-ranks-filter=${log_rank} \
    --master_port ${master_port} \
    --tee 3 \
    -m wan_va.train "${train_args[@]}"
