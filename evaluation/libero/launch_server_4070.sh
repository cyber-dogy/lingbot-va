#!/usr/bin/env bash

set -euo pipefail

CONFIG_NAME=${CONFIG_NAME:-libero_4070}
PORT=${PORT:-29056}
MASTER_PORT=${MASTER_PORT:-29061}
SAVE_ROOT=${SAVE_ROOT:-outputs/libero_4070}

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

python -m torch.distributed.run \
  --nproc_per_node 1 \
  --master_port "${MASTER_PORT}" \
  wan_va/wan_va_server.py \
  --config-name "${CONFIG_NAME}" \
  --port "${PORT}" \
  --save_root "${SAVE_ROOT}"
