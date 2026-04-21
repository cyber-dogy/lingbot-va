#!/usr/bin/env bash

set -euo pipefail

START=${START:-0}
END=${END:-1}
PORT=${PORT:-29056}
TEST_NUM=${TEST_NUM:-1}
LIBERO_BENCHMARK=${LIBERO_BENCHMARK:-libero_spatial}
OUT_DIR=${OUT_DIR:-outputs/libero}

python evaluation/libero/client.py \
  --libero-benchmark "${LIBERO_BENCHMARK}" \
  --port "${PORT}" \
  --test-num "${TEST_NUM}" \
  --task-range "${START}" "${END}" \
  --out-dir "${OUT_DIR}"
