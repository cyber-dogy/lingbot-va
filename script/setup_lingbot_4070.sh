#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

CONDA_HOME=${CONDA_HOME:-$HOME/miniconda3}
ENV_NAME=${ENV_NAME:-lingbot_4070}
LIBERO_REPO=${LIBERO_REPO:-$HOME/MyProjects/LIBERO}

source "${CONDA_HOME}/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda create -n "${ENV_NAME}" python=3.10 -y
fi

conda activate "${ENV_NAME}"

python -m pip install -U pip setuptools wheel
python -m pip install \
  torch==2.9.0 \
  torchvision==0.24.0 \
  torchaudio==2.9.0 \
  --index-url https://download.pytorch.org/whl/cu128

python -m pip install ninja packaging
if ! python -m pip install flash-attn --no-build-isolation; then
  echo "flash-attn install failed, keeping torch attention fallback." >&2
fi

python -m pip install -r "${REPO_ROOT}/envs/lingbot_4070_requirements.txt"
python -m pip install lerobot==0.3.3 --no-deps
python -m pip install -e "${REPO_ROOT}" --no-deps

if [ -f "${LIBERO_REPO}/setup.py" ]; then
  python -m pip install -e "${LIBERO_REPO}"
fi

cat <<EOF

${ENV_NAME} is ready.

Activate:
  source "${CONDA_HOME}/etc/profile.d/conda.sh"
  conda activate "${ENV_NAME}"

Recommended env vars for headless eval:
  export MUJOCO_GL=egl
  export PYOPENGL_PLATFORM=egl
  export MUJOCO_EGL_DEVICE_ID=0

EOF
