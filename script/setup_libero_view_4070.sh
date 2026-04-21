#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

HOST_PYTHON=${HOST_PYTHON:-python}
PYTHON_BIN=${PYTHON_BIN:-/usr/bin/python3.10}
VENV_DIR=${VENV_DIR:-$HOME/.venvs/libero_view}
LIBERO_REPO=${LIBERO_REPO:-$HOME/MyProjects/LIBERO}

if ! command -v "${HOST_PYTHON}" >/dev/null 2>&1; then
  echo "Host python not found: ${HOST_PYTHON}" >&2
  exit 1
fi

if ! "${HOST_PYTHON}" -c "import virtualenv" >/dev/null 2>&1; then
  "${HOST_PYTHON}" -m pip install --user virtualenv
fi

"${HOST_PYTHON}" -m virtualenv -p "${PYTHON_BIN}" "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

python -m pip install -U pip setuptools wheel
python -m pip install -r "${REPO_ROOT}/envs/libero_view_requirements.txt"

if [ ! -f "${LIBERO_REPO}/setup.py" ]; then
  rm -rf "${LIBERO_REPO}"
  git clone --depth 1 https://github.com/Lifelong-Robot-Learning/LIBERO.git "${LIBERO_REPO}"
fi

python -m pip install -e "${LIBERO_REPO}"

cat <<EOF

libero_view is ready at ${VENV_DIR}

Activate:
  source "${VENV_DIR}/bin/activate"

Recommended render mode for visible windows:
  export MUJOCO_GL=glx
  unset PYOPENGL_PLATFORM

Still required once per machine with sudo:
  sudo apt update
  sudo apt install -y libegl1 libglfw3 libosmesa6-dev libgl1-mesa-glx patchelf libvulkan1 mesa-vulkan-drivers vulkan-tools

EOF
