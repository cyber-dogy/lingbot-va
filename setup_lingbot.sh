#!/bin/bash
set -e

echo "=== 创建 lingbot 环境（不打扰后台训练）==="
conda create -n lingbot python=3.10.16 -y

echo "=== 激活环境 ==="
source $(conda info --base)/etc/profile.d/conda.sh
conda activate lingbot

echo "=== 安装 PyTorch 2.9.0 + CUDA 12.8 (RTX 5090 支持) ==="
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 \
    --index-url https://download.pytorch.org/whl/cu128

echo "=== 安装基础依赖 ==="
pip install websockets einops diffusers==0.36.0 transformers==4.55.2 \
    accelerate msgpack opencv-python matplotlib ftfy easydict \
    tqdm imageio[ffmpeg] Pillow safetensors scipy wandb numpy==1.26.4

echo "=== 安装后训练依赖 ==="
pip install lerobot==0.3.3 --no-deps

echo "=== 安装 flash-attn ==="
pip install --upgrade pip setuptools wheel
export TORCH_CUDA_ARCH_LIST="10.0"
pip install flash-attn --no-build-isolation

echo "=== 验证安装 ==="
python -c "import torch; print(f'✓ PyTorch: {torch.__version__}'); print(f'✓ CUDA: {torch.version.cuda}'); print(f'✓ GPU: {torch.cuda.get_device_name(0)}')"

echo "=== 完成！使用 'conda activate lingbot' 激活环境 ==="