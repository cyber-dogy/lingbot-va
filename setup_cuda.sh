#!/bin/bash
# 在 lingbot 环境中安装 CUDA Toolkit

source $(conda info --base)/etc/profile.d/conda.sh
conda activate lingbot

echo "=== 安装 CUDA Toolkit 12.8 ==="
conda install -c nvidia/label/cuda-12.8.0 cuda-toolkit -y

echo "=== 设置环境变量 ==="
export CUDA_HOME=$CONDA_PREFIX
echo "export CUDA_HOME=\$CONDA_PREFIX" >> ~/.bashrc

echo "=== 验证 CUDA ==="
which nvcc
nvcc --version

echo "=== 重新安装 flash-attn ==="
pip install --upgrade pip setuptools wheel
TORCH_CUDA_ARCH_LIST="10.0" pip install flash-attn --no-build-isolation

echo "=== 完成 ==="
