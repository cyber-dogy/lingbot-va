#!/bin/bash
# LingBot-VA 模型和数据下载脚本

set -e

DATA_ROOT="/home/gjw/MyProjects/lingbot-va"
MODEL_DIR="$DATA_ROOT/models"
DATA_DIR="$DATA_ROOT/data"

echo "=== 创建目录 ==="
mkdir -p $MODEL_DIR $DATA_DIR

echo "=== 安装下载工具 ==="
source $(conda info --base)/etc/profile.d/conda.sh
conda activate lingbot
pip install -q huggingface-hub[cli] modelscope

echo ""
echo "=== 下载 lingbot-va-base 预训练模型 (~14GB) ==="
huggingface-cli download robbyant/lingbot-va-base \
    --local-dir $MODEL_DIR/lingbot-va-base \
    --local-dir-use-symlinks False \
    --resume-download

echo ""
echo "=== 下载完成 ==="
echo "模型目录: $MODEL_DIR"
ls -lh $MODEL_DIR/
