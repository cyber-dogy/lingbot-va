#!/bin/bash
source $(conda info --base)/etc/profile.d/conda.sh
conda activate lingbot

# 国内镜像设置
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=1

DATA_DIR="/home/gjw/MyProjects/lingbot-va/data/robotwin-clean-and-aug-lerobot"
mkdir -p $DATA_DIR

echo "=== 使用 hf-mirror.com 国内镜像下载 ==="
echo "开始时间: $(date)"

# 使用新的 hf 命令（旧版 fallback 到 huggingface-cli）
hf download --repo-type dataset robbyant/robotwin-clean-and-aug-lerobot \
    --local-dir $DATA_DIR \
    --resume-download 2>/dev/null || \
huggingface-cli download --repo-type dataset robbyant/robotwin-clean-and-aug-lerobot \
    --local-dir $DATA_DIR \
    --resume-download

echo "完成时间: $(date)"
echo "数据路径: $DATA_DIR"
