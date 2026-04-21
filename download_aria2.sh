#!/bin/bash
# 使用 aria2 多线程下载数据集

DATA_DIR="/home/gjw/MyProjects/lingbot-va/data/robotwin-clean-and-aug-lerobot"
mkdir -p $DATA_DIR

echo "=== 使用 aria2 多线程下载 ==="
echo "数据目录: $DATA_DIR"
echo "开始时间: $(date)"
echo ""

# 先下载元数据文件
echo "[1/3] 下载元数据文件..."
aria2c -x 8 -s 8 -c -d $DATA_DIR \
    https://hf-mirror.com/datasets/robbyant/robotwin-clean-and-aug-lerobot/resolve/main/README.md \
    https://hf-mirror.com/datasets/robbyant/robotwin-clean-and-aug-lerobot/resolve/main/meta/info.json \
    https://hf-mirror.com/datasets/robbyant/robotwin-clean-and-aug-lerobot/resolve/main/meta/episodes.jsonl \
    https://hf-mirror.com/datasets/robbyant/robotwin-clean-and-aug-lerobot/resolve/main/meta/tasks.jsonl

# 创建目录结构
mkdir -p $DATA_DIR/videos/chunk-000/observation.images.cam_high
mkdir -p $DATA_DIR/latents/chunk-000/observation.images.cam_high

echo ""
echo "[2/3] 准备下载视频文件（共约 50GB）..."
echo "由于文件数量较多，建议使用 huggingface-cli 继续下载:"
echo ""
echo "  export HF_ENDPOINT=https://hf-mirror.com"
echo "  huggingface-cli download --repo-type dataset robbyant/robotwin-clean-and-aug-lerobot \\"
echo "      --local-dir $DATA_DIR \\"
echo "      --resume-download"
echo ""

echo "[3/3] 或者使用 git-lfs 克隆完整仓库:"
echo "  cd /home/gjw/MyProjects/lingbot-va/data"
echo "  git clone --depth=1 https://hf-mirror.com/datasets/robbyant/robotwin-clean-and-aug-lerobot"
echo ""

echo "完成时间: $(date)"
