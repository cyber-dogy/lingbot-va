#!/bin/bash
# Robotwin 数据集下载守护脚本 - 仅下载 10 个指定任务
# 每隔 5 分钟检查一次 huggingface-cli 下载进程，如果断了就自动重启

DATA_DIR="/home/gjw/Myprojects/lingbot-va/data/robotwin-clean-and-aug-lerobot"
LOG_FILE="/home/gjw/Myprojects/lingbot-va/download_10tasks.log"
WATCHDOG_LOG="/home/gjw/Myprojects/lingbot-va/watchdog_10tasks.log"
PID_FILE="/home/gjw/Myprojects/lingbot-va/download_10tasks.pid"
HF_CLI="/home/gjw/.conda/envs/lingbot/bin/huggingface-cli"

# 清除代理，设置镜像
unset HTTP_PROXY
unset HTTPS_PROXY
unset ALL_PROXY
export NO_PROXY="*"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=1
export PATH="/home/gjw/.conda/envs/lingbot/bin:$PATH"

# 10 个指定任务（当前已有的 9 个 + handover_mic）
TASKS=(
    "lerobot_robotwin_eef_aug_500/adjust_bottle-aloha-agilex_randomized_500-1000/**"
    "lerobot_robotwin_eef_aug_500/beat_block_hammer-aloha-agilex_randomized_500-1000/**"
    "lerobot_robotwin_eef_aug_500/blocks_ranking_rgb-aloha-agilex_randomized_500-1000/**"
    "lerobot_robotwin_eef_aug_500/blocks_ranking_size/**"
    "lerobot_robotwin_eef_aug_500/click_alarmclock-aloha-agilex_randomized_500-1000/**"
    "lerobot_robotwin_eef_aug_500/click_bell-aloha-agilex_randomized_500-1000/**"
    "lerobot_robotwin_eef_aug_500/dump_bin_bigbin-aloha-agilex_randomized_500-1000/**"
    "lerobot_robotwin_eef_aug_500/grab_roller-aloha-agilex_randomized_500-1000/**"
    "lerobot_robotwin_eef_aug_500/handover_block-aloha-agilex_randomized_500-1000/**"
    "lerobot_robotwin_eef_aug_500/handover_mic-aloha-agilex_randomized_500-1000/**"
)

# 记录启动时间
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Watchdog (10 tasks) started" >> "$WATCHDOG_LOG"

while true; do
    # 检查是否有活跃的 huggingface-cli 进程（排除自身和 grep）
    RUNNING_PID=$(pgrep -f "$HF_CLI download --repo-type dataset robbyant/robotwin-clean-and-aug-lerobot")
    
    if [ -z "$RUNNING_PID" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Download process not found. Restarting..." >> "$WATCHDOG_LOG"
        
        # 清理可能损坏的 .metadata 缓存（避免 huggingface_hub 的 bug）
        find "$DATA_DIR/.cache/huggingface/download" -name "*.metadata" -type f -delete 2>/dev/null
        
        # 构建 include 参数列表
        INCLUDE_ARGS=(
            "README.md"
            ".gitattributes"
            "empty_emb.pt"
            "${TASKS[@]}"
        )
        
        # 启动下载（追加日志）
        nohup "$HF_CLI" download \
            --repo-type dataset \
            robbyant/robotwin-clean-and-aug-lerobot \
            --local-dir "$DATA_DIR" \
            --include "${INCLUDE_ARGS[@]}" \
            >> "$LOG_FILE" 2>&1 &
        
        NEW_PID=$!
        echo "$NEW_PID" > "$PID_FILE"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Restarted with PID: $NEW_PID" >> "$WATCHDOG_LOG"
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Download is running (PID: $RUNNING_PID)" >> "$WATCHDOG_LOG"
    fi
    
    # 每 5 分钟检查一次
    sleep 300
done
