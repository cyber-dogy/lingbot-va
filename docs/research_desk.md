# LingBot-VA Research Desk

## 当前阶段结论

### 2026-04-19 · 单任务单卡 smoke 已测通

- 成功任务：`click_bell-aloha-agilex_randomized_500-1000`
- 成功模式：`SMOKE_MODE=1`，只训练输出头
- 成功 step：`1`
- 成功 checkpoint：
  - `train_out/smoke/click_bell_step1_smoke/checkpoints/checkpoint_step_1/transformer`
- WandB：
  - project: `cyber-dogy/lingbot-va-smoke`
  - run id: `tzoicldy`
  - run name: `test_lln`
- 关键指标：
  - `latent_loss=0.0422`
  - `action_loss=0.7244`
  - `grad_norm=3.78`

## 当前真实能力边界

### 已确认可行

- 单任务数据读取
- 本地官方权重加载
- 单卡前向
- 单卡反向
- optimizer update
- checkpoint 保存
- WandB 记录

### 已确认不可直接做

- 在当前 `RTX 5090 D v2 24GB` 上直接做全参数单卡 RoboTwin post-train

已看到的失败模式：

- 默认数据集初始化曾被 `128` 个 Pool worker 拖慢
- `world_size=1` 仍走 FSDP 时，`backward()` 额外触发显存压力
- 去掉 FSDP 后，全参数 `AdamW` 和全参数 `SGD` 仍会在 `optimizer.step()` 阶段因显存不足失败

## 当前代码侧有效改动

- 单任务训练支持：
  - `--dataset-repo-name`
  - `script/run_va_posttrain_single_task.sh`
- 训练态 attention：
  - 训练入口显式使用 `attn_mode=flex`
- WandB：
  - 自动复用本地登录
- 单卡路径：
  - `world_size=1` 时不再初始化分布式/FSDP
- smoke：
  - 新增 `--smoke-mode`
  - 新增 `--optimizer-type`
  - smoke 下只训练 `proj_out` 和 `action_proj_out`

## 对项目理解的最短结论

- 这是一个视频 latent + 动作联合建模的后训练项目，不是 RL 交互训练。
- 训练得到的是新的 `transformer` 权重，而不是直接的 success rate。
- 真正的任务效果验证要靠后续的推理或评测脚本。

### 2026-04-19 · 离线 demo exporter 已打通本地单任务验证

- 已新增：
  - `script/export_robotwin_task_demos.py`
  - `script/run_export_robotwin_task_demos.sh`
- 导出内容：
  - `prediction.mp4`
  - `ground_truth.mp4`
  - `comparison.mp4`
  - `action_compare.png`
  - `metrics.json`
  - `summary.csv`
  - `summary_metrics.png`
- 当前本地单卡稳定配置：
  - `infer_attn_mode=flashattn`
  - `guidance_scale=1`
  - `action_guidance_scale=1`
  - `aux_vae_device=cpu`
  - `enable_text_encoder_offload=True`
- 已验证产物：
  - `eval_out/robotwin_task_demos/click_bell_short_demo`
- 已验证指标：
  - task: `click_bell-aloha-agilex_randomized_500-1000`
  - `num_chunks=2`
  - `num_inference_steps=2`
  - `action_num_inference_steps=4`
  - `video_mse=0.003538`
  - `action_mse=0.009784`
- 当前默认短视频长度：
  - 默认 `NUM_CHUNKS=6`
  - 在当前 `12 fps` 导出设置下，已实测 `pred_video_frames=45`
  - 实际时长约 `3.75` 秒
- 重要边界：
  - 当前这组参数是“本地快速可视化 smoke 导出”，不是官方高质量推理配置。
  - 如果切回更高 `guidance / diffusion steps`，单卡时延和显存压力都会显著上升。

## 下一步建议

### 如果目标是“继续熟悉项目”

1. 先重复一次 smoke，确认自己能独立复现
2. 再看 `wan_va/train.py`、`wan_va/dataset/lerobot_latent_dataset.py`、`wan_va/modules/model.py`
3. 选一个单任务继续观察 loss 变化和 checkpoint 结构

### 如果目标是“开始正式训练”

1. 优先准备多卡
2. 或者设计更明确的参数高效训练方案
3. 再决定是否做优化器状态卸载或更激进的显存压缩

### 如果目标是“开始评测”

1. 先确认目标 checkpoint 的 `attn_mode`
2. 再接 `evaluation/robotwin` 或推理 server/client 脚本
