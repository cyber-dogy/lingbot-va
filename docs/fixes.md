# Fixes Log

**每个 agent 在修改本项目代码前必须先读此文件。**

每次发现 bug、做出修改、观察到实验结果后，必须在本文件末尾追加一条记录。

格式固定为：

`### YYYY-MM-DD HH:MM:SS ±TZ · 简明标题`

并严格包含四个字段：

- `范围`：涉及的文件、模块或脚本
- `背景`：发生了什么，为什么值得记录
- `处理`：实际改了什么，或采取了什么应对动作
- `结果`：当前结论、关键指标、产物路径、后续状态

补充要求：

- 标题必须单看就能知道“这条记录在说什么”，不能只写文件名。
- 结果必须尽量写清 `run_name / checkpoint / success rate / 下一步状态`，避免只写 `none`、`待观察` 这类模糊词。
- 允许简略，但必须保证人和后续模型都能快速读懂上下文。
- `fixes.md` 是事实留痕源；如果某次改动已经形成跨线路的阶段结论，应同步提炼到 `docs/research_desk.md`，不要只停留在这里。

---

### 2026-04-19 10:12:34 +0800 · 单任务训练被 128 个数据集初始化进程拖慢

- `范围`：`wan_va/dataset/lerobot_latent_dataset.py`
- `背景`：第一次单任务 smoke 在 `Setting up datasets...` 后长期无进展。中断后发现 `MultiLatentLeRobotDataset` 默认起了 128 个 `ForkPoolWorker` 去构造子数据集，对 1 个 task 的 smoke 明显过量。
- `处理`：把数据集初始化 worker 改成按 `dataset_init_worker / repo 数 / CPU 数` 取最小值；单 repo 场景直接串行初始化，并保留必要中文注释。
- `结果`：单任务数据集初始化恢复到约 1-2 秒量级；后续 run 已能稳定进入 `Starting training for 1 steps...`。下一步转向单卡显存链路排查。

### 2026-04-19 10:15:53 +0800 · 单卡 world_size=1 仍走 FSDP 导致 backward 阶段显存浪费

- `范围`：`wan_va/distributed/util.py`、`wan_va/train.py`
- `背景`：第二次 smoke 在 `backward()` 报 `torch.OutOfMemoryError`，堆栈落在 FSDP reduce-scatter。根因是 `world_size=1` 时仍初始化分布式并包 FSDP，单卡没有收益，反而增加显存和同步开销。
- `处理`：改为仅在 `world_size > 1` 时初始化分布式/FSDP；单卡路径直接走普通模型；训练里的 gradient sync 和 checkpoint 保存补了非 FSDP 分支。
- `结果`：单卡训练已能越过 `backward()`，进一步暴露出真正的单卡瓶颈在 `optimizer.step()`。下一步排查优化器状态显存。

### 2026-04-19 10:17:18 +0800 · 全参数单卡 optimizer.step 仍超出 24GB 显存

- `范围`：`wan_va/train.py`、`script/run_va_posttrain.sh`
- `背景`：去掉 FSDP 后，`AdamW` 在首次状态初始化时仍 OOM；切到 `SGD` 后，全参数更新也在 multi-tensor step 阶段 OOM，说明当前 `RTX 5090 D v2 24GB` 不适合直接做全参数单卡 post-train。
- `处理`：新增 `--optimizer-type` 和 `--smoke-mode`；`smoke_mode` 下冻结 backbone，只训练 `proj_out`、`action_proj_out` 两个输出头；脚本新增环境变量透传。
- `结果`：全参数单卡训练仍不建议在当前机器上直接做；已提供可复现的 smoke 配置，下一步验证最小训练链路是否能完整走通。

### 2026-04-19 10:18:44 +0800 · click_bell 单任务 smoke 已完整跑通并保存 checkpoint

- `范围`：`script/run_va_posttrain.sh`、`script/run_va_posttrain_single_task.sh`、`wan_va/train.py`、`train_out/smoke/click_bell_step1_smoke`
- `背景`：需要一个真实成功的最小训练样例，帮助后续学习和排障有锚点。
- `处理`：执行 `TASK_NAME=click_bell-aloha-agilex_randomized_500-1000 NUM_STEPS=1 SAVE_INTERVAL=1 LOAD_WORKER=0 SMOKE_MODE=1 OPTIMIZER_TYPE=sgd WANDB_PROJECT=lingbot-va-smoke bash script/run_va_posttrain_single_task.sh`，并保持 WandB 联通。
- `结果`：`run_name=test_lln`，`wandb_run=tzoicldy`，`latent_loss=0.0422`，`action_loss=0.7244`，`grad_norm=3.78`；checkpoint 已保存到 `train_out/smoke/click_bell_step1_smoke/checkpoints/checkpoint_step_1/transformer`；下一步如果要做正式训练，需要多卡或更明确的参数高效训练方案。

### 2026-04-19 11:22:11 +0800 · 离线 demo exporter 增加本地单卡可跑的导出链路

- `范围`：`script/export_robotwin_task_demos.py`、`script/run_export_robotwin_task_demos.sh`、`wan_va/configs/shared_config.py`、`wan_va/wan_va_server.py`
- `背景`：需要在 SSH 环境下把本地 10 个 RoboTwin task 导出成可拷走的 demo 目录，但原始推理链路默认更偏 server 场景，本地 24GB 单卡既缺批量导出入口，也容易在 cache / VAE 常驻上踩显存边界。
- `处理`：新增批量导出脚本和 shell 包装；推理配置支持 `infer_attn_mode`、`guidance_scale`、`action_guidance_scale`、推理步数覆盖；server 侧补了 `aux_vae_device`、`enable_text_encoder_offload`、prompt 输出目录清理、`decode_one_video()` 的 `detach()` 修复，并把 tokenizer 并行警告在导出入口压掉。
- `结果`：离线导出目录结构已固定为 `prediction.mp4 / ground_truth.mp4 / comparison.mp4 / action_compare.png / metrics.json / summary.csv / summary_metrics.png`；本地单卡推荐起点为 `flashattn + guidance=1 + action_guidance=1 + aux_vae_device=cpu + text_encoder_offload=True`；下一步可以直接批量跑 10 个 task 产出短 demo。

### 2026-04-19 11:22:12 +0800 · click_bell 短轨迹离线导出已成功落盘

- `范围`：`script/export_robotwin_task_demos.py`、`eval_out/robotwin_task_demos/click_bell_smoke_export`、`eval_out/robotwin_task_demos/click_bell_short_demo`
- `背景`：需要一个真实成功的推理/可视化样例，确认导出脚本不仅能起，而且真的会生成视频、图表和 summary。
- `处理`：先用 `num_chunks=1`、`num_inference_steps=2`、`action_num_inference_steps=4` 跑通整条导出链路，再用相同低步数配置补跑 `num_chunks=2` 的短轨迹 demo。
- `结果`：`click_bell-aloha-agilex_randomized_500-1000` 已成功导出两组结果；`click_bell_smoke_export` 指标为 `video_mse=0.001202 / action_mse=0.019562`；`click_bell_short_demo` 指标为 `video_mse=0.003538 / action_mse=0.009784`；关键产物位于 `eval_out/robotwin_task_demos/click_bell_short_demo/click_bell-aloha-agilex_randomized_500-1000`；下一步可直接把同一命令扩展到全部 10 个本地 task。

### 2026-04-19 12:00:00 +0800 · 用户请求的 click_bell 单任务推理已再次成功导出

- `范围`：`script/run_export_robotwin_task_demos.sh`、`script/export_robotwin_task_demos.py`、`eval_out/robotwin_task_demos/click_bell_user_run_20260419_1`
- `背景`：需要在当前机器上直接执行一次真实推理，确认当前代码和环境无需额外手工干预就能产出可查看的 demo 文件。
- `处理`：执行 `OUTPUT_ROOT=/home/gjw/MyProjects/lingbot-va/eval_out/robotwin_task_demos/click_bell_user_run_20260419_1 TASK_NAMES=click_bell-aloha-agilex_randomized_500-1000 NUM_CHUNKS=2 NUM_INFERENCE_STEPS=2 ACTION_NUM_INFERENCE_STEPS=4 bash script/run_export_robotwin_task_demos.sh`，沿用 `flashattn + guidance=1 + action_guidance=1` 的本地单卡配置。
- `结果`：任务 `click_bell-aloha-agilex_randomized_500-1000` 推理成功，`status=success`；指标为 `video_mse=0.003545 / video_mae=0.025363 / action_mse=0.009784 / action_mae=0.028475`；产物已落到 `eval_out/robotwin_task_demos/click_bell_user_run_20260419_1/click_bell-aloha-agilex_randomized_500-1000`，包含 `prediction.mp4 / ground_truth.mp4 / comparison.mp4 / action_compare.png / metrics.json / summary.csv`；下一步可直接扩展到其余 9 个本地 task 批量导出。

### 2026-04-19 12:07:52 +0800 · 默认 demo 导出长度已改成 3-5 秒并完成实测

- `范围`：`script/export_robotwin_task_demos.py`、`script/run_export_robotwin_task_demos.sh`、`docs/research_desk.md`、`eval_out/robotwin_task_demos/click_bell_3to5s_default`
- `背景`：原默认值 `NUM_CHUNKS=2` 只会导出约 1.08 秒视频，不符合“每个任务输出 3-5 秒短视频”的目标。
- `处理`：把 exporter 和 shell 包装的默认 `num_chunks / NUM_CHUNKS` 都改为 `6`，并补充中文注释说明当前 12 fps 下对应时长；随后用 `click_bell-aloha-agilex_randomized_500-1000` 重新执行默认配置验证。
- `结果`：新默认已实测成功，`num_chunks=6` 时得到 `pred_video_frames=45`，按 `12 fps` 计算约 `3.75` 秒；验证产物位于 `eval_out/robotwin_task_demos/click_bell_3to5s_default/click_bell-aloha-agilex_randomized_500-1000`，指标为 `video_mse=0.003563 / action_mse=0.006737`；后续直接不传 `NUM_CHUNKS` 也会默认导出 3-5 秒短视频。
