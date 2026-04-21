# LingBot-VA 项目速读

## 1. 这个项目在做什么

`LingBot-VA` 不是传统的“只输出动作”的 policy 项目，而是一个把视频世界模型和动作建模放到同一个 Transformer 里的后训练项目。

更直白一点：

- 它输入的是机器人任务片段对应的 `视频 latent + 动作序列 + 文本描述`
- 它训练的是一个联合世界模型
- 它输出的是一个新的 `transformer` 权重，用于后续推理、评测或继续训练

## 2. 你的训练输入是什么

当前 RoboTwin 训练配置默认读取：

- 预训练权重目录：`models/lingbot-va-posttrain-robotwin`
- 数据集目录：`data/robotwin-clean-and-aug-lerobot`
- 空文本 embedding：`data/robotwin-clean-and-aug-lerobot/empty_emb.pt`

训练样本不是原始图片，而是数据集类整理后的四个张量：

- `latents`
  含义：三路相机视频经过 Wan VAE 编码后的 latent
  典型形状：`[48, F, 24, 20]`
- `text_emb`
  含义：文本描述经过文本编码器后的 embedding
  典型形状：`[512, 4096]`
- `actions`
  含义：对齐到 latent 时间轴后的 30 维动作张量
  典型形状：`[30, F, 16, 1]`
- `actions_mask`
  含义：动作有效位掩码
  典型形状：`[30, F, 16, 1]`

当前 smoke 成功使用的单任务 repo 是：

- `click_bell-aloha-agilex_randomized_500-1000`

## 3. 模型的主要结构是什么

核心模型是 `wan_va/modules/model.py` 里的 `WanTransformer3DModel`。

可以把它理解成四段输入拼起来送入同一个 Transformer：

1. 带噪视频 latent
2. 条件视频 latent
3. 带噪动作 latent
4. 条件动作 latent

关键结构点：

- Transformer 层数：`30`
- 注意力头数：`24`
- 每头维度：`128`
- 视频 latent 通道：`48`
- 动作维度：`30`
- 文本维度：`4096`
- patch size：`(1, 2, 2)`

输入嵌入方式：

- 视频 latent：先做 patchify，再进 `patch_embedding_mlp`
- 动作：进 `action_embedder`
- 文本：进 `text_embedder`

训练时注意力模式使用 `flex`，推理时通常改用 `torch` 或 `flashattn`。
现在这条线已经在训练入口里强制覆盖训练态为 `flex`，不需要再手改官方权重目录。

## 4. 你在训练时实际在做什么

这不是强化学习，不是在和环境交互刷回报；它做的是一个后训练式的扩散/flow matching 学习过程。

训练流程可以理解为：

1. 读取一段任务片段的 latent、动作和文本 embedding
2. 分别给视频 latent 和动作张量加噪声
3. 把 noisy 输入、条件输入、文本条件一起送进 `WanTransformer3DModel`
4. 预测去噪目标
5. 分别计算：
   - `latent_loss`
   - `action_loss`
6. 反向传播并更新参数
7. 定期保存新的 `transformer` checkpoint

所以你训练得到的不是“最终视频”或“最终成功率”，而是一个新的模型权重，后续要靠评测脚本或 server/client 推理流程去验证。

## 5. 训练输出是什么

训练主产物在 `save_root/checkpoints/checkpoint_step_x/transformer/` 下，核心文件是：

- `config.json`
- `diffusion_pytorch_model.safetensors`

这些文件组成一个新的 `transformer` checkpoint。

如果开了 WandB，还会得到：

- 本地 run 目录：`wandb/run-*/`
- 远端 run 页面

## 6. 你最后可以得到什么

你最终能得到三类东西：

1. 新的后训练模型权重
   用于继续训练、离线分析，或接到评测/推理脚本
2. 训练过程指标
   例如 `latent_loss`、`action_loss`、`grad_norm`
3. 事实记录
   通过 `docs/fixes.md` 和 `docs/research_desk.md` 保留实验上下文

## 7. 当前这台机器的真实状态

- 你这台机器有一张 `RTX 5090 D v2 24GB`
- 它可以跑通单任务单卡 smoke
- 它目前不能直接做全参数单卡 RoboTwin post-train

已经验证到的事实：

- 全参数单卡训练在 `backward()` 和 `optimizer.step()` 阶段都会触发显存不足
- 因此新增了 `smoke_mode`

`smoke_mode` 的定义：

- 只训练输出头 `proj_out` 和 `action_proj_out`
- 目的是验证训练链路是否通
- 不是正式 full fine-tune 配置

## 8. 你现在最应该怎么用它

如果你想先理解项目：

1. 先看本文件
2. 再看 `docs/research_desk.md`
3. 最后按 `docs/fixes.md` 追溯具体 bug 和实验变化

如果你想先验证机器能跑：

1. 用单任务
2. 用 `SMOKE_MODE=1`
3. 先拿到 `1 step + 1 checkpoint`

如果你想继续做正式训练：

1. 优先考虑多卡
2. 或者考虑参数高效训练/更多冻结策略
3. 再评估是否需要优化器状态卸载、8-bit optimizer 等降显存手段
