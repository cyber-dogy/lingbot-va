# LingBot-VA 文档目录

## 推荐阅读顺序

1. [project_guide.md](project_guide.md)
2. [research_desk.md](research_desk.md)
3. [fixes.md](fixes.md)

## 文件说明

- `project_guide.md`
  作用：给第一次接触 `lingbot-va` 的人快速建立输入、输出、模型结构、训练目标和产物认知。
- `research_desk.md`
  作用：沉淀当前阶段结论、关键风险、可复现实验入口和下一步建议。
- `fixes.md`
  作用：事实留痕源。所有 bug、修复、实验观察都必须追加到这里。

## 当前最新结论

- 2026-04-19 已经打通单任务单卡 smoke 路径。
- 成功 smoke 产物：
  - checkpoint: `train_out/smoke/click_bell_step1_smoke/checkpoints/checkpoint_step_1/transformer`
  - wandb run: `tzoicldy`
- 当前机器的 `RTX 5090 D v2 24GB` 不能直接承载全参数单卡 post-train；已提供 `smoke_mode` 用于先验证训练链路。
