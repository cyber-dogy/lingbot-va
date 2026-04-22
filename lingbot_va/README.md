# LingBot-VA Line

这条线按 `pdit` 的包内分层方式整理成独立目录，目标是后续可以整包迁走。

当前目录结构：

- `cli/`：命令行入口
- `config/`：配置 schema 与 JSON 读取
- `data/`：数据集与数据注册表
- `model/`：encoder / backbone / head
- `policy/`：策略封装与 loss
- `train/`：builder / runner / checkpoint / eval
- `configs/`：默认配置样例

当前先提供了一套最小可工作的合成数据训练闭环，后续可以在不依赖 `wan_va`、`pdit` 的前提下，逐步替换成真实的 LingBot-VA 数据与模型实现。
