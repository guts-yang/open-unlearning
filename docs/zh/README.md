# OpenUnlearning 中文介绍

本目录是仓库英文介绍与文档的中文对应版本，便于阅读项目定位、实验流程与复现准备。

- **命令、配置键、文件路径、代码块保持英文**，避免和 Hydra / 脚本不一致。
- 论文名、方法名、指标名一般保留英文，必要时加中文说明。
- 若中文与英文冲突，以根目录 `README.md` 与 `docs/` 下英文文档为准。

## 文档对照

| 中文 | 对应英文 |
|------|----------|
| [项目概述](./overview.md) | [`README.md`](../../README.md) |
| [实验配置与运行](./experiments.md) | [`docs/experiments.md`](../experiments.md) |
| [评测](./evaluation.md) | [`docs/evaluation.md`](../evaluation.md) |
| [组件扩展](./components.md) | [`docs/components.md`](../components.md) |
| [Hydra 用法](./hydra.md) | [`docs/hydra.md`](../hydra.md) |
| [复现结果](./repro.md) | [`docs/repro.md`](../repro.md) |
| [贡献指南](./contributing.md) | [`docs/contributing.md`](../contributing.md) |
| [文献与链接](./links.md) | [`docs/links.md`](../links.md) |

排行榜与社区方法说明仍为英文：[`community/leaderboard.md`](../../community/leaderboard.md)。

## 本仓库实际入口（与部分英文文档路径不同）

当前代码树里请使用下列路径；部分英文 README / `docs/experiments.md` 写成了 `src/train.py`、`setup_data.py`、`experiment=unlearn/tofu/default` 等，**在本克隆中并不存在**。

| 用途 | 本仓库实际路径 |
|------|----------------|
| 训练 / 遗忘 | `python src/train.py --config-name=unlearn.yaml` 或 `train.yaml` |
| 评测 | `python src/eval.py --config-name=eval.yaml` |
| 数据准备 | `python setup_data.py --eval_logs`（可选 `--idk`、`--wmdp`） |
| TOFU 遗忘实验配置 | `experiment=unlearn/tofu/default` → `configs/experiment/unlearn/tofu/default.yaml` |
| 基线脚本 | `bash scripts/tofu_unlearn.sh`、`bash scripts/muse_unlearn.sh` |
| 输出目录 | `saves/${mode}/${task_name}` |

中文文档中的命令已按上表写成**可在本仓库直接对照代码**的形式。
