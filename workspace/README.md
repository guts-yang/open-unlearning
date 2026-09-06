# workspace

本仓库的**本地实验区**。官方 `src/`、`configs/`、`scripts/`、`community/` 在未明确「合入主线」前不要混入实验文件。约束见 `.cursor/rules/open-unlearning-workspace.mdc`。

| 目录 | 用途 |
|------|------|
| `scripts/` | 本地跑数、sweep、汇总 |
| `src/` | 未合入的 trainer / metric |
| `configs/` | 实验用 Hydra overlay |
| `notes/` | 实验记录 |
| `saves/` | checkpoint、评测 JSON、hydra 日志（不入库） |

训练/评测请覆盖输出路径，例如 `paths.output_dir=workspace/saves/unlearn/<task_name>`，不要写到仓库根 `saves/`。
