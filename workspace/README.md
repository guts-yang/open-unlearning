# workspace

本仓库的**本地实验区**。官方 `src/`、`configs/`、`scripts/`、`community/`、`docs/` 在未明确「合入主线」前不要混入实验文件。约束见 `.cursor/rules/open-unlearning-workspace.mdc`。

| 目录 | 用途 |
|------|------|
| `scripts/` | 训练/评测入口、sweep、汇总 |
| `src/` | 未合入的 trainer / metric 插件 |
| `configs/` | 实验用 Hydra overlay |
| `docs/` | 远程权重下载与训练手册 |
| `results/` | 原论文主表 + 本仓库实测（无实测则「未复现」） |
| `notes/` | 实验记录 |
| `saves/` | checkpoint、评测 JSON、hydra 日志（不入库） |

```bash
# 单测在 workspace/tests/（本地跑，不入库）
PYTHONPATH=src:workspace/src python -m pytest workspace/tests -q

python workspace/scripts/train.py experiment=unlearn/blade_tofu01_smoke \
  paths.output_dir=workspace/saves/unlearn/blade_tofu01_smoke

# 本机跑数脚本在 workspace/scripts/local/（AutoDL 路径，不入库）
bash workspace/scripts/local/run_simnpo_baldro.sh
```

训练必须覆盖 `paths.output_dir` 到 `workspace/saves/`。手册从 `workspace/docs/00_overview.md` 读起（BLADE → GROM → BalDRO → ALTER）。
