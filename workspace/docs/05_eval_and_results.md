# 评测产物约定

| 类型 | 路径 | 是否入库 |
|------|------|----------|
| ckpt、hydra 日志、`TOFU_EVAL.json` / `MUSE_SUMMARY.json` / `LMEval_SUMMARY.json` | `workspace/saves/` | 否 |
| 汇总 CSV | `workspace/results/` | 可 |
| 命令、seed、失败摘要 | `workspace/notes/` | 可 |

## 汇总命令

```bash
python workspace/scripts/aggregate_blade.py
python workspace/scripts/aggregate_baldro.py
python workspace/scripts/aggregate_alter.py
```

无 JSON 时 CSV 只有表头 +「未复现」行。有 JSON 才填写实测值。

BLADE 论文 HM 与框架 `src/evals/metrics/harmonic.py`（Mem/Priv/Utility）不是同一公式。
