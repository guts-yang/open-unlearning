# 评测产物约定

| 类型 | 路径 | 是否入库 |
|------|------|----------|
| ckpt、hydra 日志、`TOFU_EVAL.json` / `MUSE_SUMMARY.json` / `LMEval_SUMMARY.json` | `workspace/saves/` | 否 |
| 汇总 Markdown（先原论文表，再实测） | `workspace/results/` | 可 |
| 命令、seed、失败摘要 | `workspace/notes/` | 可 |

## 汇总命令

```bash
python workspace/scripts/aggregate_blade.py
python workspace/scripts/aggregate_grom.py
python workspace/scripts/aggregate_baldro.py
python workspace/scripts/aggregate_alter.py
```

各 `*_table.md` 先抄原文主表（PDF 在 `workspace/docs/papers/`），再留「本仓库实测」。`aggregate_*.py` 只替换 `<!-- MEASURED:START -->`…`END`，不得覆盖原论文节。无 JSON 时实测格为「未复现」，禁止把论文数字填进实测列。

BLADE 论文 HM 与框架 `src/evals/metrics/harmonic.py`（Mem/Priv/Utility）不是同一公式。GROM TOFU Final 也不是 FQ p 值。
