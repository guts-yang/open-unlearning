# 评测产物约定

| 类型 | 路径 | 是否入库 |
|------|------|----------|
| ckpt、hydra 日志、`TOFU_EVAL.json` / `MUSE_SUMMARY.json` / `LMEval_SUMMARY.json` / `LeakRU_EVAL.json` / `LeakRU_SUMMARY.json` | `workspace/saves/` | 否 |
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

BLADE 论文 HM 与框架 `src/evals/metrics/harmonic.py`（Mem/Priv/Utility）不是同一公式。GROM TOFU Final 也不是 FQ p 值。LeakRU 的 `leakru_fq`（相对 Acc）也不是 TOFU 的 KS `forget_quality` p 值，禁止填进同一格。

## LeakRU

论文：[`workspace/docs/papers/【LeakRU】Leak-Resistant Unlearning A New Benchmark for Evaluating Multi-Hop Reasoning Consistency and Recovery Robustness.pdf`](papers/【LeakRU】Leak-Resistant Unlearning%20A%20New%20Benchmark%20for%20Evaluating%20Multi-Hop%20Reasoning%20Consistency%20and%20Recovery%20Robustness.pdf)（arXiv:2608.04519）。插件在 `workspace/src/open_unlearning_plugins/leakru/`，不改官方 `src/`。

**挂到 OU 评测链（opt-in）**：不要用 `eval=leakru` 顶掉 TOFU/MUSE。把实验里的 `override /eval: tofu` 改成 `tofu_and_leakru`（MUSE 用 `muse_and_leakru`）。入口仍是 `workspace/scripts/eval.py`，或遗忘训练末尾 `do_eval`。同一 `paths.output_dir` 下会同时有 `TOFU_EVAL.json`（或 MUSE）和 `LeakRU_EVAL.json`。

指标：

| key | 含义 |
|-----|------|
| `leakru_acc` | 按 `logic_type`（HS/MT/DS/LL/MP/CC）分组的生成准确率；`V(·)` 为 gold ∪ aliases 的规范化子串匹配 |
| `leakru_fq` | `1 - Acc(θ_u)/Acc(θ_p)`；θ_p 来自 `eval.leakru.pretrained_logs_path` 指向的 `LeakRU_EVAL.json`（与 TOFU `retain_logs_path` 同一机制） |
| `leakru_rr` | 对 θ_p 答对的题 `S_suc` 做 Probab / FocusOnKey / Quantization 后再算正确率 |

Utility 继续用已有的 `model_utility` / MUSE ROUGE / `lm_eval`，LeakRU 不评 BBH。

**本机（无权重）**：先测代码，不下载数据。

```bash
python -m pytest workspace/tests/test_leakru.py workspace/tests/test_register.py workspace/tests/test_configs.py -q
```

**远程（数据与权重就绪后，当前仓库未执行）**：

1. 把官方 LeakRU json/jsonl 放到 `/root/autodl-tmp/data/leakru/`（或设 `LEAKRU_DATA_DIR` / `eval.leakru.data_path`）。文件不存在会报错，不要用假题填充。
2. `source /root/autodl-tmp/env_hf.sh` 并激活 venv。
3. 对原始模型 θ_p 跑一遍，得到 `LeakRU_EVAL.json`。
4. 对遗忘模型 θ_u 再跑，设置 `eval.leakru.pretrained_logs_path` 为上一步的 `LeakRU_EVAL.json`。
5. 读同目录 `LeakRU_SUMMARY.json`。完整 hydra 命令见 [`06_leakru_run.md`](06_leakru_run.md)。
