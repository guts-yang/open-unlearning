# TOFU · `Llama-3.2-1B-Instruct`

本机复现结果。更新时间：2026-09-02T10:42:08+08:00

硬件：2× Tesla V100-PCIE-32GB；注意力 **sdpa**（V100 无 FA2）；torch 2.8.0+cu128。  
评测脚本：`bash scripts/ou_eval_baselines.sh`（产物写 `*_local`，不覆盖官方目录）。  
官方对照：`scripts/ou_compare_official.py` 容差 ±0.02 → **PASS**。

---

## P0-3 本地基线（forget10 · 四维口径）

四维按 OU 论文 §F.1：`ou_aggregate.py`，归一化分母 = 本机 `full_local`，sMIA 参考 = 本机 `retain90_local`。

| 行 | Mem | Priv | Utility | Agg | 论文 Table 3 | 说明 |
|----|-----|------|---------|-----|--------------|------|
| Init finetuned（`full_local`） | 0.0000 | 0.0999 | 1.0000 | 0.0000 | 0.00 / 0.10 / 1.00 / 0.00 | 锚点命中 |
| Retain（`retain90_local`） | 0.3457 | 1.0000 | 1.0207 | 0.6157 | 0.31 / 1.00 / 0.99 / 0.58 | Mem 与论文差 +0.036（已知，见 `docs/zh/ou-table3-p0.md`）；Utility>1 因本机 Fluency(retain) > Fluency(full) |

产物：

- `$SAVES/eval/tofu_Llama-3.2-1B-Instruct_retain90_local/{TOFU_SUMMARY,TOFU_EVAL,ou_aggregate}.json`
- `$SAVES/eval/tofu_Llama-3.2-1B-Instruct_full_local/evals_forget10/` 同上

### 归一化分母（init-finetuned = full_local）

| 指标 | 本机 full_local | 官方 eval log |
|------|-----------------|---------------|
| ES | 0.70558 | 0.70627 |
| EM | 0.97355 | 0.97394 |
| ParaProb | 0.10032 | 0.10042 |
| TR_OU（prob_mean） | 0.68492 | 0.68498 |
| MU | 0.59876 | 0.59915 |
| Fluency（`forget_Q_A_gibberish`） | **0.85796** | 官方从未评测 |

### sMIA 参考（retain90_local AUC）

mia_loss 0.38726 / mia_zlib 0.30891 / mia_min_k 0.38204 / mia_min_k++ 0.47646

### 原始指标（本机 vs 官方，对照已过）

| 指标 | retain90 官方 | retain90 本地 | Δ | full 官方 | full 本地 | Δ |
|------|---------------|---------------|---|-----------|-----------|---|
| extraction_strength | 0.05896 | 0.05947 | +0.00051 | 0.70627 | 0.70558 | −0.00069 |
| exact_memorization | 0.58489 | 0.58557 | +0.00068 | 0.97394 | 0.97355 | −0.00038 |
| model_utility | 0.59112 | 0.59089 | −0.00024 | 0.59915 | 0.59876 | −0.00039 |
| mia_loss | 0.38736 | 0.38726 | −0.00010 | 0.99649 | 0.99648 | −0.00001 |
| mia_zlib | 0.30889 | 0.30891 | +0.00002 | 0.99794 | 0.99793 | −0.00001 |
| mia_min_k | 0.38261 | 0.38204 | −0.00057 | 0.99665 | 0.99664 | −0.00001 |
| mia_min_k_plus_plus | 0.47763 | 0.47646 | −0.00116 | 0.99803 | 0.99806 | +0.00003 |
| forget_Q_A_PARA_Prob | 0.05341 | 0.05341 | +0.00000 | 0.10042 | 0.10032 | −0.00010 |
| forget_truth_ratio（closer） | 0.62746 | 0.62751 | +0.00005 | 0.47556 | 0.47517 | −0.00039 |
| forget_Q_A_Prob | 0.11613 | 0.11620 | — | — | 0.88057 | — |
| forget_Q_A_ROUGE | 0.37907 | 0.37882 | — | — | 0.81975 | — |
| forget_quality | — | 1.0000 | — | 3.91e-22 | 3.91e-22 | — |
| privleak | 23.48（官方） | 0.0926 | 口径：本地自评参考自己 | — | −99.46 | — |
| forget_Q_A_gibberish | — | **0.90674** | 官方无 | — | **0.85796** | 官方无 |

说明：官方 retain90 的 `privleak=23.48` 与本地 `0.09` 不可直接比——本地首次评测用官方 retain 日志作参考，重跑时用本地自己作参考，接近 0 是预期。对照脚本不比 privleak。

七方法自训（S1/R1/G1/N1/U1/GA1/TPO1）尚未产出；完成后由 `ou_table.py` 写 `results/ou_table3.md`。

---

## 归档：2026-08-28 默认超参 + 旧六指标

下列 4 个 ckpt **不是** Table 3 调参自训，不能写进论文四维表。保留供同口径内比较。

SOTA = 6 项「越高越好」均值最高者（FQ / MU / 1−RL / TR / 1-Prob / 1-ES）。

| 基座名称 | SOTA 方法 | FQ↑ | MU↑ | 1−RL↑ | TR↑ | 1-Prob↑ | 1-ES↑ | 全称 | 设置一致性 |
|----------|-----------|-----|-----|-------|-----|---------|-------|------|------------|
| Llama-3.2-1B-Instruct | **RMU** | 4.24e-17 | 0.5853 | 0.9604 | 0.7718 | 0.9999 | 0.9675 | Representation Misdirection for Unlearning（Li et al., 2024） | 8/28 未调参默认超参 |

| 基座名称 | 方法 | FQ↑ | MU↑ | 1−RL↑ | TR↑ | 1-Prob↑ | 1-ES↑ |
|----------|------|-----|-----|-------|-----|---------|-------|
| Llama-3.2-1B-Instruct | SimNPO | 2.78e-16 | 0.5800 | 0.5211 | 0.5177 | 0.4145 | 0.8188 |
| Llama-3.2-1B-Instruct | **RMU** | 4.24e-17 | 0.5853 | 0.9604 | 0.7718 | 0.9999 | 0.9675 |
| Llama-3.2-1B-Instruct | UNDIAL | 6.16e-18 | 0.5655 | 0.6623 | 0.5617 | 0.7434 | 0.9549 |
| Llama-3.2-1B-Instruct | AltPO | 6.39e-06 | 0.5345 | 0.5929 | 0.6153 | 0.4922 | 0.8535 |
