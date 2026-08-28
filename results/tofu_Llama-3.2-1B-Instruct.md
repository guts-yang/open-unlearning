# TOFU · `Llama-3.2-1B-Instruct`

本机复现结果（动态表）。同一方法同一 forget split 会被覆盖更新。

更新时间：2026-08-28T10:19:03+08:00

## 本机指标

| 方法 | split | FQ | MU | TR | forget Prob | forget ROUGE | privleak | ES |
|------|-------|----|----|----|-------------|--------------|----------|----|
| SimNPO | forget10 | 2.78e-16 | 0.5800 | 0.5177 | 0.5855 | 0.4789 | -97.7850 | 0.1812 |
| RMU | forget10 | 4.24e-17 | 0.5853 | 0.7718 | 9.02e-05 | 0.0396 | 57.8010 | 0.0325 |
| UNDIAL | forget10 | 6.16e-18 | 0.5655 | 0.5617 | 0.2566 | 0.3377 | -93.7317 | 0.0451 |

FQ = forget_quality，MU = model_utility，TR = forget_truth_ratio，ES = extraction_strength。

## 官方未调参对照（仅 FQ / MU / TR）

来源：`docs/repro.md`，硬件与 torch 不同，只比量级。

| 方法 | split | 官方 FQ | 官方 MU | 官方 TR | 本机 FQ | 本机 MU | 本机 TR |
|------|-------|---------|---------|---------|---------|---------|---------|
| SimNPO | forget10 | 2.47e-203 | 0.5400 | 1.07e-05 | 2.78e-16 | 0.5800 | 0.5177 |
| RMU | forget10 | 3.15e-15 | 0.5900 | 0.7600 | 4.24e-17 | 0.5853 | 0.7718 |
| UNDIAL | forget10 | — | — | — | 6.16e-18 | 0.5655 | 0.5617 |

## 参照 / SOTA（TOFU 1B · forget10）

仓库 `community/leaderboard.md` 的 1B 表没有调参方法，**没有覆盖全部 8 列的公开 SOTA**。下面分两档：

### 上界与未遗忘下界

| 参照 | FQ | MU | TR | forget Prob | forget ROUGE | privleak | ES | 说明 |
|------|----|----|----|-------------|--------------|----------|----|------|
| Retain（上界） | 1.0000 | 0.5900 | 0.6300 | 0.1161 | 0.3791 | 23.4775 | 0.0590 | 没学过 forget；FQ/MU/TR 来自 `docs/repro.md`，其余来自官方 retain90 `TOFU_SUMMARY.json` |
| Finetuned（下界） | 1.66e-21 | 0.6000 | 0.4800 | — | — | — | — | 遗忘前的 full 模型，`docs/repro.md` |

方向：FQ / MU 越高越好；TR 越接近 1 越好；forget Prob / ROUGE / ES 越低通常越「忘得干净」（但效用崩了也会变低）；privleak 越接近 0 越好。

### 官方未调参方法里最好（不是 Retain）

来源：`docs/repro.md` Llama-3.2-1B-Instruct forget10。Prob / ROUGE / privleak / ES 无官方方法对照。

| 指标 | 最好方法 | 数值 |
|------|----------|------|
| FQ | NPO | 0.0200 |
| MU | RMU | 0.5900 |
| TR | RMU | 0.7600 |

论文 Table 3 调参后综合分第一是 SimNPO（Agg 0.53），超参不同，对不上本表 8 列。未调参时 SimNPO 的 FQ 极差，**不是** FQ 的参照最优。
