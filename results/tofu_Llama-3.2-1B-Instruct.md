# TOFU · `Llama-3.2-1B-Instruct`

本机复现结果（动态表）。同一方法同一 forget split 会被覆盖更新。

更新时间：2026-09-03T09:27:09+08:00

## P0-3 本地基线（forget10 · 四维口径）

对照官方 `open-unlearning/eval` ±0.02 已 PASS。Fluency 分母来自本机 `full_local`。

| 行 | Mem | Priv | Utility | Agg |
|----|-----|------|---------|-----|
| Init finetuned（full_local） | 0 | 0.0999 | 1.0000 | 0 |
| Retain（retain90_local） | 0.3457 | 1.0000 | 1.0207 | 0.6157 |

Fluency 分母（full gibberish）见 `ou_aggregate.json`。denominators={'ES': 0.7055827512713931, 'EM': 0.9735549531877041, 'ParaProb': 0.10032089233398438, 'TR_ou': 0.6849163209604744, 'MU': 0.5987621967668202, 'Fluency': 0.8579648167081178}。

## 复现 Leaderboard（每基座取 SOTA）

SOTA = 当前复现 runs 中 6 项「越高越好」指标（FQ / MU / 1−RL / TR / 1-Prob / 1-ES）的均值最高者。

| 基座名称 | SOTA 方法 | FQ↑ | MU↑ | 1−RL↑ | TR↑ | 1-Prob↑ | 1-ES↑ | 全称 | 设置一致性 |
|----------|-----------|-----|-----|-------|-----|---------|-------|------|------------|
| Llama-3.2-1B-Instruct | **RMU** | 2.41e-29 | 0 | 1.0000 | 0.8357 | 1.0000 | 0.9675 | Representation Misdirection for Unlearning（Li et al., 2024） | 本机复现（forget10），硬件/torch 与官方 2×L40s 不同、只比量级；与官方未调参设置一致（FQ/MU/TR 量级吻合） |

FQ = forget_quality，MU = model_utility，TR = forget_truth_ratio，ES = extraction_strength，RL = forget ROUGE-L。1−RL / 1-Prob / 1-ES 已翻转为「越高越好」。privleak 等原始值见 `results/*.json`。

## 本机各方法明细（统一方向：越高越好）

| 基座名称 | 方法 | FQ↑ | MU↑ | 1−RL↑ | TR↑ | 1-Prob↑ | 1-ES↑ |
|----------|------|-----|-----|-------|-----|---------|-------|
| Llama-3.2-1B-Instruct | SimNPO | 0.0299 | 0.4562 | 0.6481 | 0.6817 | 0.8762 | 0.9423 |
| Llama-3.2-1B-Instruct | **RMU** | 2.41e-29 | 0 | 1.0000 | 0.8357 | 1.0000 | 0.9675 |
| Llama-3.2-1B-Instruct | UNDIAL | 1.79e-13 | 0.4417 | 0.6884 | 0.6118 | 0.8629 | 0.9613 |
| Llama-3.2-1B-Instruct | AltPO | 6.83e-09 | 0.5912 | 0.6656 | 0.5958 | 0.9716 | 0.9560 |
| Llama-3.2-1B-Instruct | NPO | 0.0126 | 0.5207 | 0.7521 | 0.6411 | 0.8771 | 0.9258 |
| Llama-3.2-1B-Instruct | GradDiff | 6.70e-137 | 0.5737 | 0.9541 | 0.0685 | 0.9817 | 0.9622 |
| Llama-3.2-1B-Instruct | TPO | 3.91e-08 | 6.67e-05 | 0.5920 | 0.4017 | 1.0000 | 0.9675 |

## 本机四维（OU Table 3 口径）

Mem=HM(1−ES,1−EM,1−ParaProb,1−TR)；Priv=HM(s_LOSS,s_ZLib,s_MinK,s_MinK++)；Utility=HM(MU,Fluency)；Agg=HM(Mem,Priv,Utility)；全部按 init-finetuned 归一化。来自 `scripts/ou_aggregate.py --json`（`--ou-summary`）。

| 基座名称 | 方法 | split | Mem | Priv | Utility | Agg | ckpt |
|----------|------|-------|-----|------|---------|-----|------|
| Llama-3.2-1B-Instruct | NPO | forget10 | 0.3120 | 0.6165 | 0.9700 | 0.5122 | /root/autodl-tmp/saves/unlearn/tofu_1B_NPO_forget10_N1 |
| Llama-3.2-1B-Instruct | SimNPO | forget10 | 0.3339 | 0.5654 | 0.8805 | 0.5085 | /root/autodl-tmp/saves/unlearn/tofu_1B_SimNPO_forget10_S_p1best |
| Llama-3.2-1B-Instruct | AltPO | forget10 | 0.5103 | 0.1914 | 0.9958 | 0.3664 | /root/autodl-tmp/saves/unlearn/tofu_1B_AltPO_forget10_A1 |
| Llama-3.2-1B-Instruct | GradDiff | forget10 | 0.9381 | 0.1623 | 0.7714 | 0.3519 | /root/autodl-tmp/saves/unlearn/tofu_1B_GradDiff_forget10_GD2 |
| Llama-3.2-1B-Instruct | UNDIAL | forget10 | 0.1219 | 0.2955 | 0.8451 | 0.2350 | /root/autodl-tmp/saves/unlearn/tofu_1B_UNDIAL_forget10_U1 |
| Llama-3.2-1B-Instruct | TPO | forget10 | 0.3994 | 0.3086 | 2.23e-04 | 6.68e-04 | /root/autodl-tmp/saves/unlearn/tofu_1B_TPO_forget10_TPO1 |
| Llama-3.2-1B-Instruct | RMU | forget10 | 0.6004 | 0.1521 | 0 | 0 | /root/autodl-tmp/saves/unlearn/tofu_1B_RMU_forget10_R1 |

## 官方未调参对照（仅 FQ / MU / TR）

来源：`docs/repro.md`，硬件与 torch 不同，只比量级。

| 方法 | split | 官方 FQ | 官方 MU | 官方 TR | 本机 FQ | 本机 MU | 本机 TR |
|------|-------|---------|---------|---------|---------|---------|---------|
| SimNPO | forget10 | 2.47e-203 | 0.5400 | 1.07e-05 | 0.0299 | 0.4562 | 0.6817 |
| RMU | forget10 | 3.15e-15 | 0.5900 | 0.7600 | 2.41e-29 | 0 | 0.8357 |
| UNDIAL | forget10 | — | — | — | 1.79e-13 | 0.4417 | 0.6118 |
| AltPO | forget10 | — | — | — | 6.83e-09 | 0.5912 | 0.5958 |
| NPO | forget10 | 0.0200 | 0.4600 | 0.7000 | 0.0126 | 0.5207 | 0.6411 |
| GradDiff | forget10 | 1.06e-239 | 0.4900 | 3.53e-27 | 6.70e-137 | 0.5737 | 0.0685 |
| TPO | forget10 | — | — | — | 3.91e-08 | 6.67e-05 | 0.4017 |

## 参照 / SOTA（TOFU 1B · forget10）

仓库 `community/leaderboard.md` 的 1B 表没有调参方法，**没有覆盖本表 6 项统一指标（FQ/MU/1−RL/TR/1-Prob/1-ES）的公开 SOTA**。下面分两档：

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

论文 Table 3 调参后综合分第一是 SimNPO（Agg 0.53），超参不同，对不上本表 6 项统一指标。未调参时 SimNPO 的 FQ 极差，**不是** FQ 的参照最优。
