# TOFU · `Llama-3.2-1B-Instruct`

本机复现结果（动态表）。同一方法同一 forget split 会被覆盖更新。

更新时间：2026-09-04T22:23:18+08:00

## 复现 Leaderboard（每基座取 SOTA）

SOTA = 当前复现 runs 中 6 项「越高越好」指标（FQ / MU / 1−RL / TR / 1-Prob / 1-ES）的均值最高者。

| 基座名称 | SOTA 方法 | FQ↑ | MU↑ | 1−RL↑ | TR↑ | 1-Prob↑ | 1-ES↑ | 全称 | 设置一致性 |
|----------|-----------|-----|-----|-------|-----|---------|-------|------|------------|
| Llama-3.2-1B-Instruct | **RMU** | 4.24e-17 | 0.5853 | 0.9604 | 0.7718 | 0.9999 | 0.9675 | Representation Misdirection for Unlearning（Li et al., 2024） | 本机复现（forget10），硬件/torch 与官方 2×L40s 不同、只比量级；与官方未调参设置一致（FQ/MU/TR 量级吻合） |

FQ = forget_quality，MU = model_utility，TR = forget_truth_ratio，ES = extraction_strength，RL = forget ROUGE-L。1−RL / 1-Prob / 1-ES 已翻转为「越高越好」。privleak 等原始值见 `results/*.json`。

## 本机各方法明细（统一方向：越高越好）

| 基座名称 | 方法 | FQ↑ | MU↑ | 1−RL↑ | TR↑ | 1-Prob↑ | 1-ES↑ |
|----------|------|-----|-----|-------|-----|---------|-------|
| Llama-3.2-1B-Instruct | SpecDiff | 0.0935 | 0.3419 | 0.6994 | 0.6429 | 0.9532 | 0.9437 |
| Llama-3.2-1B-Instruct | SimNPO | 2.78e-16 | 0.5800 | 0.5211 | 0.5177 | 0.4145 | 0.8188 |
| Llama-3.2-1B-Instruct | **RMU** | 4.24e-17 | 0.5853 | 0.9604 | 0.7718 | 0.9999 | 0.9675 |
| Llama-3.2-1B-Instruct | UNDIAL | 6.16e-18 | 0.5655 | 0.6623 | 0.5617 | 0.7434 | 0.9549 |
| Llama-3.2-1B-Instruct | AltPO | 6.39e-06 | 0.5345 | 0.5929 | 0.6153 | 0.4922 | 0.8535 |

## 本机四维（OU Table 3 口径）

Mem=HM(1−ES,1−EM,1−ParaProb,1−TR)；Priv=HM(s_LOSS,s_ZLib,s_MinK,s_MinK++)；Utility=HM(MU,Fluency)；Agg=HM(Mem,Priv,Utility)；全部按 init-finetuned 归一化。来自 `scripts/ou_aggregate.py --json`（`--ou-summary`）。

| 基座名称 | 方法 | split | Mem | Priv | Utility | Agg | ckpt |
|----------|------|-------|-----|------|---------|-----|------|
| Llama-3.2-1B-Instruct | SpecDiff | forget10 | 0.4268 | 0.6812 | 0.7267 | 0.5784 | /root/autodl-tmp/saves/unlearn/tofu_1B_SpecDiff_forget10_seed2 |

## 官方未调参对照（仅 FQ / MU / TR）

来源：`docs/repro.md`，硬件与 torch 不同，只比量级。

| 方法 | split | 官方 FQ | 官方 MU | 官方 TR | 本机 FQ | 本机 MU | 本机 TR |
|------|-------|---------|---------|---------|---------|---------|---------|
| SpecDiff | forget10 | — | — | — | 0.0935 | 0.3419 | 0.6429 |
| SimNPO | forget10 | 2.47e-203 | 0.5400 | 1.07e-05 | 2.78e-16 | 0.5800 | 0.5177 |
| RMU | forget10 | 3.15e-15 | 0.5900 | 0.7600 | 4.24e-17 | 0.5853 | 0.7718 |
| UNDIAL | forget10 | — | — | — | 6.16e-18 | 0.5655 | 0.5617 |
| AltPO | forget10 | — | — | — | 6.39e-06 | 0.5345 | 0.6153 |

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
