# GROM

论文：*GROM: Gradient-Free Rapid One-Shot Machine Unlearning*（arXiv:2608.05783）。官方代码 pin：`Batorskq/GROM @6dd9591`。闭式编辑，**不是**同作者的 EvoMU。

Seed 0。不报标准 Forget Quality（KS p 值）。TOFU-10% 官方基座是 Llama-3.2-**1B**，与 BLADE 3B 矩阵不对齐。

## 协议（论文）

- TOFU Final = 均值(1−Rouge-L, 1−Prob, 1−Extr, MU)
- WMDP Final = ½((1−AccBio) + MMLU)
- MUSE Final = ½(KnowMem_Dr + ForgetAvg)，ForgetAvg = [(100−VerbMem_Df)+(100−KnowMem_Df)+(100−|PrivLeak|)] / 3
- Time：单卡 H100 上闭式编辑 wall-clock，不含加载与评测

## 原论文结果

### Table 2 · TOFU（节选）

TOFU-5%（LLaMA2-7B-Chat）：

| Method | 1-Rouge-L↑ | 1-Prob↑ | 1-Extr↑ | MU↑ | Final↑ | Time (m)↓ |
| --- | --- | --- | --- | --- | --- | --- |
| Retain | 0.61 | 0.85 | 0.93 | 0.62 | 0.71 | — |
| SimNPO | 0.74 | 0.97 | 0.92 | 0.58 | 0.73 | 2.6 |
| **GROM** | **0.95** | **1.00** | **0.97** | **0.62** | **0.79** | **1.8** |

TOFU-10%（LLaMA3.2-1B-Instruct）：

| Method | 1-Rouge-L↑ | 1-Prob↑ | 1-Extr↑ | MU↑ | Final↑ | Time (m)↓ |
| --- | --- | --- | --- | --- | --- | --- |
| Retain | 0.62 | 0.88 | 0.94 | 0.59 | 0.70 | — |
| SimNPO | 0.65 | 0.94 | 0.94 | 0.56 | 0.70 | 2.5 |
| **GROM** | **0.78** | **1.00** | **0.94** | **0.60** | **0.75** | **0.5** |

全文方法行见原文 Table 2。

### Table 4 · WMDP-Bio（Llama-3-8B-Instruct）

| Method | 1−AccBio↑ | MMLU↑ | Final↑ | Time (m)↓ |
| --- | --- | --- | --- | --- |
| Original | 0.27 | 0.65 | 0.46 | — |
| SimNPO | 0.75 | 0.44 | 0.60 | 21.6 |
| NPO | 0.73 | 0.51 | 0.62 | 23.9 |
| **GROM** | **0.71** | **0.55** | **0.63** | **0.2** |

### Table 5 · MUSE

News（LLaMA2-7B）：

| Method | VerbMem Df↓ | KnowMem Df↓ | PrivLeak→0 | KnowMem Dr↑ | Final↑ | Time (m)↓ |
| --- | --- | --- | --- | --- | --- | --- |
| Retain | 20.75 | 33.32 | 0.00 | 53.79 | 67.88 | — |
| SimNPO | 2.34 | 44.84 | 72.93 | 39.65 | 49.81 | 36.3 |
| **GROM** | **15.68** | **24.01** | **−3.80** | 26.37 | **55.93** | **0.3** |

Books（论文写 ICLM-7B；复现跟 yaml 用 `MUSE-Books_target`）：

| Method | VerbMem Df↓ | KnowMem Df↓ | PrivLeak→0 | KnowMem Dr↑ | Final↑ | Time (m)↓ |
| --- | --- | --- | --- | --- | --- | --- |
| Retain | 14.30 | 28.90 | 0.00 | 74.50 | 80.05 | — |
| SimNPO | 0.00 | 0.00 | −19.82 | 48.27 | 70.83 | 53.8 |
| **GROM** | **2.40** | **36.89** | **−0.22** | **65.56** | **76.20** | **0.3** |

4-bit 攻击见原文 Table 6（可选后续）。ZsRE 本仓库不做。

## 本仓库实测

<!-- MEASURED:START -->
| run | bench | note | 1-Rouge | 1-Prob | 1-Extr | MU | Final | Final_paper | Time_min | Time_paper | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 未复现 | TOFU-05-7B |  |  |  |  |  |  | 0.79 |  | 1.8 | 未复现 |
| 未复现 | TOFU-10-1B | 官方基座 1B，非 BLADE 3B |  |  |  |  |  | 0.75 |  | 0.5 | 未复现 |
| 未复现 | MUSE-News |  |  |  |  |  |  | 55.93 |  | 0.3 | 未复现 |
| 未复现 | MUSE-Books | 跟 yaml MUSE-Books_target |  |  |  |  |  | 76.20 |  | 0.3 | 未复现 |
| 未复现 | WMDP-Bio | Bio forget 门控 |  |  |  |  |  | 0.63 |  | 0.2 | 未复现 |
<!-- MEASURED:END -->
