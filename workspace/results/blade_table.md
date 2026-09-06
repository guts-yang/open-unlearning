# BLADE

论文：*BLADE: Bilevel Low-rank Augmented-Lagrangian Erasure for LLM Unlearning*（arXiv:2608.22557）。原文：[workspace/docs/papers/【172】BLADE- Bilevel Low-rank Augmented-Lagrangian Erasure for LLM Unlearning.pdf](../docs/papers/【172】BLADE- Bilevel Low-rank Augmented-Lagrangian Erasure for LLM Unlearning.pdf)。官方代码 pin：`tzpranto/blade @84f7504`。

## 协议（论文）

- 框架：OpenUnlearning；5 个 seed 报均值±标准差（正文未逐条列出 seed 编号；本仓库训练用 42 / 123 / 456 / 789 / 1024）。
- TOFU：Llama-3.2-1B/3B-Instruct，forget 1% / 5% / 10%。**HM = hmean(MU, 1−Prob, 1−RG)**。不是框架 Forget Quality（KS p 值）。
- MUSE：Llama-2-7B，Books / News。**HM = hmean(1−fk, 1−vm, rk)**。
- KnowUnDo：Llama-2-7B-chat + 域 LoRA；**HM = hmean(1−fgt_R, ret_R, MMLU)**。本 fork 无 evaluator，实测延后。
- 算力：论文为单卡 H100；本仓库目标 2×A800。

## 原论文结果

### Table 1 · TOFU Llama-3.2-1B-Instruct（5 seeds）

Forget 1%：

| Method | MU↑ | Prob↓ | RG↓ | HM↑ |
| --- | --- | --- | --- | --- |
| Gold (retrain) | 0.597 | 0.166 | 0.414 | 0.655 |
| GradAscent | 0.596 | 0.477 | 0.456 | 0.553 |
| GradDiff | 0.581 | 0.421 | 0.464 | 0.564 |
| NPO | 0.596 | 0.474 | 0.438 | 0.560 |
| SimNPO | 0.594 | 0.858 | 0.734 | 0.240 |
| RMU | 0.557 | 0.414 | 0.415 | 0.576 |
| BLURNPO | 0.598 | 0.676 | 0.588 | 0.417 |
| PDU | 0.602 | 0.186 | 0.313 | 0.690 |
| **BLADE** | **0.599** | **0.002** | **0.038** | **0.808** |

Forget 5%：

| Method | MU↑ | Prob↓ | RG↓ | HM↑ |
| --- | --- | --- | --- | --- |
| Gold (retrain) | 0.599 | 0.127 | 0.383 | 0.676 |
| GradAscent | 0.008 | 0.003 | 0.112 | 0.022 |
| GradDiff | 0.453 | 0.073 | 0.375 | 0.614 |
| NPO | 0.454 | 0.245 | 0.308 | 0.603 |
| SimNPO | 0.596 | 0.848 | 0.741 | 0.248 |
| RMU | 0.546 | 0.369 | 0.423 | 0.583 |
| BLURNPO | 0.518 | 0.475 | 0.407 | 0.541 |
| PDU | 0.588 | 0.073 | 0.214 | 0.740 |
| **BLADE** | **0.597** | **0.015** | **0.054** | **0.800** |

Forget 10%：

| Method | MU↑ | Prob↓ | RG↓ | HM↑ |
| --- | --- | --- | --- | --- |
| Gold (retrain) | 0.591 | 0.116 | 0.379 | 0.677 |
| GradAscent | 0.000 | 0.000 | 0.001 | 0.000 |
| GradDiff | 0.435 | 0.047 | 0.339 | 0.617 |
| NPO | 0.393 | 0.213 | 0.210 | 0.590 |
| SimNPO | 0.597 | 0.842 | 0.734 | 0.255 |
| RMU | 0.572 | 0.106 | 0.322 | 0.691 |
| BLURNPO | 0.089 | 0.087 | 0.253 | 0.154 |
| PDU | 0.592 | 0.004 | 0.065 | 0.797 |
| **BLADE** | **0.593** | **0.009** | **0.039** | **0.803** |

标准差见原文 Table 1（BLADE 各列约 ±0.00–0.01）。LLM judge HM_J：BLADE 0.929 / 0.919 / 0.919（1B 三 split）。

### Table 11 · TOFU Llama-3.2-3B（Appendix D，5 seeds）

| Split | Method | MU↑ | Prob↓ | RG↓ | HM↑ |
| --- | --- | --- | --- | --- | --- |
| fgt01 | Gold | 0.663 | 0.179 | 0.409 | 0.656 |
| fgt01 | PDU | 0.692 | 0.168 | 0.283 | 0.742 |
| fgt01 | **BLADE** | **0.654** | **0.000** | **0.015** | **0.846** |
| fgt05 | Gold | 0.659 | 0.130 | 0.388 | 0.698 |
| fgt05 | PDU | 0.687 | 0.009 | 0.085 | 0.843 |
| fgt05 | **BLADE** | **0.655** | **0.007** | **0.028** | **0.842** |
| fgt10 | Gold | 0.661 | 0.115 | 0.382 | 0.698 |
| fgt10 | PDU | 0.680 | 0.000 | 0.029 | 0.857 |
| fgt10 | **BLADE** | **0.647** | **0.005** | **0.038** | **0.836** |

全文方法行见原文 Table 11。论文注：PDU 在 3B forget05/10 的 HM 略高于 BLADE；BLADE 在 judge 上仍领先。

### Table 2 · MUSE Llama-2-7B（5 seeds）

| Method | Books fk↓ | vm↓ | rk↑ | HM↑ | News fk↓ | vm↓ | rk↑ | HM↑ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Gold (retrain) | 0.303 | 0.145 | 0.687 | 0.739 | 0.324 | 0.204 | 0.552 | 0.660 |
| GradAscent | 0.000 | 0.000 | 0.000 | 0.000 | 0.001 | 0.020 | 0.003 | 0.009 |
| GradDiff | 0.000 | 0.000 | 0.004 | 0.011 | 0.329 | 0.043 | 0.269 | 0.479 |
| NPO | 0.303 | 0.342 | 0.574 | 0.638 | 0.517 | 0.274 | 0.435 | 0.522 |
| SimNPO | 0.238 | 0.002 | 0.600 | 0.753 | 0.628 | 0.542 | 0.513 | 0.440 |
| RMU | 0.210 | 0.113 | 0.598 | 0.738 | 0.495 | 0.267 | 0.434 | 0.531 |
| BLURNPO | 0.175 | 0.000 | 0.547 | 0.742 | 0.302 | 0.146 | 0.274 | 0.478 |
| PDU | 0.139 | 0.129 | 0.372 | 0.600 | 0.525 | 0.092 | 0.508 | 0.577 |
| **BLADE** | **0.089** | **0.000** | **0.638** | **0.818** | **0.545** | **0.211** | **0.490** | **0.544** |

News 上 PDU HM 0.577 高于 BLADE 0.544。

### Table 13 · KnowUnDo（5-fold）

| Domain | Method | fgt_R↓ | ret_R↑ | MMLU↑ | HM↑ |
| --- | --- | --- | --- | --- | --- |
| Copyright | SimNPO | 0.048 | 0.317 | 0.435 | 0.461 |
| Copyright | PDU | 0.011 | 0.285 | 0.442 | 0.442 |
| Copyright | **BLADE** | **0.040** | **0.332** | **0.449** | **0.477** |
| Privacy | SimNPO | 0.302 | 0.557 | 0.438 | 0.544 |
| Privacy | PDU | 0.022 | 0.452 | 0.444 | 0.546 |
| Privacy | **BLADE** | **0.128** | **0.608** | **0.462** | **0.605** |

本仓库对照论文 **BLADE 行**。验收：HM 绝对偏差 ±0.03。只跑 1 个 seed 须在 notes 注明。

## 本仓库实测

<!-- MEASURED:START -->
| run | split | seed | MU | Prob | RG | FQ | HM_paper | HM_paper_ref | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 未复现 | TOFU-1B-01 |  |  |  |  |  |  | 0.808 | 未复现 |
| 未复现 | TOFU-1B-05 |  |  |  |  |  |  | 0.800 | 未复现 |
| 未复现 | TOFU-1B-10 |  |  |  |  |  |  | 0.803 | 未复现 |
| 未复现 | TOFU-3B-01 |  |  |  |  |  |  | 0.846 | 未复现 |
| 未复现 | TOFU-3B-05 |  |  |  |  |  |  | 0.842 | 未复现 |
| 未复现 | TOFU-3B-10 |  |  |  |  |  |  | 0.836 | 未复现 |
| 未复现 | MUSE-Books |  |  |  |  |  |  | 0.818 | 未复现 |
| 未复现 | MUSE-News |  |  |  |  |  |  | 0.544 | 未复现 |
<!-- MEASURED:END -->
