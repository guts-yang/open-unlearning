# 原论文结果 vs 本仓库实测

原文 PDF 在 [`workspace/docs/papers/`](../docs/papers/)。各方法文件先抄论文主表，再留实测网格；缺跑不得把论文数字填进实测列。

| 文件 | 论文 | PDF |
| --- | --- | --- |
| [blade_table.md](blade_table.md) | BLADE（arXiv:2608.22557） | 【172】BLADE- Bilevel Low-rank Augmented-Lagrangian Erasure for LLM Unlearning.pdf |
| [baldro_table.md](baldro_table.md) | BalDRO（WWW 2026, arXiv:2601.09172） | 【168】BalDRO-A Distributionally Robust Optimization based Framework for Large Language Model Unlearning.pdf |
| [alter_table.md](alter_table.md) | ALTER（AAAI 2026, arXiv:2603.01792） | 【173】ALTER- Asymmetric LoRA for Token-Entropy-Guided Unlearning of LLMs.pdf |

`python workspace/scripts/aggregate_*.py` 只改各文件里 `<!-- MEASURED:START -->` 到 `END` 的实测表。
