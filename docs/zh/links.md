# 文献与链接

对应原文：[`docs/links.md`](../links.md)

本仓库已实现功能所对应的论文与资源。欢迎补全缺漏。

[OpenUnlearning: Accelerating LLM Unlearning via Unified Benchmarking of Methods and Metrics](https://arxiv.org/abs/2506.12618) 介绍了：框架技术报告；在 450+ 公开模型上的遗忘评测元评测；在 TOFU 上用 10 个指标对比 8 类方法。

## 已实现方法

| 方法 | 资源 |
|------|------|
| GradAscent、GradDiff | 多篇工作中的朴素基线（含 MUSE、TOFU） |
| NPO | [论文](https://arxiv.org/abs/2404.05868)，[代码](https://github.com/licong-lin/negative-preference-optimization) |
| SimNPO | [论文](https://arxiv.org/abs/2410.07163)，[代码](https://github.com/OPTML-Group/Unlearn-Simple) |
| IdkDPO | [TOFU](https://arxiv.org/abs/2401.06121) |
| RMU | [WMDP](https://www.wmdp.ai/)，[代码](https://github.com/centerforaisafety/wmdp/tree/main/rmu)；后又用于 [G-effect](https://github.com/tmlr-group/G-effect/blob/main/dataloader.py) |
| UNDIAL | [论文](https://arxiv.org/pdf/2402.10052)，[代码](https://github.com/dong-river/LLM_unlearning/tree/main) |
| AltPO | [论文](https://arxiv.org/pdf/2409.13474)，[代码](https://github.com/molereddy/Alternate-Preference-Optimization) |
| SatImp | [论文](https://arxiv.org/pdf/2505.11953)，[代码](https://github.com/Puning97/SatImp-for-LLM-Unlearning) |
| WGA（G-effect） | [论文](https://arxiv.org/pdf/2502.19301)，[代码](https://github.com/tmlr-group/G-effect) |
| CE-U（交叉熵遗忘） | [论文](https://arxiv.org/pdf/2503.01224) |
| PDU | [论文](https://arxiv.org/abs/2506.05314) |

## 基准

| 基准 | 资源 |
|------|------|
| TOFU | [论文](https://arxiv.org/abs/2401.06121) |
| MUSE | [论文](https://arxiv.org/abs/2407.06460) |
| WMDP | [论文](https://arxiv.org/abs/2403.03218) |

## 评测指标

| 指标 | 资源 |
|------|------|
| 逐字概率 / ROUGE、简单 QA-ROUGE | 多篇工作中的朴素指标（含 MUSE、TOFU） |
| MIA（LOSS、ZLib、Reference、GradNorm、MinK、MinK++） | [MIMIR](https://github.com/iamgroot42/mimir)，[MUSE](https://arxiv.org/abs/2407.06460) |
| PrivLeak | [MUSE](https://arxiv.org/abs/2407.06460) |
| Forget Quality、Truth Ratio、Model Utility | [TOFU](https://arxiv.org/abs/2401.06121) |
| Extraction Strength（ES） | Carlini et al., 2021；遗忘场景见 Wang et al., 2025 |
| Exact Memorization（EM） | Tirumala et al., 2022；遗忘场景见 Wang et al., 2025 |
| lm-evaluation-harness | [仓库](https://github.com/EleutherAI/lm-evaluation-harness) |

## 其他资料

综述：

- [Machine Unlearning in 2024](https://ai.stanford.edu/~kzliu/blog/unlearning)
- [Rethinking Machine Unlearning for Large Language Models](https://arxiv.org/abs/2402.08787)

相关仓库：

- [TOFU 原版](https://github.com/locuslab/tofu)
- [MUSE 原版](https://github.com/swj0419/muse_bench)
- [Unlearning Comparator](https://github.com/gnueaj/Machine-Unlearning-Comparator)
- [Awesome LLM Unlearning](https://github.com/chrisliu298/awesome-llm-unlearning)
- [Awesome Machine Unlearning](https://github.com/tamlhp/awesome-machine-unlearning)
- [Awesome GenAI Unlearning](https://github.com/franciscoliu/Awesome-GenAI-Unlearning)
