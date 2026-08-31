# OpenUnlearning 项目概述

对应原文：[`README.md`](../../README.md)

**一套易于扩展的框架，用于统一 LLM 机器遗忘（unlearning）评测基准。**

论文：[arXiv:2506.12618](https://arxiv.org/abs/2506.12618) · 代码：[locuslab/open-unlearning](https://github.com/locuslab/open-unlearning) · 模型：[Hugging Face `open-unlearning`](https://huggingface.co/open-unlearning) · 协议：MIT

---

## 概述

本仓库提供 TOFU、MUSE、WMDP 遗忘基准的高效、精简实现，并支持 12+ 遗忘方法、5+ 数据集、10+ 评测指标、7+ LLM 架构。每一类都可以继续扩展。

欢迎社区把新的基准、遗忘方法、数据集和指标合入本仓库，扩大 OpenUnlearning 的能力，通过更广泛的使用获得反馈，推动领域进展。

若本仓库或 [Hugging Face](https://huggingface.co/open-unlearning) 上的模型对你有帮助，请引用技术报告（bibtex 见原文 README）。

---

## 更新摘要

**2025-06-20**：论文 *OpenUnlearning: Accelerating LLM Unlearning via Unified Benchmarking of Methods and Metrics* 发布。要点：框架设计说明；在 450+ 公开模型上做遗忘评测的元评测（[含知识 / 不含知识的 TOFU 模型](https://huggingface.co/collections/open-unlearning/tofu-models-w-and-w-o-knowledge-6861e4d935eb99ba162e55cd)、[遗忘后的 TOFU 模型](https://huggingface.co/collections/open-unlearning/tofu-unlearned-models-6860f6cf3fe35d0223d92e88)）；在 TOFU 上用 10 个指标对比 8 类方法。

更早更新（节选）：

- 增加 UNDIAL、AltPO；支持 WMDP 与 Zephyr；接入 `lm-evaluation-harness`（WMDP、MMLU、GSM8K 等）。
- 增加 6 种成员推断（MIA：LOSS、ZLib、Reference、GradNorm、MinK、MinK++），以及 Extraction Strength（ES）、Exact Memorization（EM）；TOFU 增加 holdout 与 MIA，可计算 MUSE 风格的 privleak。
- 增加 RMU（基于表示工程的遗忘）。
- **2025-02-27**：本仓库取代已停止维护的原版 TOFU 代码 [`github.com/locuslab/tofu`](https://github.com/locuslab/tofu)。

合并新版本后请重新运行数据准备脚本，刷新评测 log，以兼容最新指标。

---

## 可用组件

| 组件 | 选项 |
|------|------|
| **基准** | [TOFU](https://arxiv.org/abs/2401.06121)、[MUSE](https://muse-bench.github.io/)、[WMDP](https://www.wmdp.ai/) |
| **遗忘方法** | GradAscent、GradDiff、NPO、SimNPO、DPO、RMU、UNDIAL、AltPO、SatImp、WGA、CE-U、PDU |
| **评测指标** | 逐字概率、逐字 ROUGE、知识 QA-ROUGE、Model Utility、Forget Quality、TruthRatio、Extraction Strength、Exact Memorization、6 种 MIA、[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) |
| **数据集** | MUSE-News（BBC）、MUSE-Books（Harry Potter）、TOFU（多种划分）、WMDP-Bio、WMDP-Cyber |
| **模型族** | TOFU：Llama-3.2、Llama-3.1、Llama-2；MUSE：Llama-2；另外：Phi-3.5、Phi-1.5、Gemma、Zephyr |

---

## 快速开始

```bash
conda create -n unlearning python=3.11
conda activate unlearning
pip install ".[lm-eval]"
pip install --no-build-isolation flash-attn==2.6.3

# 英文 README 写的是 python setup_data.py --eval
# 本仓库实际脚本与参数如下：
python setup_data.py --eval_logs
# 将官方模型的评测 log（含 retain 对照）下载到 saves/eval
# 其他选项：python setup_data.py --help
```

---

## 更新后的 TOFU 基准

目标模型已换成更多、更新的架构，规模约 1B–8B：Llama 3.2 1B、Llama 3.2 3B、Llama 3.1 8B，以及原版 TOFU 中的 Llama-2 7B（重训）。每种架构都在 TOFU 的 `full`、`retain90`、`retain95`、`retain99` 上微调，共 16 个模型。`full` 是遗忘起点（target）；其余是各 forget 划分对应的 retain 对照。模型见 [HuggingFace 集合](https://huggingface.co/collections/open-unlearning/tofu-new-models-67bcf636334ea81727573a9f0)，路径可在实验配置或命令行中覆盖。

---

## 运行实验

通过 Hydra 配置。更细的参数覆盖、分布式、普通微调见 [实验配置与运行](./experiments.md)。

### 执行遗忘

在 TOFU `forget10` 上用 GradAscent：

```bash
python src/train.py --config-name=unlearn.yaml experiment=unlearn/tofu/default \
  forget_split=forget10 retain_split=retain90 trainer=GradAscent task_name=SAMPLE_UNLEARN
```

- `experiment`：[`configs/experiment/unlearn/tofu/default.yaml`](../../configs/experiment/unlearn/tofu/default.yaml)（训练集、评测基准、模型路径等）。
- `forget_split` / `retain_split`：forget / retain 划分。
- `trainer`：加载 [`configs/trainer/GradAscent.yaml`](../../configs/trainer/GradAscent.yaml)，实现见 [`src/trainer/unlearn/grad_ascent.py`](../../src/trainer/unlearn/grad_ascent.py)。

### 执行评测

在 TOFU `forget10` 上评测：

```bash
model=Llama-3.2-1B-Instruct
python src/eval.py --config-name=eval.yaml experiment=eval/tofu/default \
  model=${model} \
  model.model_args.pretrained_model_name_or_path=open-unlearning/tofu_${model}_full \
  retain_logs_path=saves/eval/tofu_${model}_retain90/TOFU_EVAL.json \
  task_name=SAMPLE_EVAL
```

- `experiment`：[`configs/experiment/eval/tofu/default.yaml`](../../configs/experiment/eval/tofu/default.yaml)
- `model`：该架构的模型与 tokenizer 配置
- `pretrained_model_name_or_path`：HuggingFace ID 或本地 checkpoint
- `retain_logs_path`：retain 对照模型的评测 log，用于 `forget_quality` 等相对指标

细节见 [评测](./evaluation.md)。

### 基线脚本

在 TOFU、MUSE 上跑标准基线。预期数字见 [复现结果](./repro.md)。

```bash
bash scripts/tofu_unlearn.sh
bash scripts/muse_unlearn.sh
```

脚本使用默认超参、未经调优。调参结果可写入 [`community/leaderboard.md`](../../community/leaderboard.md)。

---

## 如何贡献

见 [贡献指南](./contributing.md)。

## 更多文档

| 文档 | 内容 |
|------|------|
| [贡献指南](./contributing.md) | 新增方法、基准、Trainer、指标、模型、数据集 |
| [评测](./evaluation.md) | 指标与 benchmark 的编写与运行 |
| [实验配置与运行](./experiments.md) | 实验配置、分布式、微调、参数覆盖 |
| [Hydra 用法](./hydra.md) | 本仓库使用的 Hydra 特性 |
| [`community/leaderboard.md`](../../community/leaderboard.md) | 社区排行榜（英文） |
| [文献与链接](./links.md) | 论文与资源 |
| [复现结果](./repro.md) | 未调参复现数字 |

维护者：Vineeth Dorna（[@Dornavineeth](https://github.com/Dornavineeth)）、Anmol Mekala（[@molereddy](https://github.com/molereddy)）。问题请在仓库提 Issue。

实现参考了 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)，并基于 [TOFU](https://github.com/locuslab/tofu) 与 [MUSE](https://github.com/swj0419/muse_bench) 重写。
