# ALTER

论文：*ALTER: Asymmetric LoRA for Token-Entropy-Guided Unlearning of LLMs*（AAAI 2026，arXiv:2603.01792v1）。原文：[workspace/docs/papers/【173】ALTER- Asymmetric LoRA for Token-Entropy-Guided Unlearning of LLMs.pdf](../docs/papers/【173】ALTER- Asymmetric LoRA for Token-Entropy-Guided Unlearning of LLMs.pdf)。GitHub `MastrOrigami/ALTER` 仅空 README（`@f664526`）。本仓库实现是**按正文重建**，不能写成「官方代码数字复现」。

## 协议（论文）

- WMDP：Zephyr-7B、Llama3-8B；forget = Bio/Cyber 选择题准确率↓，retain = 全量 MMLU↑；另报 GPT-4o Flu-mean↑ / Flu-var↓（协议未公开 → 流畅度格子本仓库标未复现）。
- TOFU：Llama2-7B、Llama3-8B，1%/5%/10%；主结果在 **Figure 4**（无数值表）。
- HarryPotter / MUSE-HP：Llama-2-7B；BLEU、ROUGE-L、ASG↓，MMLU↑。OpenUnlearning MUSE 指标与此**分列**。
- 配置：η_B=1e-3，η_A=1e-5，β=γ=1.0，λ=0.01，batch=4，epoch=3；高熵阈值 Sq>1.2，τ_high=0.8，τ_low=0.01；推理条件路由，不要 merge LoRA 后再评。
- 硬件：论文 H800 80GB。

## 原论文结果

### Table 1 · WMDP（forget 选择题）+ 全量 MMLU（retain）

Llama3-8B：

| Method | WMDP-Bio↓ | WMDP-Cyber↓ | MMLU↑ | Flu-mean↑ | Flu-var↓ |
| --- | --- | --- | --- | --- | --- |
| Base | 71.2 | 45.3 | 62.1 | 2.97 | 1.91 |
| RMU | 49.4 | 37.0 | 40.1 | 2.96 | 1.88 |
| ELM | 33.3 | 26.6 | 57.2 | 3.07 | 2.18 |
| GA | 23.3 | 24.0 | 24.8 | 1.00 | 0.00 |
| RL | 24.7 | 26.6 | 23.0 | 1.00 | 0.00 |
| NPO | 58.1 | 34.4 | 50.1 | 3.07 | 1.86 |
| NPO KL | 64.3 | 41.3 | 56.0 | 2.97 | 1.96 |
| NPO GD | 56.2 | 33.1 | 51.9 | 3.03 | 2.08 |
| LoRA | 28.7 | 32.1 | 39.6 | 2.23 | 1.42 |
| AsymLoRA | 25.7 | 28.8 | 55.3 | 2.23 | 1.42 |
| **Ours (ALTER)** | **24.4** | **25.6** | **57.8** | **3.46** | **1.17** |

Zephyr-7B：

| Method | WMDP-Bio↓ | WMDP-Cyber↓ | MMLU↑ | Flu-mean↑ | Flu-var↓ |
| --- | --- | --- | --- | --- | --- |
| Base | 64.4 | 44.3 | 58.5 | 2.97 | 1.98 |
| RMU | 30.2 | 27.3 | 57.8 | 2.92 | 2.03 |
| ELM | 29.6 | 27.2 | 56.2 | 2.99 | 2.00 |
| GA | 24.7 | 26.8 | 23.0 | 1.00 | 0.00 |
| RL | 24.0 | 24.7 | 26.4 | 1.00 | 0.00 |
| NPO | 63.5 | 43.6 | 57.8 | 2.98 | 2.12 |
| NPO KL | 64.3 | 45.3 | 57.4 | 2.95 | 1.91 |
| NPO GD | 63.5 | 43.1 | 58.0 | 2.93 | 2.08 |
| LoRA | 34.2 | 32.1 | 37.4 | 2.47 | 1.57 |
| AsymLoRA | 27.1 | 26.3 | 54.1 | 2.47 | 1.57 |
| **Ours (ALTER)** | **24.4** | **24.0** | **56.4** | **3.11** | **1.33** |

正文：目标是把 WMDP 压到约随机（∼25%）同时保住 MMLU。

### Table 2 · HarryPotter（Llama-2-7B）

| Method | BLEU | R-L | ASG↓ | MMLU↑ | Ful.↑ |
| --- | --- | --- | --- | --- | --- |
| Original | 74.8 | 85.1 | 74.1 | 46.3 | 4.0 |
| Retain | 1.9 | 9.8 | 0 | 47.8 | 2.8 |
| Fine-tune | 6.4 | 17.2 | 5.9 | 46.0 | 1.9 |
| GA | 0 | 0 | 6.0 | 26.9 | 1.0 |
| GD | 3.9 | 14.5 | 3.4 | 43.6 | 1.8 |
| NPO | 1.5 | 5.3 | 2.5 | 42.7 | 2.9 |
| KL | 1.2 | 8.9 | 0.8 | 41.1 | 3.1 |
| WHP | 23.6 | 17.9 | 14.9 | 44.4 | 2.5 |
| ELM | 8.1 | 9.0 | 2.7 | 44.6 | 2.8 |
| LoRA | 7.2 | 11.5 | 3.5 | 38.9 | 2.3 |
| A-LoRA | 5.9 | 10.4 | 1.9 | 43.8 | 2.3 |
| **Ours (ALTER)** | **4.7** | **9.6** | **1.3** | **44.6** | **3.3** |

### TOFU

无数值表，见原文 Figure 4。定性：AsymLoRA/ALTER 在 1%/5%/10% 上接近 Retain 的 utility，且 forget quality 接近完全遗忘。实测不要填论文图上的目测数字。

公开可跑：WMDP-Cyber。Bio forget corpus 未齐则 Bio 训练写未复现。流畅度（GPT-4o）未复现。Llama3-8B 的 HF ID 须写入 notes。

## 本仓库实测

<!-- MEASURED:START -->
| run | model | wmdp_bio | wmdp_cyber | mmlu | note | status |
| --- | --- | --- | --- | --- | --- | --- |
| 未复现 | zephyr-7b-beta |  |  |  | 推定实现；训练未跑 | 未复现 |
| 未复现 | llama3-8b |  |  |  | 论文锚点 Bio 24.4 / Cyber 25.6 / MMLU 57.8；训练未跑 | 未复现 |
<!-- MEASURED:END -->
