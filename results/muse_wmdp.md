# MUSE / WMDP 本机复现

更新时间：2026-09-05T07:30:47+08:00

口径见 `docs/zh/【3】muse-wmdp-repro.md`。**不进** `ou_table3_runs.jsonl`。

SOTA 列为 `docs/repro.md` 官方对照（KnowMem_Df / KnowMem_Dr / VerbMem_Df / PrivLeak）；后四列为本机复现。

## 本轮

### MUSE

| 基座名称 | SOTA | 代码链接 | 方法 | KnowMem_Df↓ | KnowMem_Dr↑ | VerbMem_Df↓ | PrivLeak→0 | 全称 | 年份/出处 | Batch | LR | Epoch | GPU型号×数量 |
|---|---|---|---|---:|---:|---:|---:|---|---|---|---|---|---|
| Llama-2-7B (MUSE-Books) | 0 / 0 / 0 / -0.67 | https://github.com/locuslab/open-unlearning | GradAscent | 0.0000 | 0.0000 | 0.0000 | -24.3343 | Gradient Ascent | 经典遗忘基线 | 32 (4×4×2) | 1e-5 | 10 | A800×2 |
| Llama-2-7B (MUSE-Books) | 0.18 / 0.30 / 0.16 / -37.79 | https://github.com/locuslab/open-unlearning | GradDiff | 0.0000 | 0.0000 | 0.0000 | -24.7689 | Gradient Difference | Liu et al., 2022 | 32 (4×4×2) | 1e-5 | 10 | A800×2 |
| Llama-2-7B (MUSE-Books) | 0.32 / 0.55 / 0.84 / -54.24 | https://github.com/locuslab/open-unlearning | NPO | 0.3687 | 0.6120 | 0.5265 | -53.7352 | Negative Preference Optimization | Zhang et al., 2024 | 32 (4×4×2) | 1e-5 | 10 | A800×2 |
| Llama-2-7B (MUSE-Books) | 0.32 / 0.54 / 0.84 / -54.26 | https://github.com/locuslab/open-unlearning | SimNPO | 0.3568 | 0.5386 | 0.2412 | -53.2359 | Simple Negative Preference Optimization | Fan et al., 2024 | 32 (4×4×2) | 1e-5 | 10 | A800×2 |
| Llama-2-7B (MUSE-Books) | 0.29 / 0.48 / 0.79 / -60.52 | https://github.com/locuslab/open-unlearning | RMU | 0.0040 | 0.0185 | 0.0111 | -12.5555 | Representation Misdirection for Unlearning | Li et al., 2024 | 32 (4×4×2) | 1e-5 | 10 | A800×2 |
| Llama-2-7B (MUSE-Books) | — | https://github.com/locuslab/open-unlearning | CEU | 0.0000 | 0.0000 | 1.45e-04 | -58.8388 | Cross-Entropy Unlearning | 仓库扩展方法 | 32 (4×4×2) | 1e-5 | 10 | A800×2 |
| Llama-2-7B (MUSE-Books) | — | https://github.com/locuslab/open-unlearning | PDU | 0.0000 | 0.0363 | 0.0047 | -54.9464 | Primal-Dual Unlearning | community/methods/PDU | 32 (4×4×2) | 1e-5 | 10 | A800×2 |
| Llama-2-7B (MUSE-News) | 0 / 0 / 0 / 52.11 | https://github.com/locuslab/open-unlearning | GradAscent | 0.0000 | 0.0000 | 0.0000 | 51.6058 | Gradient Ascent | 经典遗忘基线 | 32 (4×4×2) | 1e-5 | 10 | A800×2 |
| Llama-2-7B (MUSE-News) | 0.41 / 0.37 / 8.92e-03 / 93.23 | https://github.com/locuslab/open-unlearning | GradDiff | 0.2316 | 0.1857 | 0.0663 | 58.2704 | Gradient Difference | Liu et al., 2022 | 32 (4×4×2) | 1e-5 | 10 | A800×2 |
| Llama-2-7B (MUSE-News) | 0.56 / 0.51 / 0.35 / -86.00 | https://github.com/locuslab/open-unlearning | NPO | 0.5541 | 0.4927 | 0.3357 | -84.6767 | Negative Preference Optimization | Zhang et al., 2024 | 32 (4×4×2) | 1e-5 | 10 | A800×2 |
| Llama-2-7B (MUSE-News) | 0.54 / 0.51 / 0.36 / -86.11 | https://github.com/locuslab/open-unlearning | SimNPO | 0.5895 | 0.4401 | 0.4086 | -99.8950 | Simple Negative Preference Optimization | Fan et al., 2024 | 32 (4×4×2) | 1e-5 | 10 | A800×2 |
| Llama-2-7B (MUSE-News) | 0.48 / 0.51 / 0.05 / 56.36 | https://github.com/locuslab/open-unlearning | RMU | 0.5590 | 0.4902 | 0.1914 | 2.3720 | Representation Misdirection for Unlearning | Li et al., 2024 | 32 (4×4×2) | 1e-5 | 10 | A800×2 |

### WMDP

| 基座名称 | SOTA | 代码链接 | 方法 | Bio Acc↓ | MMLU↑ | Cyber Acc↓ | MT-Bench↑ | 全称 | 年份/出处 | Batch | LR | Epoch | GPU型号×数量 |
|---|---|---|---|---:|---:|---:|---:|---|---|---|---|---|---|
| Zephyr-7B-β (WMDP-cyber) | — | https://github.com/locuslab/open-unlearning | GradAscent | — | 0.2689 | 0.2456 | — | Gradient Ascent | 经典遗忘基线 | 16 (1×8×2) | 5e-5 | 80 steps | A800×2 |
| Zephyr-7B-β (WMDP-cyber) | — | https://github.com/locuslab/open-unlearning | GradDiff | — | 0.2551 | 0.2456 | — | Gradient Difference | Liu et al., 2022 | 16 (1×8×2) | 5e-5 | 80 steps | A800×2 |
| Zephyr-7B-β (WMDP-cyber) | — | https://github.com/locuslab/open-unlearning | NPO | — | 0.2551 | 0.2456 | — | Negative Preference Optimization | Zhang et al., 2024 | 16 (1×8×2) | 5e-5 | 80 steps | A800×2 |
| Zephyr-7B-β (WMDP-cyber) | — | https://github.com/locuslab/open-unlearning | RMU | — | 0.5540 | 0.2778 | — | Representation Misdirection for Unlearning | Li et al., 2024 | 16 (1×8×2) | 5e-5 | 80 steps | A800×2 |
| Zephyr-7B-β (WMDP-cyber) | — | https://github.com/locuslab/open-unlearning | SimNPO | — | 0.2349 | 0.2632 | — | Simple Negative Preference Optimization | Fan et al., 2024 | 16 (1×8×2) | 5e-5 | 80 steps | A800×2 |

## 此前（V100）

单卡 fp16 + Adafactor，有效 batch 32（1×32×1）。官方对照日志标在方法名里。

| 基座名称 | SOTA | 代码链接 | 方法 | KnowMem_Df↓ | KnowMem_Dr↑ | VerbMem_Df↓ | PrivLeak→0 | 全称 | 年份/出处 | Batch | LR | Epoch | GPU型号×数量 |
|---|---|---|---|---:|---:|---:|---:|---|---|---|---|---|---|
| Llama-2-7B (MUSE-News) | 0 / 0 / 0 / 52.11 | https://github.com/locuslab/open-unlearning | GradAscent | 0.0000 | 0.0000 | 0.0000 | 54.6809 | Gradient Ascent | 经典遗忘基线 | 32 (1×32×1) | 1e-5 | 10 | V100×1 |
| Llama-2-7B (MUSE-News) | 0.41 / 0.37 / 8.92e-03 / 93.23 | https://github.com/locuslab/open-unlearning | GradDiff | 0.3268 | 0.2607 | 0.1007 | 36.4190 | Gradient Difference | Liu et al., 2022 | 32 (1×32×1) | 1e-5 | 10 | V100×1 |
| Llama-2-7B (MUSE-News) | 0.54 / 0.51 / 0.36 / -86.11 | https://github.com/locuslab/open-unlearning | SimNPO | 0.5550 | 0.3841 | 0.3468 | -99.8321 | Simple Negative Preference Optimization | Fan et al., 2024 | 32 (1×32×1) | 1e-5 | 10 | V100×1 |
| Llama-2-7B (MUSE-News) | — | https://github.com/locuslab/open-unlearning | CEU | 0.0000 | 0.0000 | 0.0000 | 21.1797 | Cross-Entropy Unlearning | 仓库扩展方法 | 32 (1×32×1) | 1e-5 | 10 | V100×1 |
| Llama-2-7B (MUSE-News) | — | https://github.com/locuslab/open-unlearning | PDU | 0.0000 | 0.0000 | 0.0000 | — | Primal-Dual Unlearning | community/methods/PDU | 32 (1×32×1) | 1e-5 | 10 | V100×1 |
| Llama-2-7B (MUSE-News) | — | https://github.com/locuslab/open-unlearning | Finetuned | 0.6443 | 0.5552 | 0.5789 | -99.8111 | Finetuned target（未遗忘） | MUSE 官方对照日志 | — | — | — | — |
| Llama-2-7B (MUSE-News) | — | https://github.com/locuslab/open-unlearning | Retain | 0.3279 | 0.5602 | 0.2016 | -4.7200 | Retrain on retain only | MUSE 官方对照日志 | — | — | — | — |
| Llama-2-7B (MUSE-Books) | 0 / 0 / 0 / -0.67 | https://github.com/locuslab/open-unlearning | GradAscent | 0.0000 | 0.0000 | 0.0000 | -14.7744 | Gradient Ascent | 经典遗忘基线 | 32 (1×32×1) | 1e-5 | 10 | V100×1 |
| Llama-2-7B (MUSE-Books) | 0.18 / 0.30 / 0.16 / -37.79 | https://github.com/locuslab/open-unlearning | GradDiff | 0.0000 | 0.0000 | 0.0000 | -14.4786 | Gradient Difference | Liu et al., 2022 | 32 (1×32×1) | 1e-5 | 10 | V100×1 |
| Llama-2-7B (MUSE-Books) | 0.32 / 0.54 / 0.84 / -54.26 | https://github.com/locuslab/open-unlearning | SimNPO | 0.4149 | 0.6933 | 0.3386 | -62.0932 | Simple Negative Preference Optimization | Fan et al., 2024 | 32 (1×32×1) | 1e-5 | 10 | V100×1 |
| Llama-2-7B (MUSE-Books) | — | https://github.com/locuslab/open-unlearning | Finetuned | 0.4712 | 0.6913 | 0.9970 | -57.3410 | Finetuned target（未遗忘） | MUSE 官方对照日志 | — | — | — | — |
| Llama-2-7B (MUSE-Books) | — | https://github.com/locuslab/open-unlearning | Retain | 0.3029 | 0.6874 | 0.1445 | 8.1600 | Retrain on retain only | MUSE 官方对照日志 | — | — | — | — |
