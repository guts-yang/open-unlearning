# GROM 远程编辑与评测

论文：*GROM: Gradient-Free Rapid One-Shot Machine Unlearning*（arXiv:2608.05783）。官方代码 pin **Batorskq/GROM@6dd9591**。闭式岭回归编辑 MLP `down_proj`，**无梯度、无迭代**。不要和同作者的 **EvoMU**（进化搜索）混淆。

Seed **0**。评测走本仓库 `workspace/scripts/eval.py`（OpenUnlearning pipeline）。本 fork **没有** `experiment=eval/wmdp/bio_l3`，WMDP 用 `experiment=eval/wmdp/default data_split=bio`。

## 官方设置（跟 yaml，不要猜）

| 任务 | 基座 | 层 | target | strength | w_r | 论文 Final | Time |
|------|------|----|--------|----------|-----|------------|------|
| TOFU-5% | `tofu_Llama-2-7b-chat-hf_full` | 26–31 | suppress | 1000 | 300 | 0.79 | 1.8 min |
| TOFU-10% | `tofu_Llama-3.2-1B-Instruct_full` | 11–15 | suppress | 65 | 100 | 0.75 | 0.5 min |
| MUSE News | `MUSE-News_target` | 28–31 | suppress | 370 | 10 | 55.93 | 0.3 min |
| MUSE Books | `MUSE-Books_target`（勿用 README 的 ICLM-7B 口误） | 28–31 | suppress | 350 | 80 | 76.20 | 0.3 min |
| WMDP-Bio | `Meta-Llama-3-8B-Instruct` | 8 | rmu | 45 | 1000 | 0.63 | 0.2 min |

TOFU-10% 官方是 **1B**，与 BLADE 3B 矩阵不对齐，实测表备注列写明。rho 一律 0.03。Wall-clock 只记闭式编辑（`grom_edit.json` 的 `elapsed_min`），不含加载与评测。论文时间为单卡 H100。

## 编辑

```bash
python workspace/scripts/train.py experiment=unlearn/grom_tofu10 \
  paths.output_dir=workspace/saves/unlearn/grom_tofu10
# 或
python workspace/scripts/run_grom.py tofu10
```

其余：`grom_tofu05`、`grom_muse_news`、`grom_muse_books`、`grom_wmdp_bio`。

MUSE 语料默认 `data/muse_news/*.jsonl`、`data/muse_books/*.jsonl`（GROM 官方路径）。若本地只有 HuggingFace MUSE，用 `--set` 式 Hydra 覆盖 `trainer.method_args.forget_jsonl=...`。

WMDP-Bio forget jsonl **门控**。没有 `data/wmdp/bio-forget-corpus.jsonl` 则该格写「未复现」。

## 评测

TOFU（口径是 1−Rouge / 1−Prob / 1−Extr + MU，**不是 FQ**）：

```bash
python workspace/scripts/eval.py experiment=eval/tofu/default \
  model.model_args.pretrained_model_name_or_path=workspace/saves/unlearn/grom_tofu10 \
  eval.tofu.retain_logs_path=<retain90 TOFU_EVAL.json> \
  paths.output_dir=workspace/saves/unlearn/grom_tofu10_eval
```

MUSE：`experiment=eval/muse/default`。WMDP：`experiment=eval/wmdp/default data_split=bio`。

汇总：`python workspace/scripts/aggregate_grom.py`。

## 可选：4-bit NF4

论文 Table 6：同一 OpenUnlearning 评测，ckpt 用 bitsandbytes 4-bit 加载。对照 SimNPO 量化后遗忘回弹。本轮不实现加载脚本，有带宽再补。

ZsRE 不做。
