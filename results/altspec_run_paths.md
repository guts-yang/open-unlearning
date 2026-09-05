# AltSpec 训练结果保存路径

根目录：`SAVES=/root/autodl-tmp/saves`  
流水线日志：`/root/autodl-tmp/logs/altspec_plan.nohup`  
汇总表：`results/specdiff_tofu.md`、`results/specdiff_tofu.jsonl`  
每条 run 完成后写入 `{ckpt}/evals/ou_aggregate.json`；`scripts/specdiff_table.py` 扫 `tofu_1B_SpecDiff_forget10_*`（跳过 `smoke_s0`、`warmup_only_s0`）。

权重与 tokenizer 在 `{ckpt}/`（`model.safetensors`、`config.json`）；Hydra 在 `{ckpt}/.hydra/config.yaml`。

## 本轮队列

| 阶段 | tag | 权重/评测目录 | 完成判据 |
|---|---|---|---|
| A warmup-only | `warmup_only_s0` | `/root/autodl-tmp/saves/unlearn/tofu_1B_SpecDiff_forget10_warmup_only_s0` | `evals/ou_aggregate.json` |
| B lr=2e-5 seed1 | `g_lr2e-5_lam1_b0.1_k0.3_ep10_seed1` | `/root/autodl-tmp/saves/unlearn/tofu_1B_SpecDiff_forget10_g_lr2e-5_lam1_b0.1_k0.3_ep10_seed1` | 同上 |
| B lr=2e-5 seed2 | `g_lr2e-5_lam1_b0.1_k0.3_ep10_seed2` | `/root/autodl-tmp/saves/unlearn/tofu_1B_SpecDiff_forget10_g_lr2e-5_lam1_b0.1_k0.3_ep10_seed2` | 同上 |
| C alt 数据 | — | `/usr/local/open-unlearning/community/methods/AltPO/data/tofu_Llama-3.2-1B-Instruct_full/forget10/alt5_seed_0.json` | 文件已存在则 skip generate |
| D V1 DPO warmup | `altspec_v1_wudpo2_seed0` | `/root/autodl-tmp/saves/unlearn/tofu_1B_SpecDiff_forget10_altspec_v1_wudpo2_seed0` | `evals/ou_aggregate.json` |
| E V2 DPO lr=5e-5 | `altspec_v2_dpo0.5_lr5e-5_seed0` | `/root/autodl-tmp/saves/unlearn/tofu_1B_SpecDiff_forget10_altspec_v2_dpo0.5_lr5e-5_seed0` | 同上 |
| E V2 DPO lr=2e-5 | `altspec_v2_dpo0.5_lr2e-5_seed0` | `/root/autodl-tmp/saves/unlearn/tofu_1B_SpecDiff_forget10_altspec_v2_dpo0.5_lr2e-5_seed0` | 同上 |

评测产物（每条 ckpt 下）：

- `evals/TOFU_EVAL.json`、`evals/TOFU_SUMMARY.json`
- `evals/ou_aggregate.json`（Mem / Priv / Utility / Agg / SpecGap）

## 已有基线（不重跑）

- lr=2e-5 κ=0.3 seed0：`/root/autodl-tmp/saves/unlearn/tofu_1B_SpecDiff_forget10_g_lr2e-5_lam1_b0.1_k0.3_ep10`
- draft / 训练起点：`open-unlearning/tofu_Llama-3.2-1B-Instruct_full`（HF cache：`/root/autodl-tmp/huggingface/hub/models--open-unlearning--tofu_Llama-3.2-1B-Instruct_full`）
- retain 参考日志：`saves/eval/tofu_Llama-3.2-1B-Instruct_retain90/TOFU_EVAL.json`

启动：`bash scripts/after_specdiff_continue.sh`（已完成的 `ou_aggregate.json` 会 skip）。

主图：`results/specdiff_fig_main.png`（`python scripts/specdiff_paper_fig.py`，依赖 `specdiff_tofu.jsonl`）。

## MUSE / WMDP 转移（未开训，等 TOFU 队列空）

脚本：`bash scripts/specdiff_muse_wmdp.sh preflight|warmup-news|news|books|wmdp|all`  
日志建议：`/root/autodl-tmp/logs/specdiff_muse_wmdp.nohup`  
记录：`results/muse_wmdp_runs.jsonl` → `results/muse_wmdp.md`  
超参：κ=0.3, λ=1, β=0.1, τ=0.02, warmup=1 GradDiff；**不用 AltPO**。MUSE lr=1e-5 / 10 epoch；WMDP lr=5e-5 / 80 step。draft = 各基准训练起点。

| 阶段 | 判据文件 |
|---|---|
| News warmup-only | `/root/autodl-tmp/saves/unlearn/muse_Llama-2-7b-hf_News_SpecDiff_warmup_only/evals/MUSE_SUMMARY.json` |
| News full | `/root/autodl-tmp/saves/unlearn/muse_Llama-2-7b-hf_News_SpecDiff_k0.3_lr1e-5_ep10/evals/MUSE_SUMMARY.json` |
| Books full | `/root/autodl-tmp/saves/unlearn/muse_Llama-2-7b-hf_Books_SpecDiff_k0.3_lr1e-5_ep10/evals/MUSE_SUMMARY.json` |
| WMDP-cyber | `/root/autodl-tmp/saves/unlearn/wmdp_zephyr-7b-beta_cyber_SpecDiff_k0.3_lr5e-5_ms80/evals/LMEval_SUMMARY.json` |

