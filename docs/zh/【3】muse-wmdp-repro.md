# MUSE / WMDP 全方法复现

> 分支：`repro/ou-table3`　|　2026-09-03　|　超参对齐 OU 官方复现页 + Hydra 实验默认，**不是** Appendix F.2 的 TOFU-1B 网格。
> 入口：`bash scripts/run_muse_wmdp_matrix.sh`（P0）或 `--full`。
> 结果：`results/muse_wmdp_runs.jsonl`、`results/muse_wmdp.md`。**禁止**写入 `ou_table3_runs.jsonl`。  
> 2026-09-04 已停跑；未完成清单见 [`【4】muse-wmdp-remaining.md`](【4】muse-wmdp-remaining.md)。

## 一、超参出处

PDF 正文 Table 3/6 与 Appendix F.2 只覆盖 **TOFU + Llama-3.2-1B**。MUSE/WMDP 用仓库公开发布口径：

| 基准 | 文件 | 锁死值 |
|---|---|---|
| MUSE | `docs/repro.md` + `configs/experiment/unlearn/muse/default.yaml` | lr=1e-5，constant，10 epoch，有效 batch 32（4×4×2 卡） |
| WMDP-cyber | `configs/experiment/unlearn/wmdp/default.yaml` | lr=5e-5，constant，**max_steps=80**，1×16 |
| PDU-MUSE | `community/methods/PDU/run.sh` | News `retain_loss_eps=1.5`，Books `0.1`，pref=50，warmup=3 |

方法默认 α/β：GradDiff/NPO/DPO `alpha=1 gamma=1`，NPO/DPO `beta=0.1`；SimNPO yaml（β=4.5, δ=0, γ=0.125）；RMU-MUSE `layers.7` + `steering_coeff=2`（全参）；RMU-WMDP 另锁 `layers.(5\|6\|7).mlp.down_proj`。

本机约束（必须记入结果 `hyper`）：

- V100 无 bf16；官方 DeepSpeed json 开了 bf16。
- AutoDL **cgroup RAM=50GiB**。7B AdamW 一阶+二阶约 56GiB，ZeRO3 CPU Adam 一分配即 SIGKILL。
- 脚本因此用 `configs/accelerate/v100_single.yaml`（单进程 fp16，**不用 DeepSpeed**）+ `optim=adafactor`，MUSE 有效 batch 仍为 32（V100 上 `1×32×1`，因 `4×2048` backward OOM）。这与 repro 的 AdamW / 2 卡 ZeRO3 不完全同实现，是本机唯一能跑完的口径。
- `--full` 默认 **两卡并行、每卡一组不同实验**（claim 锁 + 跳过已 ok / 进程中 / 本轮已 fail）。不是把同一实验拆到两张卡。NPO/SimNPO/DPO 要参考模型，单卡 32GB 可能 OOM，失败记 `.done_fail` 不重试。

## 二、数据划分

| 基准 | 模型（必须钉死，勿只写 `model=Llama-2-7b-hf`） | forget / retain |
|---|---|---|
| MUSE-News | `muse-bench/MUSE-News_target` | `raw` forget / retain1 |
| MUSE-Books | `muse-bench/MUSE-Books_target` | 同上 |
| WMDP-cyber | `HuggingFaceH4/zephyr-7b-beta` | `data/wmdp/wmdp-corpora/cyber-{forget,retain}-corpus.jsonl` |

Tokenizer：MUSE target 仓无词表，用数据盘缓存的 `NousResearch/Llama-2-7b-hf`（`configs/model/Llama-2-7b-hf.yaml` 已改）。

不做：scal/sust、WMDP bio/chem、TPO/AltPO/Idk*、S3 corpus、F.2 27 点搜索。

## 三、矩阵

**P0**（`run_muse_wmdp_matrix.sh`）：News × {GradAscent, GradDiff, NPO, SimNPO, RMU} + WMDP RMU。

**--full**：Books 五方法 + News/Books × {DPO, UNDIAL, CEU, WGA, SatImp, PDU} + WMDP × {GradAscent, GradDiff, NPO, SimNPO, UNDIAL, CEU, WGA, SatImp}。

DPO 需要 forget `alternate` 字段，MUSE `PretrainingDataset` 可能训失败，失败写入 jsonl 的 `train_fail` 后继续。

评完默认删 7B 权重（`KEEP_CKPT=1` 可留），只留 `evals/`。

## 四、评测

- MUSE：forget/retain KnowMem、VerbMem、PrivLeak、ES；对照官方 `saves/eval/muse_*_{retrain,target}`。
- WMDP：`wmdp_cyber` acc + `mmlu`。不跑 `ou_aggregate.py`。

## 五、命令

```bash
source /root/autodl-tmp/env_hf.sh
source /root/autodl-tmp/envs/unlearning/bin/activate
# 默认两卡并行、每卡一组不同实验（不是把同一实验拆两卡）
nohup bash scripts/run_muse_wmdp_matrix.sh --full \
  > /root/autodl-tmp/logs/muse_wmdp_matrix.nohup 2>&1 &
# 单卡排队：bash scripts/run_muse_wmdp_matrix.sh --full --serial
# KEEP_CKPT=1 bash scripts/muse_unlearn_one.sh NPO News
```
