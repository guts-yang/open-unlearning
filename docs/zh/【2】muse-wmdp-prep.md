# MUSE / WMDP 复现准备清单

> 分支：`repro/ou-table3`　|　撰写：2026-09-01　|　状态：**准备未开始，均可无 GPU 提前做**
> 相关文档：[TOFU 自训计划](./ou-table3-selftrain.md)　|　[P0 评估口径](./ou-table3-p0.md)　|　[上机环境](./reproduce-prep.md)

## 一、结论速览

仓库对两个基准的支持是现成的（训练/评测配置都在），但有 **5 项前置要准备，其中 4 项不依赖 GPU，现在就能做**。范围建议不做全量：

- **WMDP**：只跑 RMU 1 组（对应 OU 论文 Table 4 的做法，正好支撑「跨 TOFU↔WMDP 桥」的论据）。
- **MUSE**：只挑 SimNPO + RMU 各 1 组（News 优先）。

新增下载约 **60GB**（数据盘剩 213GB，够）。

## 二、MUSE（Llama-2-7b，News/Books）—— 需准备 3 项

| # | 准备项 | 现状 | 怎么做 |
|---|---|---|---|
| 1 | 数据集 `muse-bench/MUSE-News` + `MUSE-Books` | ❌ HF 缓存没有 | `huggingface-cli download`（走 hf-mirror，无 GPU 可做） |
| 2 | target 模型 `muse-bench/MUSE-News_target` / `Books_target` | ❌ 没有（`saves/eval/muse_*` 里只是官方对照**日志**，不是权重） | 同上，各约 13GB，无 GPU 可做 |
| 3 | retrain 对照日志 | ✅ 已有（`saves/eval/muse_Llama-2-7b-hf_{News,Books}_retrain/MUSE_EVAL.json`） | 无需动作 |

训练入口：`experiment=unlearn/muse/default`（模型默认 `muse-bench/MUSE-${data_split}_target`，forget/retain 数据直接走 HF 数据集，不需要 muse_bench 库）；现成脚本 `scripts/muse_unlearn.sh` 含 GA/GradDiff/NPO/DPO/RMU，**无 SimNPO**，要跑 SimNPO 需仿写。

默认超参：batch 4×8=32、`lr=1e-5`、`lr_scheduler=constant`、10 epoch；官方脚本单卡（`CUDA_VISIBLE_DEVICES=0`）。

成本：7B + News 语料，每组约 **1-2 小时**。

## 三、WMDP —— 需准备 2 项，有一个易踩的点

| # | 准备项 | 现状 | 怎么做 |
|---|---|---|---|
| 1 | corpus 数据 | ❌ `data/wmdp/` 不存在 | `python setup_data.py --wmdp`（wget S3 的 `wmdp-corpora.zip` + 密码 `wmdpcorpora` unzip）——只依赖网络，**无 GPU 可做** |
| 2 | `HuggingFaceH4/zephyr-7b-beta` | ❌ HF 缓存没有 | 约 14GB，**无 GPU 可做** |

**易踩的点**：`configs/experiment/unlearn/wmdp/default.yaml` 的默认模型是 **`zephyr-7b-beta`，不是 Llama-2-7b**（官方 RMU-WMDP 论文用的就是 zephyr）——缓存里已有的 `NousResearch/Llama-2-7b-hf` 对 WMDP 默认配置**用不上**。若坚持用 Llama-2-7b，要 override 模型并核对 RMU 的 `module_regex` / `trainable_params_regex`（默认锁 `layers.(5|6|7).mlp.down_proj`，zephyr 与 Llama-2 都是 32 层，恰好兼容）。

训练默认：RMU、`max_steps=80`（**按步数不按 epoch**）、batch 1×16、`lr=5e-5`、单卡口径。

评测：`lm_eval` 跑 `wmdp_${data_split}`（bio/chem/cyber）+ `mmlu` 作 utility；`lm-eval 0.4.11` 已装。**注意**：lm-eval 的 `wmdp_*` 选择题任务首跑会从 HF 拉 `cais/wmdp` MCQ 数据，建议一并预下载。

仓库**没有** WMDP 版的 `tofu_unlearn_one.sh`，需要时仿写一个 `wmdp_unlearn_one.sh`（训 → lm_eval → 落盘）。

成本：训练每组约 **0.5-1 小时**；lm-eval 评测 7B 每次约 10-20 分钟。

## 四、指标口径（不能套 TOFU 四维）

MUSE/WMDP 在 OU 论文里对应 **Table 4/5**，不是 Table 3 的 Mem/Priv/Utility/Agg 四维，`ou_aggregate.py` **不适用**：

| 基准 | 指标口径 |
|---|---|
| **MUSE**（Table 5 口径） | Forget KnowMem↑ / Retain KnowMem↑ / VerbMem↓ / PrivLeak→0；聚合分是**二维 HM(KnowMem, PrivLeak)**，与 TOFU 的三维 Agg 口径不同 |
| **WMDP**（Table 4 口径） | WMDP accuracy↓（遗忘后应降到随机水平 ~25%）+ MUSE-News KnowMem↑ 作 utility，**不聚合**，两个数分开报 |

另注意：`configs/eval/muse.yaml` 默认只启用 5 个指标（knowmem×2 / verbmem / privleak / extraction_strength），4 个 MIA 和 gibberish 都是注释状态——与 TOFU P0 改造前的情况一样。若论文需要 MUSE 的 MIA 数值，要先做一次「取消注释」的口径改造（方法与 TOFU 完全同构，改完同样要留 config diff 与 commit hash）。

## 五、提前下载命令（无 GPU，可直接跑）

```bash
source /root/autodl-tmp/env_hf.sh

# 1) WMDP corpus（wget + 密码 unzip）
python setup_data.py --wmdp

# 2) MUSE 数据集 + target 模型（约 28GB）
huggingface-cli download muse-bench/MUSE-News --repo-type dataset
huggingface-cli download muse-bench/MUSE-Books --repo-type dataset
huggingface-cli download muse-bench/MUSE-News_target
huggingface-cli download muse-bench/MUSE-Books_target

# 3) zephyr-7b-beta（约 14GB，WMDP 默认模型）
huggingface-cli download HuggingFaceH4/zephyr-7b-beta

# 4) lm-eval 的 wmdp MCQ 数据（首次评测会拉）
python -c "import datasets; datasets.load_dataset('cais/wmdp', 'wmdp_bio')"
```

## 六、建议执行顺序

TOFU 自训主表跑完、P0-3 与训练管线验证通过之后再做：

1. 提前下载（本清单第五节，可与 TOFU 训练并行）。
2. `bash scripts/muse_unlearn.sh` 改造为只跑 SimNPO/RMU × News（或仿 `tofu_unlearn_one.sh` 写 `muse_unlearn_one.sh` 以接入落盘链路）。
3. 仿写 `wmdp_unlearn_one.sh`：RMU + zephyr + `max_steps=80` 单组，评测跑 `wmdp_cyber` + `mmlu`。
4. 结果各自成表（MUSE 二维 HM / WMDP 两数分报），**不进** `ou_table3_runs.jsonl`（口径不同，混了会污染 TOFU 汇总表）。
