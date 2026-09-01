# MUSE / WMDP 复现准备清单

> 分支：`repro/ou-table3`　|　撰写：2026-09-01　|　状态：**下载未开始，均可无 GPU 提前做**
> 相关文档：[TOFU 自训计划](./【1】ou-table3-selftrain.md)　|　[P0 评估口径](./ou-table3-p0.md)　|　[上机环境](./reproduce-prep.md)

## 〇、顶级约束（先读）

> 本节的单一份权威版本在 `.codebuddy/CONSTRAINTS.md`（项目级约束文件），此处为面向本文读者的摘要；若两处不一致以 CONSTRAINTS.md 为准。

1. **系统盘只有 30GB**：仓库代码在系统盘（剩约 16GB）。模型权重、数据集、评测产物**一律只落数据盘** `/root/autodl-tmp`（`HF_HOME`、`saves`、`paths.output_dir` 已全部指向数据盘，保持这个约定）。
2. **删除大文件后必须清回收站**：本容器的删除是先进 `/root/autodl-tmp/.Trash-0/`（数据盘根）再滞留，`du` 查空间时**必须包含隐藏目录**（`du -sh /root/autodl-tmp/*` 看不到 `.Trash-0`，要用 `du -sh /root/autodl-tmp` 全量或显式点出）。2026-09-01 实测：删 64G 大件后 `df` 纹丝不动，63G 全躺在 `.Trash-0/files/`。
3. **执行环境的 safe-delete 机制**：单次删除命令递归条目数 >500 会被拦截（`SAFE_DELETE_BULK_CONFIRM_REQUIRED`），且按 **turn 累计**计数（本轮删满 500 后，本轮内不再放行任何删除）。删除大量小文件要分批（`ls | xargs -n 200 rm -rf`）；删大目录（文件数少）不受影响。
4. **数据盘当前状态（2026-09-01 实测）**：已删除旧实验遗留 64G（Llama-2-7b-chat 38G、tofu_ft_llama2 13G、paper_models 13G，mogpu saves ~400K 归档至 `/root/autodl-tmp/_archive_mogpu/`），**但 63G 仍滞留在 `.Trash-0/` 待清理**（见文末「待办」）。清理后可用约 **273G**。

## 〇½、存储预算（全计划合计 ~110G，清理回收站后余量 ~160G）

| 项目 | 预算 | 依据 |
|---|---|---|
| A. TOFU 自训 10 组 ckpt | ~25G | 实测单个 1B ckpt 2.4G（`save_strategy='no'` 只存最终模型） |
| B. MUSE/WMDP 下载 | ~45G | 见下文下载清单 |
| C. MUSE/WMDP 训练 3 组 ckpt | ~40G | 7B ckpt 每个 13-15G |
| **合计** | **~110G** | 清理回收站后可用 ~273G → **余量 ~160G** |

弹性项：TPO beta 补搜 5 组 +12G；MUSE Books 也跑则 +13G；均在余量内。

## 一、结论速览

仓库对两个基准的支持是现成的（训练/评测配置都在），但有 **5 项前置要准备，其中 4 项不依赖 GPU，现在就能做**。范围建议不做全量：

- **WMDP**：只跑 RMU 1 组（对应 OU 论文 Table 4 的做法，正好支撑「跨 TOFU↔WMDP 桥」的论据）。
- **MUSE**：只挑 SimNPO + RMU 各 1 组（News 优先）。

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

## 五、下载执行结果（~45G，2026-09-01 20:40 全部就绪 ✅）

**执行方式**：双后台脚本并行（WMDP corpus 走 S3 直连极慢 ~10KB/s，不能阻塞 hf-mirror 快通道）：

| 脚本 | 覆盖项 | 状态文件 |
|---|---|---|
| `scripts/download_muse_wmdp.sh` | 项 1（S3 慢通道） | `/root/autodl-tmp/logs/download_muse_wmdp.status` |
| `scripts/download_muse_wmdp_hf.sh` | 项 2-7（hf-mirror 快通道，实测 ~40MB/s） | `/root/autodl-tmp/logs/download_muse_wmdp_hf.status` |

| # | 内容 | 大小 | 状态 |
|---|------|------|------|
| 1 | WMDP corpus（bio/chem/cyber） | ~2G | ✅ **cyber 全齐 / bio 缺 forget**（见下） |
| 2 | `muse-bench/MUSE-News`（数据集） | ~0.2G | ✅ |
| 3 | `muse-bench/MUSE-Books`（数据集） | ~0.3G | ✅ |
| 4 | `muse-bench/MUSE-News_target`（7B） | 26G | ✅（hub 实测 26G，比论文口径 13G 大，含 fp32 冗余格式） |
| 5 | `muse-bench/MUSE-Books_target`（7B） | 26G | ✅ |
| 6 | `HuggingFaceH4/zephyr-7b-beta`（WMDP 默认模型） | 27G | ✅ |
| 7 | `cais/wmdp` MCQ（bio/chem/cyber：3128/408/1987 题） | <0.1G | ✅（首轮 config 名笔误 `wmdp_bio`→应为 `wmdp-bio`，已修） |

### WMDP corpora 的落地方式（重要）

S3 直连太慢（250M 爬了 1 小时才 14M），改走 HF 官方 `cais/wmdp-corpora`（parquet 子目录，879M）→ 本地转 jsonl：

```
/root/autodl-tmp/data/wmdp/               # 仓库内 data/wmdp 软链到这里
├── wmdp-corpora -> wmdp-corpora_jsonl    # hydra 读的路径（data_files 相对路径不变）
│   ├── cyber-forget-corpus.jsonl   1000 docs /  22M   ← WMDP-cyber 训练 forget 集
│   ├── cyber-retain-corpus.jsonl   4473 docs /  61M   ← WMDP-cyber retain 集
│   └── bio-retain-corpus.jsonl    60887 docs / 1.9G
├── wmdp-corpora_jsonl/                   # 上面软链的目标（真实 jsonl）
└── wmdp-corpora_hf/                      # HF 原始 parquet（保留备查）
```

- `wmdp-cyber`（`configs/experiment/unlearn/wmdp/default.yaml` 的默认 `data_split=cyber`）数据**全齐**，hydra 干跑验证通过（zephyr 已缓存、`wmdp_cyber` 任务就绪）。
- **缺口归因：`bio-forget-corpus` 缺失**。HF 官方 `cais/wmdp-corpora` repo 不含它；补源 `cais/wmdp-bio-forget-corpus` 是 **gated repo（403，需在 HF 网页申请授权）**。两条路：① S3 慢通道 wget 继续爬完 zip 后解压（后台进行中，不阻塞）；② 在 HF 网页给账号申请 gated 授权后 `huggingface-cli download cais/wmdp-bio-forget-corpus`。跑 WMDP-cyber 不受影响。
- HF 大件下载注意：hf-mirror 长连接在 7B 单分片（~4.9G）处会掐断（`IncompleteRead`），hub 缓存支持断点续传，重跑 `download_muse_wmdp_hf.sh` 即续传（自动补跑循环已验证有效：News_target 首轮 2.0G 处断，第二轮续传 1 分钟完成）。

## 六、建议执行顺序

TOFU 自训主表跑完、P0-3 与训练管线验证通过之后再做：

1. 提前下载（本清单第五节，可与 TOFU 训练并行）。
2. `bash scripts/muse_unlearn.sh` 改造为只跑 SimNPO/RMU × News（或仿 `tofu_unlearn_one.sh` 写 `muse_unlearn_one.sh` 以接入落盘链路）。
3. 仿写 `wmdp_unlearn_one.sh`：RMU + zephyr + `max_steps=80` 单组，评测跑 `wmdp_cyber` + `mmlu`。
4. 结果各自成表（MUSE 二维 HM / WMDP 两数分报），**不进** `ou_table3_runs.jsonl`（口径不同，混了会污染 TOFU 汇总表）。

## 七、待办：清空回收站（立即执行，释放 63G）

2026-09-01 删除的 64G 旧实验大件滞留在回收站（重命名为 `*.2`），`df` 仍显示 91G used。安全机制因「turn 累计删除数」拦截了工具侧的后续删除，**手动执行一条命令即可**：

```bash
rm -rf /root/autodl-tmp/.Trash-0/files/models--NousResearch--Llama-2-7b-chat-hf.2 \
       /root/autodl-tmp/.Trash-0/files/models--locuslab--tofu_ft_llama2-7b.2 \
       /root/autodl-tmp/.Trash-0/files/paper_models
df -h /root/autodl-tmp   # 期望：used 从 91G 降到 ~28G，可用 ~272G
```

删除后再确认：`du -sh /root/autodl-tmp/.Trash-0` 应只剩 <10M 的零碎文件。
