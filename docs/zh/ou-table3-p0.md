# P0：OU Table 3 四维口径改造与聚合脚本（训练前准备）

> 分支：`repro/ou-table3`　改造前 HEAD：`6f331a6`（base = 上游 main `4ad738a`）
> 盘点时间：**2026-08-31**。本文记录 P0 交付内容、口径公式、校准证据、评测就绪命令与回查路线。

## 一、P0 交付物

| # | 交付物 | 状态 |
|---|--------|------|
| 1 | `configs/eval/tofu.yaml`：启用 7 个缺失指标（6 个取消注释 + 新增 `forget_Q_A_PARA_Prob`） | ✅ 已改（diff 见 §六） |
| 2 | `scripts/ou_aggregate.py`：三级 HM 四维聚合（Mem/Priv/Utility/Agg）+ 归一化分母打印 | ✅ 已建（`hmean` 与 `src/evals/metrics/utility.py` 的 `hm_aggregate` 同语义） |
| 3 | 官方数据无 GPU 校准：Init/Retain 锚点命中情况 | ✅ 见 §三 |
| 4 | retain90 / init-finetuned(full) 评测就绪命令 + 官方对照检查点 | ✅ 见 §五 |
| 5 | 评估器 HEAD 与 config 最终 diff | ✅ 见 §六 |

P1/P2 不在本次范围（未训练、未免训评估）。P0-3 的 retain90/full 评测需在 **GPU（AutoDL）** 机器执行，本机仅完成代码与校准。

## 二、四维口径（OU 论文 §F.1，已核对原文）

```
Mem     = HM(1 - norm(ES), 1 - norm(EM), 1 - norm(Para.Prob), 1 - norm(TR))
Priv    = HM(s_MIA(loss), s_MIA(zlib), s_MIA(min_k), s_MIA(min_k++))
Utility = HM(norm(MU), norm(Forget Fluency))
Agg     = HM(Mem, Priv, Utility)        # 论文 Table 3（三维，含 Priv）
```

要点（均经源码/论文核对）：

1. **归一化**：`norm(x) = x_target / x_init_finetuned`，全部子指标按 init-finetuned（full）模型归一化。论文原文："we scale all metrics with init finetuned model … all scores are effectively divided by the target model's MU"。
2. **TR 方向**：`TOFU_SUMMARY.json` 里的 `forget_truth_ratio` 是 TOFU 原版 `closer_to_1_better` 聚合（`wrong/correct` 取 min，语义为**不确定度**，高=遗忘好），**不能**直接用于 Mem 的 `1−TR`（会方向相反甚至出现 `TR_norm>1 → 1−TR_norm<0`）。
   Mem 应使用 **OU 论文定义**的 truth ratio（`src/evals/metrics/memorization.py` 注释："as used in OpenUnlearning paper's meta-evaluation"）：`prob_mean` 聚合 `correct/(correct+wrong)`，语义为**记忆度**（高=记忆强），`1−TR` 才符合"高=遗忘好"。
   实现：从 `TOFU_EVAL.json` 的 `forget_truth_ratio.value_by_index[*].score`（= `wrong/correct` 原始值）现场计算 `TR = mean(1/(1+score))`。因此 **target / init 必须提供 TOFU_EVAL.json**（本地评测都会产出）。
3. **sMIA**（参照 `src/evals/metrics/privacy.py` 的 privleak 相对偏差逻辑）：
   ```
   dev = (1 - auc_target - (1 - auc_retain)) / (1 - auc_retain) = (auc_retain - auc_target) / (1 - auc_retain)
   s_MIA = 1 / (1 + theta * |dev|),   theta = 9.05
   ```
   `dev<0` 为泄露侧（target AUC 高于 retain）、`dev>0` 为过遗忘侧；`target==retain` 时 `dev=0 → s=1`（Retain→Priv=1.00 恒等）。
4. **Forget Fluency** = `forget_Q_A_gibberish` 的 agg_value = Gibberish 检测器（4 类，`class 0 = clean`）P(clean)，越高越好。
5. **Agg 表区分**：论文 **Table 3** Agg=HM(Mem, Priv, Utility)（本仓库复现口径，用户靶值来源）；**Table 6** 的 Agg=HM(Mem, Utility)（Priv 仅展示，caption 明写 "Privacy scores are not used in the aggregation"）。两表 Mem/Priv/Utility 列相同（如 SimNPO 0.32/0.63/1.00），仅 Agg 算法不同。已用 Table 6 全部 10 行验证二维 HM 自洽、用 SimNPO/RMU/Retain/GradDiff 验证三维 HM 自洽。

## 三、无 GPU 校准结果（官方 `open-unlearning/eval` 数据）

校准用官方 1B 的 full / retain90 `TOFU_SUMMARY.json` + `TOFU_EVAL.json`（经 `hf-mirror.com` 下载；官方 summary 无 `forget_Q_A_PARA_Prob`/`forget_Q_A_gibberish`，前者已从 EVAL fallback 取到，后者官方从未评测 → 校准期 Utility 用 `--assume-fluency 1.0`）。

| 行 | 来源 | Mem | Priv | Utility | Agg |
|----|------|-----|------|---------|-----|
| Init finetuned | 论文 Table 3 | 0.00 | 0.10 | 1.00 | 0.00 |
| Init finetuned | `ou_aggregate.py` | **0.0000** ✅ | **0.0999** ✅ | **1.0000** ✅ | **0.0000** ✅ |
| Retain | 论文 Table 3 | 0.31 | 1.00 | 0.99 | 0.58 |
| Retain | `ou_aggregate.py` | 0.3462 ⚠️(+0.036) | **1.0000** ✅ | **0.9933** ✅ | 0.6128 ⚠️(+0.033) |

结论：

- **Init finetuned 四维全命中**（0.00/0.10/1.00/0.00）。sMIA 参数 theta=9.05 由 Init 四攻击 dev≈−0.994~−0.997 反解 `1/(1+theta·|dev|)=0.10` 得到，回代得 Priv=0.0999。
- **Retain 命中 Priv/Utility**（恒等与 MU_norm=0.987 均成立）。
- **Retain Mem=0.346 与论文 0.31 偏差 +0.036**：在官方 ES/EM/ParaProb 数值（norm 后 0.9165/0.3995/0.4681）与 HM 结构下，数学上无法同时让四个非负分量给出 0.31（需要 `TR_norm>1`），故 TR=prob_mean 口径是现有数据下最接近的候选。偏差大概率来自论文 Retain 行的评测数据与官方 `open-unlearning/eval` 不同（论文未公开其表 3 生成代码）。**留待 P0-3 本地评测后复核**（本地评测数值应与官方一致，若仍对不上即属论文口径差异，不影响相对排名口径）。

**归一化分母（init-finetuned = full 模型）清单**（P0-2 要求打印，供人工核对）：

| 指标 | 值 | 来源 |
|------|-----|------|
| ES | 0.70627 | 官方 summary |
| EM | 0.97394 | 官方 summary |
| ParaProb | 0.10042 | 官方 EVAL（summary 无此 key） |
| TR_OU | 0.68498 | 官方 EVAL `value_by_index` 现算（prob_mean） |
| MU | 0.59915 | 官方 summary |
| Fluency | **None（官方未评测）** | 需 P0-3 本地评测 full 模型补齐 |

**sMIA 参考（retain90 各 MIA AUC）**：mia_loss 0.38736 / mia_zlib 0.30889 / mia_min_k 0.38261 / mia_min_k_plus_plus 0.47763。

## 四、锚点锁定状态与风险

| 锚点 | 论文值 | 本实现 | 状态 |
|------|--------|--------|------|
| Init Mem | 0.00 | 0.0000（norm 全 1 → 1−norm 全 0） | ✅ 锁定（结构恒等） |
| Init Priv | 0.10 | 0.0999 | ✅ 锁定（theta=9.05 反解） |
| Init Utility | 1.00 | 1.0000 | ✅ 锁定（恒等） |
| Retain Priv | 1.00 | 1.0000 | ✅ 锁定（恒等） |
| Retain Utility | 0.99 | 0.9933 | ✅ 锁定（需本地 Fluency 复核） |
| Retain Mem | 0.31 | 0.3462 | ⚠️ 偏差 +0.036，记录待复核 |
| GradDiff Priv | 3.27e-03 | 未验证 | ⚠️ 对称映射下过遗忘侧 `s(dev=+0.6)≈0.15`，与论文极低值不符；论文预示过遗忘惩罚更重。P1 拿到本地 GradDiff 评测后如需对齐，可将 s_MIA 改为分段映射（dev>0 用 `exp(−theta·dev)`，theta≈9），已在脚本留 `--s-mia-theta` 参数便于调整 |

**P1 命中失败回查路线**（用户既定策略）：① 先查 P0-3 本地 retain90/full 评测与官方对照（§五检查点）→ ② 再查 Mem 的 TR 定义与归一化（§二.2）→ ③ 再查 sMIA 映射（§二.3、§四）→ ④ 最后查 Utility 的 Fluency 来源（gibberish class_id）。

## 五、评测就绪命令（GPU / AutoDL 执行）

前置：AutoDL 开机挂卡；`source /root/autodl-tmp/env_hf.sh`；`source /root/autodl-tmp/envs/unlearning/bin/activate`；仓库**唯一副本**在 `/usr/local/open-unlearning`（数据盘重复副本已于 2026-09-01 删除，详见 `docs/zh/reproduce-prep.md`）。

> 2026-09-01 更新：下面的命令已脚本化，直接跑 `bash scripts/ou_eval_baselines.sh` 即可
> （默认写到 `*_local` 目录并自动做官方对照），细节见本文 §八。

**0) 拉取官方对照 log**（建议先做，作 P0-3 检查点基准）：

```bash
python setup_data.py --eval_logs    # → saves/eval/tofu_Llama-3.2-1B-Instruct_retain90|full/...
```

**1) P0-3a：retain90 评测（生成含新指标的 retain log，PrivLeak 参考）**：

```bash
CUDA_VISIBLE_DEVICES=0 python src/eval.py experiment=eval/tofu/default.yaml \
  forget_split=forget10 holdout_split=holdout10 \
  model=Llama-3.2-1B-Instruct \
  task_name=tofu_1B_retain90_p0 \
  model.model_args.pretrained_model_name_or_path=open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90 \
  model.tokenizer_args.pretrained_model_name_or_path=open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90 \
  paths.output_dir=saves/eval/tofu_Llama-3.2-1B-Instruct_retain90 \
  retain_logs_path=saves/eval/tofu_Llama-3.2-1B-Instruct_retain90/TOFU_EVAL.json
```

**2) P0-3b：init-finetuned（full）评测（补归一化分母 Fluency + 本地对照）**：

```bash
CUDA_VISIBLE_DEVICES=0 python src/eval.py experiment=eval/tofu/default.yaml \
  forget_split=forget10 holdout_split=holdout10 \
  model=Llama-3.2-1B-Instruct \
  task_name=tofu_1B_full_p0 \
  model.model_args.pretrained_model_name_or_path=open-unlearning/tofu_Llama-3.2-1B-Instruct_full \
  model.tokenizer_args.pretrained_model_name_or_path=open-unlearning/tofu_Llama-3.2-1B-Instruct_full \
  paths.output_dir=saves/eval/tofu_Llama-3.2-1B-Instruct_full/evals_forget10 \
  retain_logs_path=saves/eval/tofu_Llama-3.2-1B-Instruct_retain90/TOFU_EVAL.json
```

**3) 官方对照检查点（**对不上先停，不往下走**）**：

| 指标 | retain90 本地应≈ | full 本地应≈ |
|------|------------------|--------------|
| ES | 0.05896 | 0.70627 |
| EM | 0.58489 | 0.97394 |
| MU | 0.59112 | 0.59915 |
| mia_loss / zlib / min_k / min_k++ | 0.3874 / 0.3089 / 0.3826 / 0.4776 | ≈0.9965 / 0.9979 / 0.9967 / 0.9980 |
| forget_Q_A_PARA_Prob | 0.05341 | 0.10042 |
| forget_truth_ratio（closer 版） | 0.62746 | 0.47556 |

另需确认本地 `TOFU_SUMMARY.json` 含新 key：`exact_memorization`、`mia_loss`/`mia_zlib`/`mia_min_k`/`mia_min_k_plus_plus`、`forget_Q_A_PARA_Prob`、`forget_Q_A_gibberish`（后两个 official 未评，只有本地会有）。

**4) 四维聚合**（Fluency 补齐后无需 `--assume-fluency`）：

```bash
python scripts/ou_aggregate.py saves/eval/tofu_Llama-3.2-1B-Instruct_retain90/TOFU_SUMMARY.json \
  --init-summary  saves/eval/tofu_Llama-3.2-1B-Instruct_full/evals_forget10/TOFU_SUMMARY.json \
  --retain-summary saves/eval/tofu_Llama-3.2-1B-Instruct_retain90/TOFU_SUMMARY.json
# 官方数据校准期（无 Fluency）：
python scripts/ou_aggregate.py <target> --init-summary <full> --retain-summary <retain90> --assume-fluency 1.0
```

脚本会自动读取各 summary 同目录的 `TOFU_EVAL.json`（TR 现算需要）；期望 retain90 输出 Mem≈0.35 / Priv=1.00 / Utility≈0.99 / Agg≈0.61（对照 §三，论文为 0.31/1.00/0.99/0.58）。

**5) P1 时每个 unlearned 模型**：评测后跑同一条 `ou_aggregate.py`（`--init-summary`=full、`--retain-summary`=retain90 不变），即得四维；模型选择按论文 F.2 用 `HM(Mem, Utility)` 排序。

## 六、评估器 HEAD 与 config 最终 diff

- 改造前 HEAD（复验）：`6f331a6`（`repro/ou-table3`，工作区原 clean，无 mogpu/esmu）
- **评估器版本 = 本 P0 提交**（`git log -1 --oneline`）。P1 启动时在汇总表记录该 hash。
- `configs/eval/tofu.yaml` 最终 diff：

```diff
     - forget_Q_A_Prob
     - forget_Q_A_ROUGE
+    - forget_Q_A_PARA_Prob # added: needed by OU §F.1 Memorization (1 - Para.Prob)
     - model_utility # populated in the metrics key as metrics.model_utility
     - privleak
     - extraction_strength
-    # - exact_memorization
-    # - mia_min_k_plus_plus
-    # - mia_min_k
-    # - mia_loss
-    # - mia_zlib
+    - exact_memorization
+    - mia_min_k_plus_plus
+    - mia_min_k
+    - mia_loss
+    - mia_zlib
+    - forget_Q_A_gibberish
     # - mia_gradnorm
     # - mia_reference # set reference model path appropriately
-    # - forget_Q_A_gibberish
```

说明：`mia_gradnorm`/`mia_reference` 保持注释（四维口径不需要）；`forget_Q_A_PARA_Prob` 之前完全不在配置中（连注释都没有），现新增为顶层指标后才会进 `TOFU_SUMMARY.json`（`src/evals/base.py` 的 `summarize()` 只保留顶层指标）。首次评测会联网拉取 Gibberish 检测器（`madhurjindal/autonlp-Gibberish-Detector-492513457`）。

## 七、口径决策记录（防止 P1 误改）

1. **Mem 的 TR 必须用 OU 定义（prob_mean，EVAL 现算）**，不要用 summary 里的 `forget_truth_ratio`（closer 版方向相反）。
2. **Agg 用 Table 3 三维**（含 Priv）；若有人拿 Table 6 二维（不含 Priv）对比会不一致，口径不同，属预期。
3. **归一化分母一律 init-finetuned（full）**，sMIA 参考一律 retain90；两者都是相对量，评测文件缺失会直接报错而非静默错值。
4. s_MIA 的 theta=9.05 由 Init→0.10 反解；改 theta 需同时满足 Retain→1.00（恒等，任意 theta 都满足）与 Init→0.10（约束条件）。

## 八、P0 收尾（2026-09-01）：已完成的准备与待挂卡项

盘点时间：2026-09-01。此时容器**未挂卡**（无 `/dev/nvidia*`，`torch.cuda.device_count()=0`），
所以只做了无 GPU 的准备，未跑任何训练/评估。

### 环境固化

| 项 | 结论 |
|----|------|
| 仓库 | **唯一副本 `/usr/local/open-unlearning`**；数据盘重复副本 `/root/autodl-tmp/open-unlearning` 已删除（删前已确认两份 tracked 文件零差异、无独有未跟踪文件）。不要再克隆第二份 |
| 方向 A/G 残留 | `git ls-files \| grep -c mogpu` = 0、`esmu` = 0；`src/mogpu/`、`src/trainer/unlearn/mogpu_dsl/` 的陈旧 `__pycache__`（仅 .pyc，152KB，无 .py 源码、无 registry import）已清理 |
| venv | `/root/autodl-tmp/envs/unlearning`：`transformers==4.51.3`（与 requirements 一致）、`torch==2.8.0+cu128`（requirements 钉 2.4.1，已记录为已知偏差）、`huggingface_hub==0.36.0` |
| 常见坑 | 直接敲 `python` 会落到 conda base（transformers 4.46.3）。每条命令前要 `source /root/autodl-tmp/env_hf.sh` + `source /root/autodl-tmp/envs/unlearning/bin/activate` |
| 磁盘 | `/` 剩 16GB（仓库在这），`/root/autodl-tmp` 剩 213GB。所有产物一律写到 `$SAVES=/root/autodl-tmp/saves` |
| 卡数 | 计划挂 **2 卡** → 沿用 `configs/accelerate/default_config.yaml`（`num_processes: 2`）；若只有 1 卡用 `GPU_IDS=0 NUM_PROCESSES=1 GRAD_ACCUM=8`（保持有效 batch 32） |

### 脚本清单（本轮新增/改造）

| 脚本 | 作用 |
|------|------|
| `scripts/ou_eval_baselines.sh` | **[新增]** P0-3a/3b：本地评测 retain90 与 full，写到 `*_local` 目录 |
| `scripts/ou_compare_official.py` | **[新增]** 本地基线 vs 官方日志逐项比对，不一致 `exit 1`（守在 P0 门口） |
| `scripts/ou_list_ckpts.py` | **[新增]** 生成 P1 ckpt 清单（P1 用） |
| `scripts/ou_eval_one.sh` / `ou_eval_batch.sh` | **[新增]** 单条/批量下-评-删（P1 用） |
| `scripts/ou_table.py` | **[新增]** 四维汇总表 + 命中判定 + 归因（P1 用） |
| `scripts/ou_aggregate.py` | **[改造]** 加 `--json`、`--s-mia-mode{piecewise}`；默认口径不变，回归验证数值零漂移（Retain 仍为 0.3462/1.0000/0.9933/0.6128） |
| `scripts/record_tofu_result.py` | **[改造]** 加可选 `--ou-summary`，传入后才追加「本机四维」表，默认输出不变 |
| `scripts/tofu_unlearn_one.sh` | **[改造]** 参数化 `GPU_IDS`/`NUM_PROCESSES`/`PER_DEVICE_BS`/`GRAD_ACCUM`；支持第 4 个参数 run_tag 与后续 hydra 覆盖项；训完自动出四维（P2 用） |

### P0-3 的关键改动：写到 `*_local`，不覆盖官方目录

`configs/eval/tofu.yaml` 里 `overwrite: false`，而 `src/evals/base.py:85` 的语义是
「已存在且非空的指标直接跳过」。若把本地评测写进官方下载日志所在的目录，
会得到**新旧指标混合**的产物，无法用于对照。因此：

| 产物 | 路径 |
|------|------|
| 官方只读基准（不写） | `saves/eval/tofu_Llama-3.2-1B-Instruct_retain90/`、`..._full/evals_forget10/` |
| P0-3 本地产出 | `saves/eval/tofu_Llama-3.2-1B-Instruct_retain90_local/`、`..._full_local/evals_forget10/` |

后续 P1/P2 的 `--init-summary` / `--retain-summary` 优先取 `*_local`（脚本已内建该回退逻辑，
找不到才回退官方并打印告警）。

### gibberish 检测器（Fluency 分母）已验证

无 GPU 下用 CPU 冒烟验证（`HF_ENDPOINT=hf-mirror.com`，模型可下载）：

```
num_labels = 4 | id2label = {0: 'clean', 1: 'mild gibberish', 2: 'noise', 3: 'word salad'}
P(class0=clean) = 0.9642  | The quick brown fox jumps over the lazy dog.
P(class0=clean) = 0.0003  | asdkjh qweoui zxcvbn mnbvcx lkjhgf dfghjk
```

结论：`class_id=0` 确实是「干净」概率，方向正确，transformers 4.51.3 可正常加载。
**但 Fluency 的归一化分母（init-finetuned 模型的 gibberish 分数）仍未产生，必须跑 P0-3b 补齐**——
在那之前 `ou_aggregate.py` 只能靠 `--assume-fluency` 占位。

### 待挂卡才能做的事

1. `bash scripts/ou_eval_baselines.sh`（P0-3a retain90 + P0-3b full），随后自动跑 `ou_compare_official.py`。
2. 对照本文 §五检查点表，逐项确认本地数值；对不上按 §四回查路线排查，**不许进 P1**。
3. 补齐后把最终分母（含 Fluency）回填到本文 §三的分母清单表。
