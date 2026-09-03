# 自训计划：TOFU forget10 七方法四维复现

> 对应内部计划名：`tofu-selftrain-7methods`（执行明细同步于 `.codebuddy/plans/`）
> 分支：`repro/ou-table3`　|　基座：Llama-3.2-1B-Instruct + TOFU forget10（retain90 / holdout10）
> 计划生成：2026-09-01　|　状态：**准备已就绪，等待挂卡执行**（GPU 未挂载）
> 四维口径：OpenUnlearning 论文 §F.1 —— Mem=HM(1−ES,1−EM,1−ParaProb,1−TR)；Priv=HM(s_LOSS,s_ZLib,s_MinK,s_MinK++)；Utility=HM(MU,Fluency)；Agg=HM(Mem,Priv,Utility)；全部按 init-finetuned 归一化。
> 相关文档：[P0 评估口径](./ou-table3-p0.md)　|　[P1 官方 ckpt 免训评估手册](./ou-table3-p1.md)

## 一、任务目标

对 7 个遗忘方法按**官方调参值自训**并评测出 OU Table 3 四维数值（Mem / Priv / Utility / Agg），产出可直接写进论文、可与同名官方 ckpt 对拍的结果；同时完成对当前已有实验结果的检查与整改。

## 二、已有实验结果的检查结论

`results/tofu_Llama-3.2-1B-Instruct.md` 现存 4 行（SimNPO / RMU / UNDIAL / AltPO），四个层面均不达标：

1. **口径不对**：记录的是旧六指标（FQ / MU / 1−RL / TR / 1-Prob / 1-ES），不是四维。
2. **指标不全**：4 个 ckpt 的 `TOFU_SUMMARY.json` 仅有 7 个指标，缺 `exact_memorization`、4 个 MIA、`forget_Q_A_gibberish`、`PARA_Prob` → 直接喂 `ou_aggregate.py` 会 `KeyError`。
3. **超参是默认的**：均为 8/28 未调参默认超参；且完全没有 GradDiff / NPO / GradAscent / TPO。
4. **基线缺失**：`retain90_local` / `full_local` 均未产出 → Fluency 归一化分母为空；`results/ou_table3_runs.jsonl` 一条记录都没有。

现有数值可解读的信号（同口径内）：遗忘强度（1−Prob）RMU 0.9999 > UNDIAL 0.7434 > AltPO 0.4922 > SimNPO 0.4145；`privleak` RMU +57.8（过遗忘侧）对 SimNPO −97.8（欠遗忘侧）；`FQ` 全部塌到 1e-16~1e-6，无法区分方法。

结论：这张表不能直接用于论文，需按新超参重训并转四维口径。

## 三、实验范围（已确认）

- **只自训**：不跑官方 ckpt 的免训评估批次（TPO / GradAscent 在 HF 上本就没有官方 ckpt）。
- **超参用官方 ckpt 文件名里的调参值**，以便与同名官方 ckpt 对拍验证管线等价。
- **8/28 的 4 个旧 ckpt 直接用新超参重训**，不单独重评。
- **TPO 先跑 beta=0.19**（论文 3B 的 forget10 值，1B 无官方值），看四维落点后再决定是否补搜（搜索档 0.19/0.23/0.27/0.30/0.32 已在 run.sh 内置）。
- **MUSE / WMDP 不在本轮范围**。建议后续补：WMDP 只跑 RMU 单组（支撑「跨 TOFU↔WMDP 桥」论据），MUSE 挑 SimNPO+RMU 各 1 组。准备工作：`setup_data.py --wmdp`（wget S3+unzip）、muse-bench 数据集下载；lm-eval 0.4.11 已装、Llama-2-7b 已缓存、MUSE retrain 对照日志已有。

### 方法优先级与必跑理由

| 优先级 | 方法 | 必跑理由 |
| --- | --- | --- |
| P0 | SimNPO | 靶子，方向 G 的 Δ_FO 打底 |
| P0 | RMU | 第二名 + 本机验证锚 + 跨 TOFU↔WMDP 桥 |
| P0 | GradDiff | 过度遗忘的极端锚点（三维垫底、二维第一），论文核心论据 |
| P1 | NPO | SimNPO 父方法，ablation 标尺 |
| P1 | UNDIAL | 自蒸馏族代表 |
| P2 | GradAscent | 下界，跑 1 次不调参 |
| 新增 | TPO | 仓库新加入的方法 |

## 四、自训矩阵（10 组，先跑最小 7 组）

命名规则：`unlearn_tofu_Llama-3.2-1B-Instruct_forget10_<Method>_<超参串>`。1B 模型 `num_hidden_layers=16`。全部超参点已核验存在同名官方 ckpt（GradAscent/TPO 除外）。

| 优先级 | 方法 | run_tag | 对应官方 ckpt 超参串 | hydra 覆盖（关键项） |
| --- | --- | --- | --- | --- |
| P0 | SimNPO | S1 | `lr1e-05_b4.5_a1_d0_g0.125_ep10` ✔ | `lr=1e-5 beta=4.5 alpha=1.0 delta=0.0 gamma=0.125 ep=10` |
| P0 | SimNPO | S2 | `lr2e-05_b3.5_a1_d1_g0.25_ep10` ✔ | `lr=2e-5 beta=3.5 delta=1.0 gamma=0.25`（估方差） |
| P0 | RMU | R1 | `lr1e-05_layer10_scoeff100_epoch10` ✔ | `module_regex='model\.layers\.10' steering_coeff=100` |
| P0 | RMU | R2 | `lr1e-05_layer5_scoeff100_epoch10` ✔ | `module_regex='model\.layers\.5'`（layer 敏感性） |
| P0 | GradDiff | G1 | `lr1e-05_alpha5_epoch10` ✔ | `alpha=5`（**过度遗忘锚点**） |
| P0 | GradDiff | G2 | `lr1e-05_alpha1_epoch10` ✔ | `alpha=1`（弱遗忘对照） |
| P1 | NPO | N1 | `lr1e-05_beta0.1_alpha1_epoch10` ✔ | `beta=0.1 alpha=1.0`（= 仓库默认） |
| P1 | UNDIAL | U1 | `lr0.0001_beta10_alpha1_epoch10` ✔ | `lr=1e-4 beta=10 alpha=1.0`（= 仓库默认） |
| P2 | GradAscent | GA1 | 官方无 ckpt | 仓库默认 `lr1e-5 / ep10`，不调参 |
| 新增 | TPO | TPO1 | 官方无 ckpt | `beta=0.19 alpha=0.0 classifier=gpt lr1e-5 ep10` |

- 最小集（7 次训练）：S1 / R1 / G1 / N1 / U1 / GA1 / TPO1
- 补充集（3 次）：S2（方差）、G2（遗忘强度对比）、R2（RMU layer 敏感性）
- 有效 batch 统一 32：`tofu_unlearn_one.sh` 默认 4×4×2卡；TPO 单卡 16×2

### 成本估算

- 训练：1B + forget10（400 条）+ retain90（3600 条），10 epoch，2 卡 ≈ 20-40 分钟/组
- 评测：约 3 分钟/组（gibberish 生成 + 4 MIA）
- 最小 7 组 ≈ 3-5 小时；完整 10 组 ≈ 4-7 小时（串行）

## 五、执行步骤（挂卡后三条命令）

```bash
# 1) P0-3 基线（约 6 分钟）：评测 retain90 与 full，自动过官方对照，不一致 exit 1
bash scripts/ou_eval_baselines.sh

# 2) 自训最小 7 组（约 3-5 小时）；稳定后 --full 补到 10 组
bash scripts/run_selftrain_matrix.sh

# 3) 出汇总表
python scripts/ou_table.py
```

每条命令前必须：`source /root/autodl-tmp/env_hf.sh` + `source /root/autodl-tmp/envs/unlearning/bin/activate`（脚本内部已做）。

### 执行要点（防回归）

- **P0-3 是硬前置**：基线未产出前 Fluency 分母为空，自训四维全不可用。
- **RMU 两个易踩映射**：`layer10` → `trainer.method_args.module_regex='model\.layers\.10'`（yaml 无 layer 参数，默认锁死 layers.7）；`scoeff100` → `steering_coeff=100`（yaml 默认 2）。1B 仅 16 层，layer15 风险高。
- **UNDIAL 默认 lr=1e-4 不是 1e-5**。
- **不覆盖官方日志目录**：`overwrite: false` 的语义是已存在指标直接跳过（`src/evals/base.py:85`），写进官方目录会得到新旧混合产物。本地一律写 `*_local`。
- **评测必须显式传 `retain_logs_path`** 且指向 `*_local`（否则 privleak 静默 NaN）。
- `ou_aggregate.py` 需要 summary 同目录的 `TOFU_EVAL.json`（TR 现算），两个文件都要留。
- 产物一律落 `$SAVES=/root/autodl-tmp/saves`（系统盘只剩 16GB）。

## 六、交付物

1. 汇总表：方法 × run_tag × 四维 × 调参超参串 × 是否命中靶值（±0.02）。
2. 自训 vs 同名官方 ckpt 的**对拍表**（验证训练管线等价；S1/R1/G1/N1/U1 有同名官方 ckpt）。
3. P0-3 归一化分母最终清单（含 Fluency）。
4. 评估器 commit hash 与 `configs/eval/tofu.yaml` 最终 diff。
5. 未命中项归因：Retain Mem +0.036、GradDiff Priv（3.27e-03 vs s(dev)≈0.15）、GradAscent/TPO 无官方 ckpt。

## 七、当前进度

| # | 任务 | 状态 |
|---|------|------|
| 1 | tpo-ready：TPO 数据入库、`ou_append_run.py`、`run.sh` 接四维 + ZeRO-3 | ✅ |
| 2 | p0-baseline：`ou_eval_baselines.sh` + 官方对照 ±0.02 | ✅ PASS（Init/Retain 本地） |
| 3–5 | 自训最小 7 组（V100） | ✅ 训完（TPO 评测中）；**不作为 Table 3 对标** |
| 6 | 阶段 A：官方 ckpt 免训评测（`scripts/ou_p1_table3.sh`） | 🔄 SimNPO 网格进行中 |
| 7 | `ou_table.py --source official` 出表；命中后再本机重训对拍 | ⏳ |

### 本机相对原文必须不一致（阶段 A 评官方权重不受 optimizer 影响）

| 项 | 原文 / 官方 | 本机 |
|---|---|---|
| GPU | 2× L40s 48GB | 2× V100-PCIE 32GB |
| Attention | `flash_attention_2` | **必须 `sdpa`**（`ou_eval_one.sh` 已写死） |
| Optimizer | `paged_adamw_32bit` | 自训用 `adamw_torch`（仅重训对拍） |
| PyTorch | 文档 `2.4.1` | `2.8.0+cu128` |
| Retain Mem | 论文 0.31 | 实现 0.346（官方 eval 日志也推不出 0.31） |
| Fluency 分母 | 官方未评 | `full_local` 0.85796 |
| Mem 负分量 | 论文未写 | `1−TR` 等略负时 **clamp 0 且不参与 HM**（`ou_aggregate.py`，jsonl `notes`） |

阶段 B（本机重训）只在 SimNPO 官方网格命中 `Agg 0.53 / Mem 0.32 / Priv 0.63 / Utility 1.00`（±0.02）之后，用该超参串跑一次 `tofu_unlearn_one.sh`，不盲跑 S1/R1。

P1 结束后本机串行复现（`scripts/ou_repro_p1_winners.sh`，等 GPU 空再开，不抢 P1）：
1. SimNPO 官方网格最好 `lr5e-05_b3.5_a1_d1_g0.125_ep10`（Agg 0.4716，tag `S_p1best`）
2. AltPO `lr2e-05_beta0.05_alpha1_epoch10`（官方 Agg 0.5826；缺 `alt5_seed_0.json` 时先 `generate.py`）

补算备注（clamp 行 + 自训 G1/GA1）写在 `results/ou_table3.md` 第二节，不在此重复。

### 执行前已修的坑

1. `ou_eval_baselines.sh` 首次评测 retain90 时 `retain_logs_path` 指向自己尚未产出的文件 → 已改为首次回退官方日志。
2. `tofu_unlearn_one.sh` 训完不写 `results/ou_table3_runs.jsonl`（汇总表唯一数据源）→ 已新增 `scripts/run_selftrain_matrix.sh`，由矩阵脚本显式传 `--method/--hyper` 收口（自训 task_name 不含超参串，必须显式传才能与官方 ckpt 对拍）。
3. TPO `run.sh` 的 `~data...name` 未加引号会被 bash 波浪号展开 → 已加引号。
4. TPO 标注数据被 `.gitignore` 的 `data/` 规则忽略（`!*/data/` 只匹配两级路径）→ 已加 `!community/methods/TPO/data/` + `!community/methods/TPO/data/**`。

## 八、关键脚本清单

| 脚本 | 作用 | 状态 |
|------|------|------|
| `scripts/ou_eval_baselines.sh` | P0-3a/3b 本地基线评测（写 `*_local`，不覆盖官方目录） | ✅ 已验证（GPU 前逻辑） |
| `scripts/ou_compare_official.py` | 本地基线 vs 官方逐项对照，不一致 exit 1 | ✅ 已验证 |
| `scripts/run_selftrain_matrix.sh` | 自训矩阵入口（最小 7 组 / `--full` 10 组 / `--only` 单组 / 断点续跑） | ✅ 语法+守门已验证 |
| `scripts/tofu_unlearn_one.sh` | 单组自训：训练→评测→`ou_aggregate --json`→更新 `results/tofu_*.md`；参数化 GPU_IDS/NUM_PROCESSES/PER_DEVICE_BS/GRAD_ACCUM，支持 run_tag + hydra 覆盖 | ✅ |
| `scripts/ou_append_run.py` | 四维结果落 jsonl（官方/自训两种 name 解析、重跑覆盖、原子写入） | ✅ 已单测 |
| `community/methods/TPO/run.sh` | TPO 单组（训练→评测→四维→落 jsonl） | ✅ Hydra 干跑通过 |
| `scripts/ou_aggregate.py` | 四维聚合（`--json` / `--s-mia-mode{symmetric,piecewise}`；负分量 clamp+备注） | ✅ |
| `scripts/ou_table.py` | 汇总表 + 命中判定（±0.02）+ 归因段 | ✅ 合成数据验证 |
| `scripts/ou_eval_one.sh` / `ou_eval_batch.sh` | 官方 ckpt 下-评-删批量（对拍补充用） | ✅ |
| `scripts/ou_backfill_aggregate.sh` | 已有 SUMMARY 的目录补算四维并落 jsonl | ✅ |
| `results/ckpt_list.json` | 138 条官方 ckpt 清单（SimNPO48/RMU54/GradDiff8/NPO8/UNDIAL8/AltPO4/IdkDPO4/IdkNLL4；GradAscent 官方 0 个） | ✅ |
