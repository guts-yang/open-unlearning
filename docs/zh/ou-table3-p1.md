# P1：官方 ckpt 免训评估（准备就绪，待挂卡执行）

> 分支：`repro/ou-table3`，HEAD `e9e9ab2`（P0 提交）。本文是 P1 的**执行手册**。
> 前置：P0-3 本地基线评测已跑完且 `ou_compare_official.py` 通过（见 [ou-table3-p0.md](./ou-table3-p0.md)）。
> 准备时间：**2026-09-01**（无 GPU 状态下完成全部代码与清单准备；实际评估待开机挂卡）。

## 一、就绪清单

| # | 准备项 | 状态 |
|---|--------|------|
| 1 | `scripts/ou_list_ckpts.py`：拉取并过滤 ckpt，生成清单 | ✅ 已跑，产出 `results/ckpt_list.json`（138 条） |
| 2 | `scripts/ou_eval_one.sh`：单条「下载→评测→四维→落盘→删权重」 | ✅ 已建，语法通过 |
| 3 | `scripts/ou_eval_batch.sh`：串行批量 + 断点续跑 + 失败不中断 | ✅ 已建，语法通过 |
| 4 | `scripts/ou_table.py`：四维汇总表 + 命中判定 + 归因段 | ✅ 已建，合成数据试跑通过 |
| 5 | `scripts/ou_aggregate.py --json`：机器可读四维输出 | ✅ 已加，回归验证数值零漂移 |
| 6 | gibberish 检测器可下载可加载（transformers 4.51.3） | ✅ 已验证：`num_labels=4`，`id2label[0]='clean'`；干净英文 P=0.96、乱码 P=0.0003 |
| 7 | P0-3 本地基线（`*_local`）+ 官方对照 | ⏳ **待 GPU**：`bash scripts/ou_eval_baselines.sh` |
| 8 | 实际评估 138 条 | ⏳ 待 GPU |

## 二、清单怎么来的

```
HF 搜索 unlearn_tofu_Llama-3.2-1B-Instruct_forget10 → 428 个仓库
  ├── 398 个 unlearn ckpt（命名：unlearn_tofu_<model>_forget10_<Method>_<超参串>）
  └── 30 个辅助模型（排除）
        neg_tofu_*_forget10_pert_* 10
        pos_tofu_*_forget10_bio_*  10
        pos_tofu_*_forget10_para_* 10
```

398 个的方法分布与配额：

| 方法 | 官方可用 | 配额 | 实际入选 | 超参网格 |
|------|---------|------|---------|---------|
| SimNPO | 48 | 48 | **48** | lr{1e-5,2e-5,5e-5} × b{3.5,4.5} × d{0,1} × g{0.125,0.25} × ep{5,10}（a1 固定） |
| RMU | 54 | 54 | **54** | lr × layer{5,10,15} × scoeff{1,10,100} × epoch{5,10} |
| GradDiff | 40 | 8 | **8** | lr{1e-5..5e-5} × alpha{1,2,5,10} × epoch{5,10} |
| NPO | 54 | 8 | **8** | lr × beta{0.05,0.1,0.5} × alpha{1,2,5} × epoch{5,10} |
| UNDIAL | 54 | 8 | **8** | lr{1e-5,1e-4,3e-4} × beta{3,10,30} × alpha{1,2,5} × epoch{5,10} |
| IdkDPO | 54 | 4 | **4** | 同 NPO |
| IdkNLL | 40 | 4 | **4** | 同 GradDiff |
| AltPO | 54 | 4 | **4** | 同 NPO |
| GradAscent | **0** | 4 | **0** | ⚠️ 官方没有该方法的 forget10 ckpt |

**合计 138 条**（用户预算 142 条，差 4 条就是 GradAscent——见 §六归因）。

抽样规则（确定性，可复现）：按 `lr` 数值排序后等间距取样；P2 要对齐的三个锚点**强制入选**
（`scripts/ou_list_ckpts.py` 的 `FORCE_INCLUDE`）：

- SimNPO `lr1e-05_b4.5_a1_d0_g0.125_ep10`（P2-T1）
- SimNPO `lr2e-05_b3.5_a1_d1_g0.25_ep10`（P2-T2）
- GradDiff `lr1e-05_alpha5_epoch10`（P2-T3，注意是 `epoch10` 不是 `ep10`）

重新生成清单：

```bash
source /root/autodl-tmp/env_hf.sh
python scripts/ou_list_ckpts.py                      # 覆盖 results/ckpt_list.json
python scripts/ou_list_ckpts.py --dry-run            # 只看统计
python scripts/ou_list_ckpts.py --quota SimNPO=48,RMU=54
```

## 三、怎么跑

```bash
source /root/autodl-tmp/env_hf.sh
source /root/autodl-tmp/envs/unlearning/bin/activate
cd /usr/local/open-unlearning

# 1) 先试跑 2 条（强烈建议，确认 gibberish 首拉 + 评测 + 聚合全链路）
bash scripts/ou_eval_batch.sh --limit 2

# 2) 按方法跑（建议顺序：先 SimNPO 看能否命中靶值）
bash scripts/ou_eval_batch.sh --only SimNPO
bash scripts/ou_eval_batch.sh --only RMU
bash scripts/ou_eval_batch.sh --only GradDiff
bash scripts/ou_eval_batch.sh            # 剩下的全跑

# 3) 重跑某条（忽略 done 记录）
bash scripts/ou_eval_batch.sh --only SimNPO --force
```

单条调试：

```bash
bash scripts/ou_eval_one.sh \
  open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_SimNPO_lr1e-05_b4.5_a1_d0_g0.125_ep10
```

### 每条流程做了什么

1. `huggingface-cli download` 到 `$SAVES/hf_ckpts/<name>`（失败重试 3 次）
2. `src/eval.py`：`retain_logs_path` **必传**（否则 privleak 静默 NaN），产物写 `$SAVES/eval/<name>`
3. `scripts/ou_aggregate.py --json`：算四维并存 `<eval_dir>/ou_aggregate.json`
4. 追加一行到 `results/ou_table3_runs.jsonl`（同一 ckpt 重跑会覆盖旧行）
5. **删除权重**，只留 `TOFU_SUMMARY.json` + `TOFU_EVAL.json`
   （两个都要留：`ou_aggregate.py` 需要 `TOFU_EVAL.json` 里的
   `forget_truth_ratio.value_by_index` 现算 TR；只拷 summary 会直接报错）

## 四、磁盘与耗时预算

| 项 | 估算 |
|----|------|
| 单次 eval | 约 3 分钟（含 gibberish 生成 `max_new_tokens=200` 与 4 个 MIA） |
| 单条峰值占用 | 权重约 2.5GB + 评测产物约 2MB，评估完权重即删 |
| 138 条总耗时 | 约 7 小时（不含下载；HF 下载是主要不确定性） |
| 数据盘 | 剩 213GB，**流式的下-评-删足够；全量落盘约 350GB 会爆** |
| 系统盘 | 仓库在 `/` 上（剩 16GB），所以所有 `paths.output_dir` 都指向 `$SAVES`，不要落到仓库内 |

## 五、验收与回查

```bash
python scripts/ou_table.py                 # → results/ou_table3.md
python scripts/ou_table.py --top 20 --tol 0.02
```

**P1 通过条件**：SimNPO 的 48 个 ckpt 中，至少有一个同时命中
`Agg 0.53 / Mem 0.32 / Priv 0.63 / Utility 1.00`（各 ±0.02）。命中才准进 P2。

不命中时的回查顺序（不要跳步）：

1. **P0-3 基线**：`python scripts/ou_compare_official.py` 是否与官方日志对齐；本地 Fluency 分母是否补齐。
2. **Mem 的 TR 定义与归一化**：必须是 OU 的 `prob_mean` 版（现算），不能用 summary 里 `closer_to_1_better` 的 `forget_truth_ratio`。见 `ou-table3-p0.md` §二.2。
3. **sMIA 映射**：默认 `symmetric`（`1/(1+9.05·|dev|)`，由 Init→0.10 反解）。
   过遗忘侧与论文不符时试 `python scripts/ou_aggregate.py ... --s-mia-mode piecewise`。
4. **Utility 的 Fluency 来源**：gibberish `class_id=0`（已验证 = `clean`）。

额外产出：SimNPO 48 点的全量附表在 `results/ou_table3.md` 末尾，含 `HM(Mem, Utility)` 列，
用来反推 OU 真实的 model-selection 规律（论文说按 HM(Mem,Utility) 选，但 SimNPO 的
Mem 0.32 ≈ Retain 的 0.31，暗示实际是「遗忘到 retain 水平即停」）。

## 六、已知缺口与归因（不静默跳过）

1. **GradAscent 缺 4 条**：官方 398 个 ckpt 里没有 GradAscent
   （只有 AltPO/NPO/RMU/UNDIAL/IdkDPO 各 54、SimNPO 48、GradDiff/IdkNLL 各 40）。
   P1 该格改由 **P2 自训补齐**，论文 Table 3 也没有 GradAscent 行。
2. **Retain Mem 0.3462 vs 论文 0.31（+0.036）**：见 `ou-table3-p0.md` §三，属论文未公开生成代码导致的口径差异。
3. **GradDiff Priv**：论文 3.27e-03，对称映射下过遗忘侧 `s(dev≈+0.6)≈0.15`；P1 拿到本地数据后用 `--s-mia-mode piecewise` 复核。
4. **P1 清单 ≠ 训练**：本阶段全部是免训评估，不写 `results/tofu_*.md` 的 SOTA 表；SOTA 回填等 P2 之后按回填规则做。
