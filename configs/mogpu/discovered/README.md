# MOGP-U 开放搜索前三名（可复现 CandidateSpec）

本目录存放 2026-08-28 开放式 GPU 搜索（8×8×4、F2=50 step、skip F3、forget10、`Llama-3.2-1B-Instruct` TOFU full）按 **FQ 降序** 的前三条公式。JSON 可直接作为 `trainer.method_args.candidate_spec_path`。

**不是** 10 epoch 验证结果，也 **没有** 通过 FQ≥0.05 门限。完整搜索产物在 `/root/autodl-tmp/saves/mogpu_open_search/`（不入库）。

实现：[`src/trainer/unlearn/mogpu_dsl/observables.py`](../../../src/trainer/unlearn/mogpu_dsl/observables.py)、[`atoms.py`](../../../src/trainer/unlearn/mogpu_dsl/atoms.py)、[`gates.py`](../../../src/trainer/unlearn/mogpu_dsl/gates.py)。基因不含 lr / steps / LoRA。

## 公共观测

仅在 `labels ≠ -100` 的目标 token 上对 \(\log p\) 取平均，再减冻结初始模型：

\[
\Delta_f=\bar\ell_\theta(x_f,y_f)-\bar\ell_{\mathrm{ref}}(x_f,y_f),\quad
\Delta_r=\bar\ell_\theta(x_r,y_r)-\bar\ell_{\mathrm{ref}}(x_r,y_r)
\]

原子（\(T=1\)，Huber 为 PyTorch `huber_loss`：\(|z|<\tau\) 时 \(\tfrac12 z^2\)，否则 \(\tau(|z|-\tfrac12\tau)\)；\(\varepsilon=10^{-6}\) 仅第三名用到）：

\[
\mathrm{ER}=\log(1+e^{(\Delta_f+\kappa)/T}),\quad
\mathrm{RD}=\mathrm{Huber}(\Delta_r;\tau),\quad
\mathrm{SM}=\log\bigl(1+e^{(\sqrt{\Delta_r^2+\varepsilon^2}+\Delta_f+\kappa)/T}\bigr)
\]

损失为 batch 上 \(\mathrm{mean}(\sum_i w_i\,\mathrm{atom}_i)\)，加载时权重会再归一化。

## 三条公式

| 排名 | 文件 | ast_hash（前 12） | F2 FQ / MU / TR |
|------|------|-------------------|-----------------|
| 1 | [`rank01_3d2389e5fe78.json`](rank01_3d2389e5fe78.json) | `3d2389e5fe78…` | 0.0126 / 0.415 / 0.674 |
| 2 | [`rank02_5847af89173e.json`](rank02_5847af89173e.json) | `5847af89173e…` | 0.00491 / 0.406 / 0.677 |
| 3 | [`rank03_2bad8f1d6e95.json`](rank03_2bad8f1d6e95.json) | `2bad8f1d6e95…` | 0.00135 / 0.440 / 0.669 |

**Rank 1**（κ=0.5，τ=0.05）：

\[
\mathcal{L}=0.3\,\mathrm{ER}+0.7\,\mathrm{RD}
\]

**Rank 2**（κ=0.3，τ=0.05）：

\[
\mathcal{L}=0.4\,\mathrm{ER}+0.6\,\mathrm{RD}
\]

**Rank 3**（κ=0.3，τ=0.05）：

\[
\mathcal{L}=0.2\,\mathrm{ER}+0.6\,\mathrm{RD}+0.2\,\mathrm{SM}
\]

完整 hash：

- rank1 `3d2389e5fe78b664b7eac83ae8307233e93bc59aa4f802b7f869c4510a807242`
- rank2 `5847af89173ef1936bcca02f2053b7052034a6c39a6647d0e8658df635cb5e9b`
- rank3 `2bad8f1d6e952a268cd5316162bf6b80ab18c0b131844183429e00a2ccdce150`

## 复现训练

与搜索时同一 recipe：全量 FT、lr=`1e-5`、模型 `open-unlearning/tofu_Llama-3.2-1B-Instruct_full`、forget10 / retain90。F2 对照用 `max_steps=50`；完整协议用 `num_train_epochs=10` 且 `max_steps=-1`。

```bash
# 例：复现 rank1 的 50-step F2（路径按本机 output 改）
accelerate launch --config_file configs/accelerate/default_config.yaml src/train.py \
  --config-name=unlearn.yaml \
  experiment=unlearn/tofu/mogpu_search \
  trainer=MOGPU \
  trainer.method_args.candidate_spec_path=configs/mogpu/discovered/rank01_3d2389e5fe78.json \
  model.model_args.pretrained_model_name_or_path=open-unlearning/tofu_Llama-3.2-1B-Instruct_full \
  model.tokenizer_args.pretrained_model_name_or_path=open-unlearning/tofu_Llama-3.2-1B-Instruct_full \
  trainer.args.max_steps=50 \
  trainer.args.seed=0 \
  forget_split=forget10 \
  retain_split=retain90 \
  retain_logs_path=saves/eval/tofu_Llama-3.2-1B-Instruct_retain90/TOFU_EVAL.json
```

评测：`python src/eval.py experiment=eval/tofu/default`，`forget_split=forget10`，`holdout_split=holdout10`，同一 `retain_logs_path`。

## Rank 1 处方网格（lr × steps，先不改公式）

MU 偏低时先锁死公式 1，只扫训练处方。不是 NSGA，也不搜 κ/τ/权重。

| 轴 | 取值 |
|----|------|
| `learning_rate` | `3e-6`, `1e-5` |
| `max_steps` | `20`, `50`, `100` |

`1e-5 × 50` 复用开放搜索 F2 的 `TOFU_SUMMARY.json`。κ、τ、权重和 10 epoch 不进第一层。

读结果（对照 [`results/tofu_Llama-3.2-1B-Instruct.md`](../../../results/tofu_Llama-3.2-1B-Instruct.md)，本机 MU 约 0.53–0.59）：

- 先按 **MU 降序**，同分再按 **FQ 降序**。
- 最佳格 MU 仍 &lt; 0.53：处方不够，再考虑加大 RD 或减小 κ。
- FQ &lt; 1e-12：几乎没忘，不做 10 epoch。
- FQ 不明显高于本机 AltPO（约 6.39e-6）：不做 10 epoch。
- MU ≥ 0.53 且 FQ 明显高于 AltPO：才对该格做一次 seed 0、10 epoch。

```bash
# 只写出 6 格计划，不训练
python scripts/run_mogpu_rank01_grid.py

# GPU 就绪后再跑
python scripts/run_mogpu_rank01_grid.py --run
```

产物：`/root/autodl-tmp/saves/mogpu_rank01_grid/`（`grid_plan.json`、跑完后还有 `grid_summary.json` / `grid_ranked.json` / `epoch10_recommendation.json`）。
