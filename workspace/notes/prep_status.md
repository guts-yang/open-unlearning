# 实验准备对照（2026-09-26）

每个新 shell：

```bash
source /root/autodl-tmp/env_hf.sh
source /root/autodl-tmp/envs/unlearning/bin/activate
cd /usr/local/open-unlearning
```

`workspace/saves` → `/root/autodl-tmp/saves/workspace`。

## 本机

| 项 | 事实 |
|---|---|
| GPU | NVIDIA H20 96GB × 1 |
| 数据盘 | `/root/autodl-tmp` ~93G 可用（54% used） |
| Python | `/root/autodl-tmp/envs/unlearning`：torch 2.4.1+cu121，transformers 4.51.3 |
| H20 约束 | **fp32 + eager**；须 `trainer.args.bf16=false bf16_full_eval=false`，否则 7B ROUGE generate SIGFPE |
| HF | `env_hf.sh` → `HF_HOME=/root/autodl-tmp/huggingface` |
| Retain logs | `/usr/local/open-unlearning/saves/eval/`（7B/1B retain99 等，~39MB） |
| 插件单测 | 本地跑：`PYTHONPATH=src:workspace/src python -m pytest workspace/tests -q`（**不入库**） |

## 已完成（可跳过）

| Task | 关键指标 |
|------|---------|
| `blade_tofu_1b_01_s42` | HM **0.810** |
| `blade_tofu_1b_05_s42` | HM **0.802** |
| `grom_tofu05` | `grom_edit.json`（无 TOFU eval） |
| `baldro_npo_dv_tofu01` | FQ **0.766**, MU **0.606** (ckpt-10) |
| `baldro_npo_g_tofu01` | FQ **0.579**, MU **0.586** (ckpt-10) |

BalDRO 7B checkpoint 权重已删（留 eval JSON）；BLADE 权重在 run 根目录 `model.safetensors`。

## 下一步：SimNPO + BalDRO

```bash
bash workspace/scripts/local/run_simnpo_baldro.sh
```

日志：`workspace/saves/unlearn/_logs/simnpo_baldro.log`。脚本会清掉未完成的 partial run、带上 `retain_logs_path` 自动算 FQ。

完成后可续跑 forget10 轮：

```bash
RESUME_FROM=forget10 bash workspace/scripts/local/run_round_robin.sh
```

## 复现策略（round-robin）

| Round | 方法顺序 |
| --- | --- |
| TOFU forget01 | BLADE 1B → BalDRO NPO-DV/G → **SimNPO+BalDRO DV/G** |
| TOFU forget05 | BLADE 1B → GROM tofu05 |
| TOFU forget10 | GROM tofu10 → BLADE 1B-10 |
| MUSE / WMDP | 权重齐后 |

## 工作区布局（上传 vs 本地）

| 入库 | 不入库 |
|------|--------|
| `configs/`、`src/`、`results/`、`scripts/{train,eval,aggregate_*,run_grom}.py` | `workspace/tests/` |
| `notes/prep_status.md` | `workspace/scripts/local/`（跑数、handoff、prefetch） |
| SimNPO YAML 等实验配置 | `workspace/saves/`、权重、日志 |

预取续跑：`python workspace/scripts/local/prefetch_assets.py`
