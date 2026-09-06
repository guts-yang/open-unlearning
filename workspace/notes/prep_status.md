# 实验准备对照（2026-09-06 本机盘点）

对照 `workspace/docs/01–05` 与 `docs/zh/reproduce-prep.md`。已就绪项不再下载、不再重装。

每个新 shell：

```bash
source /root/autodl-tmp/env_hf.sh
source /root/autodl-tmp/envs/unlearning/bin/activate
cd /usr/local/open-unlearning
```

`workspace/saves` → `/root/autodl-tmp/saves/workspace`（已建软链）。

## 跳过（已就绪）

| 项 | 位置 / 事实 |
|---|---|
| Python 环境 | `/root/autodl-tmp/envs/unlearning`：torch 2.8.0+cu128、transformers 4.51.3、flash-attn 2.8.3、lm-eval 0.4.11；**不要** conda create / 不要用 miniconda base |
| HF 镜像与缓存 | `env_hf.sh`：`HF_HOME=/root/autodl-tmp/huggingface`，`HF_ENDPOINT=https://hf-mirror.com` |
| peft 0.14.0 | 已装进上述 venv（手册要求，原先缺） |
| 插件单测环境 | `PYTHONPATH=src:workspace/src python -m pytest workspace/tests -q`：**10 passed / 2 failed**（见下，不挡 BLADE 1B） |
| TOFU 数据 | Hub `locuslab/TOFU` |
| MUSE 数据 | Hub `muse-bench/MUSE-{Books,News}` |
| WMDP 评测 + MMLU | Hub `cais/wmdp`、`cais/mmlu` |
| WMDP Cyber 训练语料 | `data/wmdp/wmdp-corpora/{cyber-forget,cyber-retain}-corpus.jsonl`（仓库 `data/wmdp` 已软链数据盘） |
| BLADE TOFU 1B 权重 | `open-unlearning/tofu_Llama-3.2-1B-Instruct_full`（2.4G） |
| MUSE 目标权重 | `muse-bench/MUSE-{Books,News}_target`（各 26G） |
| ALTER Zephyr | `HuggingFaceH4/zephyr-7b-beta`（27G） |
| TOFU retain 评测日志 | `/root/autodl-tmp/saves/eval/` 已有 1B/3B/Llama-2-7b-chat 的 retain90/95/99 与 full 的 `TOFU_EVAL.json`。**不要**再跑 `setup_data.py --eval_logs` |
| 仓库副本 | 只此一份 `/usr/local/open-unlearning` |

TOFU FQ 评测时 CLI 传绝对路径，例如 forget01：

`eval.tofu.retain_logs_path=/root/autodl-tmp/saves/eval/tofu_Llama-3.2-1B-Instruct_retain99/TOFU_EVAL.json`

（forget05→retain95，forget10→retain90。）

## 未做 / 阻塞

| 项 | 状态 | 何时才需要 |
|---|---|---|
| **GPU** | 当前 `nvidia-smi` 空、`torch.cuda.is_available()==False` | **冒烟与全量训练的硬阻塞**；AutoDL 开机挂卡后再跑 |
| BLADE TOFU 3B 权重 | Hub 无 `tofu_Llama-3.2-3B-Instruct_full` | 1B 停一次验收后再下 |
| BalDRO Llama-2-7B 权重 | 无 `tofu_Llama-2-7b-chat-hf_full` | BLADE 停后再下 |
| ALTER Llama3-8B | 论文未钉 ID，本地无 Meta-Llama-3-8B | ALTER 阶段再定 ID |
| `NousResearch/Llama-2-7b-hf` | Hub 目录仅 ~2.3M，权重不完整 | BLADE MUSE 用的是 `MUSE-*_target`，先不补 |
| WMDP Bio forget jsonl | `wmdp-corpora_jsonl/` 无 `bio-forget-corpus.jsonl` | 按手册：Bio 训练格子写「未复现」 |
| `setup_data.py --wmdp` | Cyber+retain 已在，跳过整包重下 | — |
| 官方 `saves/`、仓库内第二份 HF 缓存 | 不要建 | 产物只走 `workspace/saves` 软链 |

## 单测失败（不改官方树；不挡 BLADE 1B）

- `test_asym_lora_freeze_base_and_train_adapters`：`wrap_linear_modules` 未 `requires_grad_(False)` 冻结 `base`。ALTER 开训前再修。
- `test_dv_large_beta_near_mean`：`beta_dv=1e6` 在 float32 下 logsumexp 均值偏差约 0.02。论文默认 `beta_dv_forget=2.0`，BalDRO 开训前再核。

## 下一步（挂卡后，手册顺序）

1. 冒烟：`python workspace/scripts/train.py experiment=unlearn/blade_tofu01_smoke task_name=blade_tofu01_smoke paths.output_dir=workspace/saves/unlearn/blade_tofu01_smoke`
2. BLADE TOFU 1B 全量（seed 42 123 456 789 1024），YAML `unlearn/blade_tofu_1b_{01,05,10}`；评测带上 `retain_logs_path`
3. 写 notes、停；确认后再 3B / MUSE / BalDRO / ALTER

汇总 Markdown 仍全是「未复现」，符合「无实测 JSON 不填论文数字」。
