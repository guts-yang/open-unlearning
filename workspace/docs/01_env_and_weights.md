# 环境、权重与数据（远程 2×A800 上执行）

本机阶段**不要**下载权重。连上服务器后再按本节操作。

## 环境

```bash
conda create -n unlearning python=3.11
conda activate unlearning
cd /path/to/open-unlearning
pip install ".[lm-eval]"
pip install --no-build-isolation flash-attn==2.6.3   # 按 README，失败则记录后继续
pip install peft==0.14.0
```

确认：`python -c "import torch; print(torch.cuda.device_count(), torch.cuda.get_device_name(0))"` 应为 2 张 A800。

插件单测（仍不下载模型）：

```bash
PYTHONPATH=src:workspace/src python -m pytest workspace/tests -q
```

## Hugging Face 权重（训练前）

登录有权限的账号后拉取：

| 用途 | ID |
|------|----|
| BLADE TOFU 1B | `open-unlearning/tofu_Llama-3.2-1B-Instruct_full` |
| BLADE TOFU 3B | `open-unlearning/tofu_Llama-3.2-3B-Instruct_full` |
| GROM / BalDRO TOFU 7B | `open-unlearning/tofu_Llama-2-7b-chat-hf_full` |
| GROM WMDP | `meta-llama/Meta-Llama-3-8B-Instruct` |
| MUSE Books/News 目标 | `muse-bench/MUSE-Books_target`, `muse-bench/MUSE-News_target` |
| ALTER Zephyr | `HuggingFaceH4/zephyr-7b-beta` |
| ALTER Llama3-8B | 论文未钉 ID；候选 `meta-llama/Meta-Llama-3-8B` 或 Instruct，写入 notes |

retain 评测日志：

```bash
python setup_data.py --eval_logs
```

将需要的 `saves/eval/.../TOFU_EVAL.json` **复制或软链**到实验可读路径，并在 CLI 设
`eval.tofu.retain_logs_path=...`。不要改 `configs/paths/default.yaml`。

## 数据

- TOFU：`locuslab/TOFU`（HF，公开）
- MUSE：`muse-bench/MUSE-{News,Books}`
- WMDP 评测：`cais/wmdp`；训练语料 `python setup_data.py --wmdp`（**GROM/ALTER 的 Bio forget jsonl 可能仍需申请**）
- GROM MUSE 窗口语料：官方期望 `data/muse_{news,books}/{forget,retain}_train.jsonl`

## 路径守卫冒烟（下载后、全量前）

```bash
python workspace/scripts/train.py \
  experiment=unlearn/blade_tofu01_smoke \
  task_name=blade_tofu01_smoke \
  paths.output_dir=workspace/saves/unlearn/blade_tofu01_smoke
```

确认 hydra 日志与 ckpt 只出现在 `workspace/saves/unlearn/blade_tofu01_smoke/`。
若误写 `saves/unlearn/...`（仓库根），入口应直接 `SystemExit`。
