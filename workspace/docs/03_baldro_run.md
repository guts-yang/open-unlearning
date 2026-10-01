# BalDRO 远程训练

Llama-2-7B 全参，2×A800 可单卡 80GB 跑脚本默认 batch；先不要上 ZeRO-3，除非 OOM。

## 对照锚点（论文 Table 1，forget01）

| 方法 | FQ | MU |
|------|----|----|
| NPO | 0.7659 | 0.5775 |
| NPO+G | 0.9188 | 0.6126 |
| NPO+DV | 0.9900 | 0.5815 |

forget05：NPO FQ 0.6284 → DV ≈ 0.9646。

论文 DV 峰值常用 `beta_dv_forget=2.0`；上游 YAML 默认 1.0。本仓库 `DrNPO.yaml` 默认 **2.0**。另跑一次 `trainer.method_args.beta_dv_forget=1.0` 作对照。Seed **0**。

## 本轮范围

只跑 **NPO+DV**（`DrNPO`）和 **SimNPO+DV**（`DrSimNPO`）。forget10、BalDRO-G、SatImp 不在本轮。

| 实验名 | 数据 | 对照日志（已写进 YAML） |
|--------|------|-------------------------|
| `unlearn/baldro_npo_dv_tofu01` | TOFU forget01 | `tofu_Llama-2-7b-chat-hf_retain99` |
| `unlearn/baldro_simnpo_dv_tofu01` | TOFU forget01 | 同上 |
| `unlearn/baldro_npo_dv_tofu05` | TOFU forget05 | `..._retain95` |
| `unlearn/baldro_simnpo_dv_tofu05` | TOFU forget05 | 同上 |
| `unlearn/baldro_{npo,simnpo}_dv_muse_news` | MUSE News | `muse_Llama-2-7b-hf_News_retrain/MUSE_EVAL.json` |
| `unlearn/baldro_{npo,simnpo}_dv_muse_books` | MUSE Books | `muse_Llama-2-7b-hf_Books_retrain/MUSE_EVAL.json` |

起点权重：`open-unlearning/tofu_Llama-2-7b-chat-hf_full`。MUSE 用已在盘上的 `muse-bench/MUSE-{News,Books}_target`。本机 H20 配置为 fp32 + eager，`gradient_checkpointing=true`。TOFU batch 8×4；MUSE 序列 2048，batch 4×8。论文没有写死 MUSE 的 lr / batch，这里沿用 OpenUnlearning MUSE 默认 lr `1e-5`。

```bash
source /root/autodl-tmp/env_hf.sh
source /root/autodl-tmp/envs/unlearning/bin/activate
python workspace/scripts/train.py experiment=unlearn/baldro_npo_dv_tofu01
```

八个实验名换成上表即可，输出目录由 YAML 的 `paths.output_dir` 指到 `workspace/saves/unlearn/<task_name>`。TOFU 汇总：`python workspace/scripts/aggregate_baldro.py`。MUSE 读各自目录里的 `MUSE_SUMMARY.json`。
