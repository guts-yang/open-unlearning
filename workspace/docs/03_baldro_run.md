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

## 顺序

1. 官方 NPO 基线（`trainer=NPO`，输出仍必须在 `workspace/saves/`）
2. `experiment=unlearn/baldro_npo_dv_tofu01`
3. `experiment=unlearn/baldro_npo_g_tofu01`
4. 扩展 forget05/10、SimNPO、SatImp、MUSE

```bash
python workspace/scripts/train.py experiment=unlearn/baldro_npo_dv_tofu01 \
  eval.tofu.retain_logs_path=<retain99 TOFU_EVAL.json> \
  task_name=baldro_npo_dv_tofu01 \
  paths.output_dir=workspace/saves/unlearn/baldro_npo_dv_tofu01
```

汇总：`python workspace/scripts/aggregate_baldro.py`。完成后停，再 ALTER。
