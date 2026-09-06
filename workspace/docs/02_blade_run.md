# BLADE 远程训练

论文锚点（Table 1/2，5-seed 均值；我们若只跑 1 seed 在 notes 注明）：

| 设置 | HM |
|------|----|
| TOFU 1B 01/05/10 | 0.808 / 0.800 / 0.803 |
| TOFU 3B 01/05/10 | 0.846 / 0.842 / 0.836 |
| MUSE Books / News | 0.818 / 0.544 |

Seeds：**42 123 456 789 1024**。不要用官方 `tofu_blade.sh` 的 1337。News 必须用 `blade_muse_news`，不要套 Books YAML。

验收：绝对偏差 ±0.03。指标：`python workspace/scripts/aggregate_blade.py` 计算 `hmean(MU, 1-Prob, 1-RG)`。

## 冒烟

```bash
python workspace/scripts/train.py experiment=unlearn/blade_tofu01_smoke \
  task_name=blade_tofu01_smoke \
  paths.output_dir=workspace/saves/unlearn/blade_tofu01_smoke
```

## 全量示例（1B forget01, seed 42）

```bash
python workspace/scripts/train.py experiment=unlearn/blade_tofu_1b_01 \
  trainer.args.seed=42 \
  task_name=blade_tofu_1b_01_s42 \
  paths.output_dir=workspace/saves/unlearn/blade_tofu_1b_01_s42
```

其余实验名：

- `unlearn/blade_tofu_1b_{01,05,10}`
- `unlearn/blade_tofu_3b_{01,05,10}`
- `unlearn/blade_muse_books`、`unlearn/blade_muse_news`

forget10 的 `T=500`，01/05 为 `T=250`。两张 A800 可并行不同 seed，但单进程训练循环是自定义 bilevel，不要默认 DeepSpeed ZeRO-3。

KnowUnDo：延后（本 fork 无 evaluator，目标需自训）。

跑完 BLADE 后写 `workspace/notes/` 中间报告，**停**，确认后再 BalDRO。
