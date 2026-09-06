# ALTER 远程训练（推定实现）

**原文未提供可复现配置**：官方仓库只有空 README；公开 PDF 无完整超参表/附录闭式/可运行脚本。本地 `ALTER.yaml` 是按正文残缺旋钮推定的，跑数与写表都必须标「非论文配置复现」，不要和 Table 1/2 论文数字对齐验收。

IHL 闭式、Tsallis `q`、expert 数、LoRA 模块均来自正文推断 + 默认，见 YAML `reconstruction_note`。推理是 **条件路由**，不要 `merge_and_unload` 后再评。

## 可先做（公开数据）

1. TOFU 小规模结构验证（可选）
2. WMDP-Cyber 训练：`experiment=unlearn/alter_wmdp_cyber`
3. 评测：

```bash
python workspace/scripts/eval.py experiment=eval/wmdp/default data_split=cyber \
  eval.lm_eval.tasks='[wmdp_cyber,mmlu]' \
  model.model_args.pretrained_model_name_or_path=<ckpt> \
  paths.output_dir=workspace/saves/unlearn/alter_wmdp_cyber_eval
```

Zephyr 论文列：Cyber ≈ 24.0，MMLU ≈ 56.4。

## Bio

forget corpus 需 CAIS 申请。获批前评测集选择题可跑，**训练格子写未复现**。

Llama3-8B 锚点：Bio 24.4 / Cyber 25.6 / MMLU 57.8。模型 ID 必须写进 notes。

## MUSE / HarryPotter

OpenUnlearning MUSE 指标与论文 BLEU/ROUGE-L/ASG **分列**。GPT-4o 流畅度协议未公开 → 未复现。
