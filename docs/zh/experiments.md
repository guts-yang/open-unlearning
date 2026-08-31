# 实验配置与运行

对应原文：[`docs/experiments.md`](../experiments.md)

> 下列命令已改成本仓库实际入口：`src/train.py`、`src/eval.py`、`experiment=unlearn/tofu/default` 等。英文原文里部分写成了 `src/train.py` / `experiment=unlearn/tofu/default`。

## 概述

本仓库支持的组件变体很多，跑一次实验前需要配置大量模块和超参。项目用 **Hydra** 把这件事变简单。

核心有三份底配置：

- `train.yaml`：普通训练
- `eval.yaml`：评测
- `unlearn.yaml`：遗忘训练

再叠加上实验专用 yaml 和命令行覆盖。常见场景（例如 LLaMA-2 在 TOFU 上遗忘、在 MUSE 上评测）已写好 experiment 配置，会填好数据集、模型和训练/评测底配置。

输出目录由任务模式（`train` / `eval` / `unlearn`）和用户给定的 `task_name` 拼成：`./saves/${mode}/${task_name}`。日志里会打印 checkpoint、log 和评测结果的位置。

## 示例命令

```bash
## 微调：configs/experiment/finetune/tofu/default.yaml
python src/train.py --config-name=train.yaml experiment=finetune/tofu/default task_name=SAMPLE_TRAIN

## 遗忘训练：configs/experiment/unlearn/tofu/default.yaml
# 输出目录：saves/unlearn/SAMPLE_UNLEARN
python src/train.py --config-name=unlearn.yaml experiment=unlearn/tofu/default task_name=SAMPLE_TRAIN

## 评测：configs/experiment/eval/muse/default.yaml
python src/eval.py --config-name=eval.yaml experiment=eval/muse/default task_name=SAMPLE_EVAL
# eval.yaml 已是 src/eval.py 的默认 config，可省略 --config-name

## 较完整的遗忘实验
python src/train.py --config-name=unlearn.yaml experiment=unlearn/muse/default data_split=News \
  trainer=NPO trainer.method_args.retain_loss_type=KL task_name=llama2_books_NPO_KL \
  retain_logs_path=saves/eval/muse_books_retain/MUSE_EVAL.json

## 更完整的遗忘实验
python src/train.py --config-name=unlearn.yaml \
  experiment=unlearn/tofu/default.yaml \
  task_name=NPO_unlearn_tofu_llama_8 \
  model=Llama-3.1-8B-Instruct \
  model.model_args.pretrained_model_name_or_path=saves/finetune/path_model_llama \
  trainer=NPO trainer.args.per_device_train_batch_size=4 \
  forget_split=forget05 retain_split=retain95 \
  retain_logs_path=saves/eval/tofu_retain95/TOFU_EVAL.json \
  paths.output_dir=saves/unlearn/NPO/evals
```

遗忘训练支持**训练过程中评测**，但仅当 **单个 accelerator 进程** 时可用；多卡 DeepSpeed 时必须先存 checkpoint，训练后再评测。

## 常用覆盖参数

评测配置结构与可覆盖字段示例：[`configs/experiment/examples/tofu_eval.yaml`](../../configs/experiment/examples/tofu_eval.yaml)

遗忘配置结构示例：[`configs/experiment/examples/muse_unlearn.yaml`](../../configs/experiment/examples/muse_unlearn.yaml)

### 模型

| 参数 | 说明 |
|------|------|
| `model` | 选用模型配置。例：`model=Llama-2-7b-hf` |
| `model.model_args.pretrained_model_name_or_path` | checkpoint 或 HuggingFace ID |
| `model.tokenizer_args.pretrained_model_name_or_path` | tokenizer 路径，需与模型一致 |
| `model.template_args` | 可选 chat 模板（起止 tag 等）。例：`apply_chat_template: false` |

### Trainer

| 参数 | 说明 |
|------|------|
| `trainer` | 训练/遗忘算法。例：`trainer=NPO` 或 `trainer=finetune` |
| `trainer.args` | HuggingFace `TrainingArguments`：batch、lr、epoch、`optim` 等 |
| `trainer.method_args` | 方法专用超参。例：`retain_loss_type`，NPO 的 `gamma, alpha, beta` |

### 数据

| 参数 | 说明 |
|------|------|
| `data` | 数据格式。例：`data=unlearn`、`data=finetune` |
| `data.forget` / `data.retain` / `data.anchor` | 指定子数据集；`data.anchor=forget` 表示按 forget 下标遍历，retain 随机抽 |
| `data_split` / `forget_split` / `retain_split` | 填充数据集路径。MUSE 用 `data_split=News` 或 `Books`；TOFU 用 `forget_split=forget01 retain_split=retain99` |

### 实验

| 参数 | 说明 |
|------|------|
| `task_name` | 实验名，用于输出路径。例：`task_name=llama2_books_NPO_KL` |
| `eval` | 评测套件。例：`eval=muse` |
| `retain_logs_path` | retain 模型评测 log，供相对指标使用 |
| `paths` | 路径相关。例：`paths.output_dir=$LOCAL_PATH` |

## 普通微调

除遗忘外，也支持在给定数据上做标准监督微调：`src/train.py` + `train.yaml`。

```bash
python src/train.py --config-name=train.yaml experiment=finetune/tofu/default \
  trainer.args.learning_rate=5e-5 task_name=llama3.2-1B_finetune_example
```

## 分布式训练

多数情况直接用 [`configs/accelerate/default_config.yaml`](../../configs/accelerate/default_config.yaml)（默认 DeepSpeed）：

```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
  --config_file configs/accelerate/default_config.yaml --main_process_port 18765 \
  src/train.py --config-name=unlearn.yaml experiment=unlearn/muse/default.yaml task_name=DISTRIBUTED_TRAIN
```

也可 `CUDA_VISIBLE_DEVICES=0,1 python ...` 走 Accelerate 的 DDP / 模型并行。模型并行可在 `model_args` 里设 `device_map="auto"`。

多进程 Accelerate **无法在训练中途跑自定义评测**。需要训练中评测时，改用单进程 DDP/模型并行，或训练结束后单卡评测：

```bash
CUDA_VISIBLE_DEVICES=0 python src/eval.py experiment=eval/muse/default.yaml task_name=SAMPLE_EVAL \
  model.model_args.pretrained_model_name_or_path=saves/unlearn/muse_unlearn_exp
```
