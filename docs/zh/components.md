# 组件扩展

对应原文：[`docs/components.md`](../components.md)

在遗忘流水线中加新组件，固定三步：

1. **实现 handler**：核心逻辑（类或函数）。同一 handler 可被多个组件复用，例如一个 ROUGE 函数服务多套数据集上的多个指标。
2. **注册**：写入对应 `*_REGISTRY`，供配置里的名字查找。
3. **写 Hydra yaml**：指定 handler 名和参数，运行脚本时直接引用。

Hydra 写法见 [Hydra 用法](./hydra.md)。

## 各组件说明

### Trainer（训练 / 遗忘算法）

继承 HuggingFace `Trainer`，代码在 [`src/trainer`](../../src/trainer/)。遗忘方法通常重写 `compute_loss`。

注册：[`TRAINER_REGISTRY`](../../src/trainer/__init__.py)，键为类名。

配置：[`configs/trainer`](../../configs/trainer/)。`handler` 对应类名；`args` 为 `TrainingArguments`；`method_args` 为方法自己的超参（如 GradDiff 的 `gamma`、`alpha`、`retain_loss_type`）。

### Dataset

继承 `torch.utils.data.Dataset`，在 [`src/data`](../../src/data/) 做加载与预处理。同一 handler（如 `PretrainingDataset`）可通过不同 yaml 实例化成 `MUSE_forget`、`MUSE_forget_sust` 等。

注册：[`DATASET_REGISTRY`](../../src/data/__init__.py)。

配置：[`configs/data/datasets`](../../configs/data/datasets/)。

遗忘时，forget 与 retain 会再包成 `ForgetRetainDataset`：默认 `anchor=forget`（epoch 长度等于 forget），retain 每个 step 随机抽一条。

### Evaluation Metric

指标计算逻辑 + 配置。详见 [评测](./evaluation.md)。

### Benchmark

把多条 metric 收成一套（TOFU、MUSE）。详见 [评测](./evaluation.md)。

### Model

多数情况用 `AutoModelForCausalLM` / `AutoTokenizer`。自定义（如带 probe 的 Llama）放在 [`src/model`](../../src/model/)，注册到 [`MODEL_REGISTRY`](../../src/model/__init__.py)。

当前**不支持 LoRA 等 PEFT 加载**；若需要，自行实现并注册 model handler。

配置：[`configs/model`](../../configs/model/)（路径、chat 模板、`attn_implementation` 等）。

### Collator

按数据集格式做 padding 与组 batch。多数用户不必新写。实现与注册见 [`src/data`](../../src/data/)（collator 与 dataset 注册在同一模块）；配置在 [`configs/collator`](../../configs/collator/)。

### Experiment

把 model、data、trainer、eval 组合成一次可跑实验。**没有 Python handler**，只靠 Hydra。文件在 [`configs/experiment`](../../configs/experiment/)。

典型 TOFU 遗忘实验会：`defaults` 里选模型、GradAscent、unlearn 数据格式、forget/retain 数据集、TOFU eval；再覆盖 `forget_split`、`retain_split`、target 模型路径、学习率与 epoch。`task_name` 未设时会报错（`???`）。

运行与覆盖方式见 [实验配置与运行](./experiments.md)。
