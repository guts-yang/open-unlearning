# 创建与运行评测

对应原文：[`docs/evaluation.md`](../evaluation.md)

评测流水线：某个 **evaluator**（对应一个基准，如 TOFU、MUSE）接收模型和一组 **metric**，计算并写出结果。评测设定写在 experiment 配置里，可直接使用。

新增指标见下文「指标」；新增基准见「Benchmark」。

## 快速评测

在 LLaMA 3.2 checkpoint 上跑 TOFU：

```bash
python src/eval.py --config-name=eval.yaml \
  experiment=eval/tofu/default \
  model=Llama-3.2-3B-Instruct \
  model.model_args.pretrained_model_name_or_path=<LOCAL_MODEL_PATH> \
  task_name=SAMPLE_EVAL
```

- `--config-name=eval.yaml`：[`configs/eval.yaml`](../../configs/eval.yaml)
- `experiment=eval/tofu/default`：[`configs/experiment/eval/tofu/default.yaml`](../../configs/experiment/eval/tofu/default.yaml)
- `model=Llama-3.2-3B-Instruct`：覆盖默认的 1B 模型配置
- 输出目录：`saves/eval/SAMPLE_EVAL`

在 MUSE-Books 上评测：

```bash
python src/eval.py --config-name=eval.yaml \
  experiment=eval/muse/default \
  data_split=Books \
  model=Llama-2-7b-hf \
  model.model_args.pretrained_model_name_or_path=<LOCAL_MODEL_PATH> \
  task_name=SAMPLE_EVAL
```

- `eval.yaml` 已是默认，可省略 `--config-name`
- `data_split=Books`：覆盖默认的 News 划分

## 指标

指标要么：对模型和数据集逐条统计；要么：基于其他指标再聚合。

既有逐点、又有均值的指标（概率、ROUGE、MIA、Truth Ratio 等）返回：

```text
{"agg_value": ..., "value_by_index": {"0": ..., "1": ..., ...}}
```

只给出整体分数的指标（TOFU 的 Forget Quality：比较 forget/retain 上 Truth Ratio 分布；MUSE 的 PrivLeak：比较 forget/holdout 上 MIA）返回 `{"agg_value": ...}`。

### 新增指标的三步

**1. 实现 handler**（[`src/evals/metrics`](../../src/evals/metrics/)）

用 `@unlearning_metric` 装饰函数。装饰器会按 yaml 自动准备 dataset、collator，放进 `kwargs`。

```python
@unlearning_metric(name="rouge")
def rouge(model, **kwargs):
    tokenizer = kwargs["tokenizer"]
    data = kwargs["data"]
    collator = kwargs["collators"]
    batch_size = kwargs["batch_size"]
    generation_args = kwargs["generation_args"]
    ...
    return {"agg_value": np.mean(rouge_values), "value_by_index": scores_by_index}
```

`kwargs` 中常见字段：`tokenizer`、`data`、`batch_size`、`collator`、`generation_args`、`pre_compute`（依赖的上游指标）、`reference_logs`（对照模型评测结果），以及 yaml 里写的方法参数。

**2. 注册**到 [`METRICS_REGISTRY`](../../src/evals/metrics/__init__.py)。

**3. 写 yaml**（[`configs/eval/tofu_metrics`](../../configs/eval/tofu_metrics/) 或 [`configs/eval/muse_metrics`](../../configs/eval/muse_metrics/)）。同一 handler 可配出多个指标。

文件开头的 `# @package eval.muse.metrics.forget_verbmem_ROUGE` **不是注释**，用来把该文件挂到最终 config 的对应路径下。

依赖其他指标时，用 `pre_compute`（避免重复计算），以及 `reference_logs`（读 retain 模型 JSON）。例如 Forget Quality 依赖 Truth Ratio，并与 retain 模型的 Truth Ratio 做 KS 检验。

## Benchmark

Benchmark（evaluator）是一组指标的集合（TOFU、MUSE 等）。

1. 在 [`src/evals`](../../src/evals/) 实现 handler（聚合、汇报、评测前准备模型等）。
2. 注册到 [`EVALUATOR_REGISTRY`](../../src/evals/__init__.py)。
3. 在 [`configs/eval`](../../configs/eval/) 写配置（如 [`configs/eval/tofu.yaml`](../../configs/eval/tofu.yaml)），用 `defaults` 引入各 metric yaml。`# @package` 会把它们填进 `metrics` 映射。

## lm-evaluation-harness

通用能力评测通过 [`LMEvalEvaluator`](../../src/evals/lm_eval.py)。任务列在 [`configs/eval/lm_eval.yaml`](../../configs/eval/lm_eval.yaml) 的 `tasks` 下（如 `mmlu`、`wmdp_cyber`、`gsm8k`）。
