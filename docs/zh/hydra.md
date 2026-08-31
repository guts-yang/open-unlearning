# Hydra 用法（本仓库用到的部分）

对应原文：[`docs/hydra.md`](../hydra.md)

下面用 [`configs/experiment/unlearn/muse/default.yaml`](../../configs/experiment/unlearn/muse/default.yaml) 说明常见写法（结构示意，键名以仓库文件为准）。

- **`# @package _global_`**：不是注释，表示该文件内容挂到配置树根上。
- **`defaults` + `override`**：从其他 yaml 填入 `model`、`trainer`、`data`、`eval` 等节点。
- **变量**：`data_split: News` 后可用 `${data_split}` 拼路径，例如 `muse-bench/MUSE-${data_split}_target`。
- **`task_name: ???`**：未在命令行或配置里提供时直接报错。

常用特性：

1. **层级访问**：代码里 `cfg.model.args.learning_rate`；命令行 `model.args.learning_rate=1e-5`。
2. **命令行覆盖**：任何已有字段都可覆盖，例如  
   `python src/train.py --config-name=unlearn.yaml experiment=unlearn/muse/default trainer.args.num_train_epochs=50 data_split=Books trainer=SimNPO task_name=unlearn_muse_simnpo`
3. **`# @package eval.muse.metrics.xxx`**：把单个 metric 文件挂到 `eval.muse.metrics.xxx`，而不是根节点。
4. **`${}` 插值**：一处定义、多处引用。
5. **`+` 新增字段**：配置里原本没有的键，用 `+trainer.args.my_new_arg=10`。
6. **`~` 删除字段**：例如去掉 flash attention：  
   `"~model.model_args.attn_implementation"`  
   在 **zsh** 里必须给 `~` 加引号或转义，否则会被当成 home 目录。

Hydra 经 PyYAML 解析，`true` 会变成 Python `True`。

评测侧可覆盖字段示例：[`configs/experiment/examples/tofu_eval.yaml`](../../configs/experiment/examples/tofu_eval.yaml)  
遗忘侧示例：[`configs/experiment/examples/muse_unlearn.yaml`](../../configs/experiment/examples/muse_unlearn.yaml)
