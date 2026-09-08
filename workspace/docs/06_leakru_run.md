# 06 · LeakRU 运行说明

总入口与产物约定见 [`05_eval_and_results.md`](05_eval_and_results.md)。设计备忘见 [`06_leakru_integration.md`](06_leakru_integration.md)。

## Schema

Hydra 数据集组放在 `workspace/configs/leakru_datasets/`（仓库根 `.gitignore` 的 `data/` 会挡住 `workspace/configs/data/`）。

每条样本至少：`question`（str）、`answer`（str）、`aliases`（list）、`logic_type` ∈ {HS,MT,DS,LL,MP,CC}、`split` ∈ {single-hop, multi-hop}。可选 `key_tokens`、`id`。

查找顺序：`eval.leakru.data_path` → 环境变量 `LEAKRU_DATA_DIR` → `/root/autodl-tmp/data/leakru/`。目录下认 `mquake.jsonl` / `mquake.json` 等。缺失则 `FileNotFoundError`，不返回空集。

`V(·)`：gold 与 aliases 的规范化子串匹配。论文附录 E 的 LLM-judge prompt 未公开，这不是官方 judge。

攻击默认（论文 Table 5）：Probab T=0.8 / top-p=0.95 / 5 trials，≥1 次命中即对；FocusOnKey greedy，重复 key tokens；Quantization greedy INT4（`bitsandbytes` nf4，需本地权重）。

`leakru_fq` 与 TOFU KS `forget_quality` 不是同一个数。`pretrained_logs_path=null` 时 FQ/RR 为 null，不当 0。

## 本机代码测试（无 GPU / 不下载）

```bash
python -m pytest workspace/tests/test_leakru.py workspace/tests/test_register.py workspace/tests/test_configs.py -q
```

## 远程（部署后再跑）

```bash
source /root/autodl-tmp/env_hf.sh
source /root/autodl-tmp/envs/unlearning/bin/activate

# 1) θ_p
python workspace/scripts/eval.py --config-name=eval.yaml \
  experiment=eval/tofu_leakru_mquake \
  task_name=leakru_theta_p \
  eval.leakru.per_type_limit=20 \
  eval.leakru.overwrite=true \
  eval.tofu.overwrite=true \
  paths.output_dir=workspace/saves/eval/leakru_theta_p

# 2) θ_u（把 model 指到遗忘 ckpt，pretrained_logs_path 指上一步 LeakRU_EVAL.json）
python workspace/scripts/eval.py --config-name=eval.yaml \
  experiment=eval/tofu_leakru_mquake \
  task_name=leakru_theta_u \
  eval.leakru.per_type_limit=20 \
  eval.leakru.overwrite=true \
  eval.leakru.pretrained_logs_path=workspace/saves/eval/leakru_theta_p/LeakRU_EVAL.json \
  model.model_args.pretrained_model_name_or_path=/path/to/unlearned \
  paths.output_dir=workspace/saves/eval/leakru_theta_u
```

全量 MQuAKE 去掉 `per_type_limit` 或设 500。LeakRU-only：`experiment=eval/leakru_mquake`。

已有 GROM/BLADE 实验 opt-in：`override /eval: tofu` 改为 `tofu_and_leakru`，并设 `eval.leakru.pretrained_logs_path`。
