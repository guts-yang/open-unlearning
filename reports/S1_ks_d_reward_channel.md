# Stage 1 · KS 统计量 D 与 W1 距离作为可调用 reward 通道

- 日期：2026-08-29
- 方案：主方案_v3_ES-MU_2026-08-28.md（§7 Prop 2′、§8 reward、§9 C-W1、§11 事实 1-2）
- 分支：`feature/chenyliao_20260829`，仓库 HEAD 与上游 `locuslab/open-unlearning` main（`4ad738a`）逐字一致

## 交付清单

| 文件 | 类型 | 说明 |
|---|---|---|
| `src/evals/metrics/privacy.py` | 修改（仅追加） | 新增 `ks_statistic` 与 `w1_distance` 两个指标；未改动 `ks_test`/`privleak`/`rel_diff` 任何行为 |
| `src/evals/metrics/__init__.py` | 修改（仅追加） | import 并 `_register_metric(ks_statistic)`、`_register_metric(w1_distance)` |
| `configs/eval/tofu_metrics/forget_quality_D.yaml` | 新增 | `handler: ks_statistic`，defaults/reference_logs/pre_compute 逐字段沿用 `forget_quality.yaml` |
| `configs/eval/tofu_metrics/w1_distance.yaml` | 新增 | `handler: w1_distance`，同一范式 |
| `tests/conftest.py` | 新增 | 将 `src/` 注入 `sys.path`（setup.py 无 package_dir，`pytest tests/` 需要） |
| `tests/test_ks_statistic_metric.py` | 新增 | toy 合成数据单测（见下） |

## 指标契约（与 `ks_test` 逐字一致）

- 输入：`kwargs["pre_compute"]["forget"]["value_by_index"]`（forget 集逐样本 TR，结构 `{"idx": {"score": float}}`）；retain 侧来自 `kwargs["reference_logs"]["retain_model_logs"]["retain"]["value_by_index"]`，通过 `${eval.tofu.retain_logs_path}` 指向**预先冻结**的 retain 模型 TR JSON（reward 只需单侧前向）。
- `ks_statistic` → `{"agg_value": float(D)}`，**不是 p 值**；`ks_2samp(..., method="asymp")` 显式指定（GOTCHA 2：不依赖默认 `auto`；D 与 method 无关，`asymp` 与单测验证公式同源）。
- `w1_distance` → `{"agg_value": float(W1)}`（`scipy.stats.wasserstein_distance`，支持不等长样本）。
- `reference_logs` 缺失时两指标均返回 `{"agg_value": None}` + `logger.warning`（与 `ks_test` 一致，禁止臆造值）。

## 单测（toy 合成数据，非论文数字）

覆盖项：
1. `ks_statistic` 返回 D 而非 p；
2. 渐近关系 `p ≈ 2·exp(−2·n_eff·D²)`，`n_eff = n1·n2/(n1+n2)` 由实际参与检验的样本数（n1=n2=20000）读出并打印；
3. `w1_distance` 与等长闭式解 `mean(|sorted_a − sorted_b|)` 一致；
4. 冻结 retain TR 经 `reference_logs.path` JSON 机制加载并喂给指标；
5. 新 yaml 的 `handler` 均在 `METRICS_REGISTRY` 中；
6. 缺 `reference_logs` 时返回 `None`。

## 验证状态（诚实声明）

- **`make quality`（ruff v0.6.9）：未验证** —— 用户本轮选择「仅写代码不验证」；代码按 ruff 0.6.9 black 风格书写，`read_lints` 无告警。
- **单测执行：未验证** —— 同上；本机未安装 numpy/scipy/pytest（PyPI 可达，可随时补装执行）。
- **G0（官方 ckpt 评测对齐 Table 3）：挂起** —— huggingface.co 被本机网络策略阻断（SSL EOF）且无 CUDA GPU（Apple M5 Pro），已获用户豁免「不得跳过 G0」顺序约束；待 GPU/HF 环境就绪后按 Stage 0→2→… 补做。

## 后续衔接

- Stage 2/4/6 依赖 G0 的实测环境（checkpoint、GPU 评测、Table 3 基线）。
- 本 Stage 的 `ks_statistic`/`w1_distance` 已注册，供 Stage 3/5（`subspace.py`、`ESMU` trainer）以配置方式调用。
- 建议在补跑时：`uv venv --python 3.11` + `uv pip install -e ".[dev]"` + `pytest tests/ -v`，并核对打印出的 `n1/n2/n_eff/D/p/p_asym`。

---

## 修订记录（2026-08-29 代码审查后）

| # | 问题 | 处理 |
|---|---|---|
| 1 | 🔴 **新 yaml 不会被执行**：`configs/eval/tofu.yaml` 的 `defaults: - tofu_metrics:` 是显式列表，不会自动扫描 `tofu_metrics/` 目录 | 已把 `forget_quality_D`、`w1_distance` 加入该列表。两者复用已在列表中的 `forget_Truth_Ratio` pre-compute，故增量成本仅为最后的聚合 |
| 2 | 🟠 **单测自身脆弱**：`test_ks_statistic_asymptotic_p_formula` 用 shift=0.02，经验 `D·√n_eff` 落在 0.8–2.0，而其前置断言要求 `> 1.0` | shift 改为 0.1，`λ = D·√n_eff ≈ 4`，远离临界；docstring 补 Smirnov 展开的说明 |
| 3 | 🟠 **DRY 违规**：`ks_statistic` / `w1_distance` / `ks_test` 前 12 行逐字相同 | 新增私有 `_collect_tr_arrays()`，两个**新**指标改用之（净减约 40 行）。**`ks_test` 保持逐字不变**，以符合「只增不改」纪律；若日后允许，可再单独提一个纯重构 commit 让 `ks_test` 也复用 |
| 4 | 🟡 `tests/__pycache__/` 混入方向A（MOGP-U）的 `test_mogpu_*.pyc` 共 5 个；且已核实**上游 locuslab 仓库无 `tests/` 目录**（API 404），该套件为全新引入 | `rm -rf tests/__pycache__`；提交前确认 `.gitignore` 已覆盖 `__pycache__/` |
| 5 | 🟡 `.gitignore` 新增 `.cursor/` `.codebuddy/`，与 Stage 1 无关 | 建议拆成独立 commit（`--other`） |

### 验证状态更新

- **依赖项已就位但不代表已验证**：隔离环境 `~/.workbuddy/binaries/python/envs/default` 内已装 `scipy 1.18.1` / `pytest 9.1.1` / `omegaconf 2.0.0` / `torch 2.13.0` / `transformers 5.16.1` / `datasets 5.0.1`；`ruff 0.6.9` 需再装到 `$V/bin/ruff`。
- **`ruff` 与 `pytest` 截至本条记录时尚未实际执行**（本机 shell 会话中途失效）。待执行的命令见 `09_OpenUnlearning实现提示词_分阶段.md` §E5。
- ⚠️ 注意：本机装的是 **transformers 5.x / datasets 5.x**，与仓库要求的 4.51.3 不一致。仅用于让 `import evals.metrics` 通过以便跑单测；**真实实验一律在远程 A800 上按仓库 `requirements.txt` 装环境**。

### 执行环境（用户 2026-08-29 确认）

- 远程：**2× A800 80GB**；本地 Apple M5 Pro（仅写代码与轻量验证）。
- A800 相对 A100 只阉割了 NVLink（600→400 GB/s），算力与显存相同 ⟹ 上述 GPU-h 估算直接适用。
- **ES 几乎不吃互联带宽**（每步只传 30 个标量 reward + 一个 37 KB 的 Gram），故 A800 的短板对本方案无效。
- 2 卡并行后：单步 7.2 s，全套 75 跑约 15 GPU-h，墙钟并发约 7.5 h。
