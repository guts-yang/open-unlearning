# 未定义设计决策清单（OPEN DESIGN DECISIONS）

> 纪律（用户任务提示词）：「遇到未定义设计决策：不许猜、不自行选默认值；代码留
> `TODO(DECISION-NEEDED): ...` 注释（写清问题与候选方案）；全部汇总到本文件，每条给出：
> **问题 / 候选方案 A/B/C / 推荐 / 影响范围**。」
>
> 本文件与代码中的 `TODO(DECISION-NEEDED)` 编号一一对应。除用户点名的 7 条外，
> 本轮实现新增 4 条（#8–#11），均为实现过程中暴露的、此前未定义的设计空隙。

## 汇总

| # | 一句话 | 涉及代码 | 状态 |
|---|---|---|---|
| 1 | `s_k` 具体是哪些标量？决定 K=96 还是 160 | `src/esmu/subspace.py`（`accumulate_gram` / `_default_scalar_nll_sum`） | 🔴 已留 TODO，占位默认=逐样本 NLL 和 |
| 2 | `∇s_k` 对哪些参数求梯度（全模型 d，还是限定模块） | `src/esmu/subspace.py`（`param_filter` 参数） | 🔴 已留 TODO，默认=None（全模型） |
| 3 | reward 各项的尺度标定方法 | 未来 `src/trainer/esmu`（Stage 5） | 本轮无代码，Stage 5 前必须裁定 |
| 4 | `D_FO` 定义：固定常数还是随搜索动态更新 | 未来 Stage 5 | 同上 |
| 5 | `σ` 初值与 `σ_t` 衰减调度形式 | 未来 Stage 5 | 同上 |
| 6 | `ρ_cross` 实测定义 | 未来 Stage 5 | 同上 |
| 7 | C★「一阶对照」具体实现 | 未来对照组（Stage 4/6） | 同上 |
| 8 | 探针计算时模型模式（train/eval）与随机性 | `src/esmu/subspace.py`（docstring） | 🟡 已留 TODO，函数不强制改模式 |
| 9 | 精确 K×K Gram 与「O(d) 峰值」的数学张力（历史梯度驻留问题） | `src/esmu/subspace.py`（`accumulate_gram`） | 🔴 已留 TODO，当前=历史梯度转存 CPU |
| 10 | 探针 batch 的来源与 K 的确定（forget 集 perturbed/paraphrased 是否各算一条） | `src/esmu/subspace.py`（`probe_batches` 契约） | 🟡 docstring 已明确契约（调用方提供 batch 即 K），编号见清单 |
| 11 | D-γ 依赖 G0 冻结 retain TR JSON（`retain_logs_path`） | `scripts/d_gamma_scan.py` | 🟢 以 CLI 参数暴露，缺省 D_KS=None |

---

## 详细条目

### #1 `s_k` 定义与 K 取值【用户点名】

**问题**：`S = span{∇_θ s_k}` 中，探针标量 `s_k` 到底是什么？候选直接影响 `K`
（子空间维数）与 G1′ 判据（`K_eff ≤ 256`）：

- 候选 A：**逐样本 NLL**（每条探针 = forget 集上一个样本的 NLL）→ `K = n_probe`
- 候选 B：**TR 分量拆分**（forget 集上 perturbed 与 paraphrased 配对样本各算一条
  NLL 或 TR 分子/分母）→ `K = 2·n_probe`，与 truth-ratio 的配对结构对齐
- 候选 C：**组合 reward**（`1 − D_KS/D_FO + …` 的每项分片）→ K 随 reward 项数

**推荐**：A（最简、梯度语义最干净），B 留作 G1′ 扫描时的对照探针集。**待裁定。**

**影响范围**：`accumulate_gram` 的默认 `scalar_fn`、K=96 还是 160、G1′ 探针保真度
曲线的横轴（`n_probe`）。当前代码 `scalar_fn=None` 的占位默认=逐样本 NLL 和
（batch 内多样本聚合），仅保证接口可运行与单测，**不是**方法设计决策。

### #2 `∇s_k` 参数范围【用户点名】

**问题**：对哪些参数求梯度？全模型 `d ≈ 1.23e9`（1B），还是限定模块（如仅
attention / 仅 MLP / 仅末层）？

**候选**：
- A：全模型（`d = 1.23e9`，单条梯度 fp32 ≈ 4.96 GB，单步成本 8.4 s 基线）
- B：仅 attention（`d ≈ 4.4e8`，显存与算力减半）
- C：最后一层 + 输出头（`d` 最小，但可能丢中低层记忆表征）

**推荐**：A（与 09 文档【E】显存账一致：单梯度 4.96 GB ≈ 13 GB/80 GB 账内的最大项）；
B/C 作为 G1′ 的附加扫描轴。**待裁定。**

**影响范围**：显存与单步成本、`param_filter` 注入点的使用方式。当前默认
`param_filter=None`（全模型）。

### #3 reward 尺度标定【用户点名】

**问题**：reward = `1 − D_KS/D_FO + 归一化 AUC 改善 − retain-NLL 护栏`（主方案 §8），
连续项（retain-NLL）与分片常数项（D_KS）量纲不同，直接相加会互相淹没。如何标定
各项尺度？

**候选**：A) 各项除以各自的 D_FO/基线值归一化；B) 各分片先做 rank/z-score 再相加；
C) 分层 reward（先保 D_KS 闸门再优化 retain）。

**推荐**：A（主方案 §8 已隐含 `1 − D_KS/D_FO` 的归一化结构）。**待裁定。**
本轮无实现。

### #4 `D_FO` 定义【用户点名】

**问题**：`D_FO`（reward 分母的参考 KS 距离）是固定常数（如 SimNPO ckpt 的 D）
还是随搜索动态更新（每轮种群后取当前最佳 D）？

**推荐**：固定常数（保证 reward 轨迹单调可比，避免奖励漂移）。**待裁定。**
本轮无实现。

### #5 `σ` 初值与 `σ_t` 衰减【用户点名】

**问题**：`σ_t` 只管跨分片率 `ρ_cross`（主方案 §8 解耦设计），初值与衰减形式未定。

**候选**：A) 线性衰减 `σ_t = σ₀(1 − t/T)`；B) 指数衰减 `σ₀·exp(−t/τ)`；C) 自适应
（按 `ρ_cross` 实测反馈调）。

**推荐**：A 或 B，以 `ρ_cross ∈ [0.2, 0.8]` 为目标反解参数。**待裁定。** 本轮无实现。

### #6 `ρ_cross` 实测定义【用户点名】

**问题**：跨分片率的实测口径：翻转样本对数比例？还是 reward 变化样本比例？

**候选**：A) 相邻两代参数空间中 Σ 样本对 reward 符号翻转的比例；B) 与初代相比的
翻转累积比例。

**推荐**：A。**待裁定。** 本轮无实现。

### #7 C★「一阶对照」实现【用户点名】

**问题**：主方案 §9 对照 C★：reward 分片常数时梯度恒 0（Prop 2′），一阶对什么求导？

**候选**：A) 对积分型分片（1-Wasserstein）求一阶导数做对照；B) 对 reward 的
retain-NLL 连续分量求导。

**推荐**：A（C-W1 已实现 `w1_distance` 指标，梯度通道天然存在）。**待裁定。**
本轮无实现（Stage 4/6 范围）。

### #8 探针计算时的模型模式与随机性【本轮新增】

**问题**：`accumulate_gram` 前向+反传时，模型处于 `train()`（dropout 开启，梯度有
随机性）还是 `eval()`（确定性）？直接影响 G1′ 保真度的可复现性与 K_eff 稳定性。

**候选**：A) 固定 `eval()`（确定性梯度，推荐）；B) 固定 `train()`（随机梯度，更接近
训练分布）；C) 由调用方显式设置（当前实现——函数不强制改模式）。

**推荐**：C 加显式文档，G1′ 扫描时对比 A/B。**待裁定。**

### #9 精确 K×K Gram 与「O(d) 峰值」的数学张力【本轮新增，🔴 最重要】

**问题**：任务 A2 硬约束「禁止同时物化 K 个 d 维向量，峰值 O(d)+O(K²)」与「返回
K×K 精确 Gram `G_ij = ⟨∇s_i, ∇s_j⟩`」在数学上存在张力：**G 的每个非对角元素
`⟨g_i, g_j⟩` 需要两条梯度向量同时在场**，精确计算 K×K Gram 的流式累计必然要保留
历史梯度（`K=96, d=1.23e9, fp32` ≈ 476 GB）。09 文档【E】显存账只算了「单梯度
4.96 GB + Gram 0.04 GB」，未覆盖这一项。

**候选**：
- A) **历史梯度转存 CPU**（当前实现）：GPU 峰值 `O(d)+O(K²)` 严格满足，CPU
  `O(K·d)`（476 GB fp32 / 238 GB bf16）——超出常规 A800 主机内存
- B) **两遍磁盘法**：pass1 逐条前向+反传把 `g_k` 写盘（GPU 峰值 O(d)），pass2
  读回两两内积（峰值 `O(B·d)` 分块）——GPU/内存均达标，磁盘 ≈ 238–476 GB，
  每次调用重算一遍
- C) **近似**：Nyström / 随机投影 / 核近似——内存达标但不精确，改变方法语义
- D) 承认 `O(K·d)` 是精确 Gram 的必要成本，放宽「O(d)」目标为「O(K·d) 且
  GPU 峰值 < 20 GB」（如 K=16 小块时可行）

**推荐**：B（数学精确 + 显存达标），用 bf16 写盘减半；A 作为 K 较小（G1′ 扫描
`n_probe ≤ 32`）时的快捷路径。**待裁定，G1′ 闸门（9/20）前必须定。**

**影响范围**：`accumulate_gram` 实现、09 文档【E】显存账修订、G1′ 扫描的
`n_probe` 上限。

### #10 探针 batch 的来源与 K 的确定【本轮新增】

**问题**：`probe_batches` 从哪个数据集切？K 如何确定？forget 集的 perturbed /
paraphrased 配对是否各算一条探针（与 #1 联动）？

**候选**：A) forget 集 `question_key` 样本随机子集（K = n_probe）；B) perturbed +
paraphrased 配对（K = 2·n_probe）；C) forget + retain 混合（K 更大，需与 #1
候选 C 区分）。

**推荐**：A（K = n_probe，G1′ 扫 `n_probe ∈ {8,16,32,64,128,256}` 时自然定 K_eff）。**待裁定。**

**影响范围**：`accumulate_gram` 调用方（Stage 5 trainer / G1′ 扫描脚本）的数据组装。

### #11 D-γ 依赖 G0 冻结 retain TR JSON【本轮新增，🟢 已按兜底实现】

**问题**：`ks_statistic`（forget_quality_D）的 retain 侧必须从冻结 JSON
（`retain_logs_path`，G0 产物）加载；没有 G0，`D_KS` 无法计算。任务 B 的
D-γ 扫描因此被 G0 阻塞。

**处理（非设计决策，属依赖顺序）**：脚本以 `--retain-logs-path` 暴露该依赖，
缺省时 D_KS 输出 `None`（报告「—」），不臆造数据——与 Stage 1 指标的兜底行为一致。
**待 G0 产出后回填。**

**影响范围**：`reports/D_gamma.md` 的完成度；Stage 2 正式验收前需先跑 G0。

---

## 本轮已落地的「非决策」选择（仅实现约束，不构成设计决策）

- 探针 batch 契约 = 仓库 `DataCollatorForSupervisedDataset` 产出
  （`input_ids` / `attention_mask` / `labels`），见 `_default_scalar_nll_sum`。
- Gram 用 fp32 累计（与单梯度 fp32 的显存账一致）。
- `effective_rank` / `whitening_sqrt_inv` 用对称特征分解（`eigvalsh`/`eigh`），
  `K ≤ 256` 时 O(K³) 可忽略（主方案 §10 G1′ 上限）。
- D-γ 判定阈值 `|ΔD| ≤ 0.05` 为脚本内**初步**规则，需远程数据复核后固化为正式
  判据。
