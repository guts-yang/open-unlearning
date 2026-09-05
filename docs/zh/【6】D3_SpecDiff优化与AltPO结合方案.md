# D3 · SpecDiff 优化与 AltPO 结合方案 v2.0

> **版本**：v2.0（2026-09-05），按实验评审意见修订
> **v1.0 → v2.0 三处实质修正**：
> ① 撤回「Utility 天花板无现有旋钮」——lr 是活自由度（5e-5→0.74 / 2e-5→0.92 / 1e-5→0.97），lr=2e-5×3seed 升级为 V2 的**零机制改动竞争基线（V0）**；
> ② warmup_steps=0 消融与实现冲突（`src/trainer/unlearn/spec_diff.py` 对 <1 直接 ValueError；warmup 实为 1 步 GradDiff）→ 改为 **warmup-only eval** 与**小 lr warmup** 两个可执行替代；
> ③ V2 从「SpecDiff + NLL(y_a)」（AltPO 弱形式）升级为「**SpecDiff clamp + DPO(y_a, y_f)**」，对齐 AltPO Table 2 的消融分级（DPO > NLL > 仅正反馈）；V1/V2 统一为 DPO 形式。
> **前置阅读**：`D3投机解码分歧遗忘_开题报告.md`、`D3_E0结果解读与下一步.md`

---

## 一、SpecDiff 网格诊断（修正版）

### 1.1 仍然成立的结论

- **κ 与 Priv 非单调**：κ=0.3 是甜蜜点（Priv=0.688、Agg 均值 0.581）；κ=0.5 时 Priv 塌至 0.198（Streisand 效应——推得越深 MIA 越易检出）。**主表钉 κ=0.3**，不跟 OU §F.2 的 HM 选点走。⚠️ 与 `results/specdiff_tofu.md` 当前「选中 κ=0.5」矛盾，写作前必须二选一。
- **λ=1、β=0.1 锁定**：λ∈{0.5,1,2,3} 与 β∈{0,0.1,0.3,0.5} 网格无更好点，不再扩。
- **ep5≈ep10**（Utility 0.747 vs 0.742）：损伤早期饱和，值得追「前几步发生了什么」——warmup-only eval（§四-A）回答的正是这个。

### 1.2 修正结论：Utility 是 lr 的活函数

| lr | N | Mem | Priv | Utility | Agg | SpecGap_r |
|---|---:|---:|---:|---:|---:|---:|
| 5e-5（κ0.3 ep10） | 3 | 0.4257 | 0.6878 | 0.7424 | 0.5809 | ≈0.30 |
| 2e-5（κ0.3 ep10） | 1 | 0.3603 | 0.4500 | 0.9240 | 0.4935 | ≈0.18 |
| 1e-5（κ0.3 ep10） | 1 | 0.3116 | 0.2575 | 0.9703 | 0.3693 | ≈0.10 |

（2e-5/1e-5 行 N=1，需 §四-B 补满 3 seed 后再定案。）

**重新定位融合要解决的真问题**：Utility 单靠 lr 就能抬到 0.92+，**不需要 AltPO**。真正的死结是——**没有任何已测单点同时满足「Mem≥0.45 ∧ SpecGap_r<0.15」**：lr=5e-5 有 Mem 但 r=0.30；lr=1e-5 过 retain 门控但 Mem/Priv/Agg 全崩。lr 旋钮把 Utility 和 SpecGap_r 绑在同一根轴上，**融合的价值主张从「修 Utility」重定位为「在中等遗忘深度下解耦 forget 推力与 retain 连带偏移」**。

---

## 二、AltPO 机制（保留 v1，补 Table 2 分级）

- **正反馈三要素**：M=5 条 zero-shot in-domain 替代答案（T=1.0 自采样）、DPO 对比目标（y_a 正 / y_f 负，β_DPO=0.05）、N/M 训练预算比。
- **Table 2 消融分级**：DPO 对比形式 > NLL 形式 > 仅正反馈（后者忘不掉）；**只有负反馈（≈NPO）→ FU=0.20 乱答**；正负都有 → FQ=0.74 且 MU 无损。这同时是「SpecDiff 缺排水渠」诊断的第三方铁证，也是 V2 必须用 DPO 而非 NLL 的依据。
- **镜像数字**：AltPO selftrain Utility=0.996 / Priv=0.191；SpecDiff κ=0.3 Utility=0.742 / Priv=0.688。**有落点无刹车 ⇄ 有刹车无落点**。

---

## 三、AltSpec 结合设计（v2.0）

### 3.0 统一参照（三重身份）

$$\pi_{\mathrm{ref}}(\mathrm{DPO}) \;=\; \mathrm{draft} \;=\; \pi_{\mathrm{ref}}(\mathrm{KL}) \;=\; \text{冻结 TOFU-full 同一份权重}$$

一个冻结模型同时充当 TV 重合度参照、KL 锚、DPO reference——省一份 deepcopy（1B 约省 2.5GB bf16），且「同源」叙事从两源（audit 同源 + train 同源）收紧为**单源**。

### 3.1 关键性质：DPO 在对称点自启动（新增）

q ≡ π_ref 时两个 log-ratio 均为 0，但梯度非零：

$$\nabla \mathcal{L}_{\mathrm{DPO}}\Big|_{q=\pi_{ref}} = -\frac{\beta_{\mathrm{DPO}}}{2}\Big[\nabla \log q(y_a|x) - \nabla \log q(y_f|x)\Big] \neq 0$$

下降方向 = 抬 y_a、压 y_f——**DPO 自带语义破缺 + 排水，不需要 warm-start**。
**推论：V2 可无 warmup 单相位运行**（trainer 去掉 loss 切换逻辑，冷启动问题在设计层面消失）；V1 的 DPO warmup 保留为对照档。

### 3.2 双推叠加问题（新增，必须显式处理）

clamp(s_f) 与 DPO 负侧**都压 gold 路径**：s_f 饱和（≤1−κ）后 clamp 梯度归零，但 **DPO 继续推** → κ 不再封顶总遗忘压力。处理方案：

- **主张重述**：「overlap 项 κ 有界；DPO 项 ref 锚定（sigmoid 弱饱和）」——不声称全程有界；
- **训练控制改为 SpecGap_f 目标带早停**：每 N 步在 forget 验证切片上测 SpecGap_f，进入 **[0.30, 0.45]** 即停——审计指标兼任训练控制器（叙事红利：「训练与验证同源」落到实处），同时天然防 Streisand 过推。

### 3.3 四档方案

| 档 | 内容 | 角色 |
|---|---|---|
| **V0（新增）** | 纯 SpecDiff，lr=2e-5 / κ=0.3 / λ=1 / β=0.1，补满 3 seed | **零机制改动竞争基线**。若 Utility≥0.90 ∧ SpecGap_r<0.15 ∧ Mem≥0.35 全过 → 主表可写，融合降级附录。预测：SpecGap_r≈0.18 不过门控 → 融合保留必要性 |
| **V1** | DPO warmup（1–2 步，复用 DPO collator）→ 标准 SpecDiff 主循环 | 治「warmup 伤效用」情形（仅当 §四-A 判定损伤在 warmup 时优先） |
| **V2（主推）** | 单相位：$\mathcal{L} = \operatorname{clamp}(s_f,\, 1{-}\kappa) + \lambda\operatorname{clamp}(1{-}s_r,\, \tau) + \beta\,\mathrm{KL}\vert_{D_r\cup gen} + w_a\,\mathcal{L}_{\mathrm{DPO}}(y_a, y_f;\ \beta_{\mathrm{DPO}}{=}0.05)$，无 warmup；SpecGap_f 目标带早停 | w_a∈{0.5, 1.0} × lr∈{2e-5, 5e-5} 各 1 seed 起步 |
| **V3** | AltPO 式交替调度（forget DPO 步 / retain 步） | 备选，仅当 V2 双推叠加失控时启用 |

### 3.4 硬门槛（采纳评审版）

$$\text{机制口径：}\quad \mathrm{SpecGap}_f \ge 0.30 \;\wedge\; \mathrm{SpecGap}_r < 0.15\ (\text{争取 } 0.10)$$
$$\text{行为口径：}\quad \mathrm{Utility} \ge 0.90 \;\wedge\; \mathrm{Mem} \ge 0.45 \;\wedge\; \mathrm{Priv} \ge 0.55$$

**差异化定位 = Mem+Priv 同保 + 可审计干净分离**。不承诺 Agg 超双母体（官方 AltPO Agg=0.583 / SpecDiff κ=0.3 均值 0.581，刷 Agg 空间有限，写了就是 over-claim）。

---

## 四、实验顺序 A–F（采纳评审版，补判据与成本）

| 步 | 内容 | 成本 | 判据 / 否决开关 |
|---|---|---|---|
| **A** | warmup-only eval：只跑 1 步 GradDiff 即停，评 Utility/Mem；（可选）warmup lr=1e-5 再切 5e-5 主循环 | ~1–2 GPU 时 | Utility≈0.74 → 伤在 warmup → V1 优先；Utility≈0.97 → 伤在主循环 → 直接 V2。**替代了不可执行的 warmup=0** |
| **B** | V0：lr=2e-5 κ=0.3 补满 3 seed | ~4–6 GPU 时 | 全过硬门槛 → 主表可写，融合降级附录；SpecGap_r 均值（±CI）>0.15 → 进 C |
| **C** | 用现成 `community/methods/AltPO/generate.py` 产 forget10 替代库（M=5、T=1.0、zero-shot），人工抽查 20 条（相关性/流畅/非拒答） | ~1–2 GPU 时 | 抽查不合格 → 调 prompt/温度重产，不进 D |
| **D** | V1：DPO warmup 1–2 步 → 标准 SpecDiff，1 seed | ~2 GPU 时 | Utility 仍 ≈0.74 → 冷启动语义不是瓶颈，停 V1 转 E |
| **E** | V2 首批：w_a=0.5 × lr∈{2e-5, 5e-5} 各 1 seed | ~6–10 GPU 时 | **kill：SpecGap_r 不降（vs 同 lr 纯 SpecDiff）或 Priv 塌向 κ=0.5 水平（<0.3）→ 停，回退 V0 主线** |
| **F** | V3 交替调度 | 备选 | 仅 E 失控时启用 |

**总预算 ~15–22 GPU 时**，按 10h/周约 4–5 周到融合裁决，在 26 周计划（W7–12 主表窗口）预算内。
实现备注：小 lr warmup→主循环的切换用**两段 run + checkpoint resume** 实现，不改 scheduler 逻辑；warmup 计数按 optimizer step 非 micro-batch。

---

## 五、实现要点（到 E 才改 trainer）

1. `spec_diff.py` 需能吃 `forget.original` / `forget.alternate`——与 DPO 同一 collator（复用 `src/trainer/unlearn/dpo.py` + `src/data/qa.py` 的 `alternate_key`）。
2. **DPO ref 与 draft 共用同一份冻结 TOFU-full**，不 deepcopy 第二份 1B。
3. 不设 `warmup_steps=0`（除非改成可微破缺，如对 q 加噪——不在本期范围）。
4. **FU/CI 评测随 V2 首批接入**（DistilBERT gibberish 分类器零额外训练成本）——现有网格完全没测乱码/不一致维度，Priv 低分很可能藏在这里。
5. SpecGap_f 目标带早停：复用 E0 探针代码做训练内嵌 eval hook（每 200 步 forget 验证切片，n≈50 够 CI 读数）。

---

## 六、写作纪律（采纳评审版）

- 引 AltPO Table 2 论证纯负反馈伤 FU：✓ 可以。
- 新颖性**只**落在「有界 TV + 同源审计 + SpecGap 目标带训练控制」；y_a 流程、DPO 形式、N/M 预算全部归 AltPO。
- κ=0.5 仅作失效模式展示（Streisand），**主表不再用它当 HM 选中点**。
- 未测 FU/CI 前，不写「不再乱答」。
- 「SpecGap 只测 gold 路径、审计口径不被 y_a 训练污染」**保留**；但删除「语义更干净」承诺——y_a 会改全词表分布，Utility/Priv/SpecGap_r 都会动，不能事先保证。
- 与 `results/specdiff_tofu.md` 的 κ 选中点矛盾在写作前必须二选一（建议：主表 κ=0.3，κ=0.5 进失效模式节）。

---

## 修订记录

| 版本 | 日期 | 变更 |
|---|---|---|
| v1.0 | 2026-09-05 | 初版：κ 诊断、AltPO 镜像互补、V1/V2/V3（V2 为 NLL 弱形式）、warmup=0 消融 |
| v2.0 | 2026-09-05 | 按评审修订：恢复 lr 自由度（新增 V0 基线）；warmup=0 → warmup-only eval；V2 升级 DPO 对比形式；新增 §3.1 DPO 自启动性质、§3.2 双推叠加与 SpecGap 目标带早停；硬门槛加 SpecGap_r<0.15；差异化定位去 Agg 承诺 |
