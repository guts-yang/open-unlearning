# 四方法复现总览

本目录是**远程训练手册**，不是官方 `docs/`。代码在 `workspace/src`，入口在 `workspace/scripts/train.py`。

## Pin

| 方法 | 论文 | 官方代码 | 本仓库接入 |
|------|------|----------|------------|
| BLADE | arXiv:2608.22557 | github.com/tzpranto/blade `@84f7504` | `workspace/src/open_unlearning_plugins/blade/` |
| GROM | arXiv:2608.05783 | github.com/Batorskq/GROM `@6dd9591` | `workspace/src/open_unlearning_plugins/grom/` |
| BalDRO | WWW 2026, arXiv:2601.09172v3, nxZhai/BalDRO `@5a93ede` | 六个 NPO/SimNPO/SatImp 包装 | `workspace/src/open_unlearning_plugins/baldro/` |
| ALTER | AAAI 2026, arXiv:2603.01792 | GitHub 仅空 README `@f664526` | **推定重建** `workspace/src/open_unlearning_plugins/alter/` |

优先级：**BLADE → GROM → BalDRO → ALTER**。GROM 与 BLADE 共用 OpenUnlearning 评测；GROM 与同作者 EvoMU **不是**同一方法。

## 口径（不要混用）

- **BLADE 论文表**：TOFU `HM = hmean(MU, 1−Prob, 1−RG)`。标准 Forget Quality（KS p 值）只作旁注。
- **GROM 论文表**：TOFU Final = 均值(`1−Rouge-L`, `1−Prob`, `1−Extr`, MU)；**不报 FQ p 值**。WMDP Final = ½((1−AccBio)+MMLU)。MUSE 用 Table 5 百分制公式。
- **BalDRO 论文表**：标准 TOFU FQ + MU。
- **ALTER**：WMDP-Bio/Cyber accuracy ↓ + MMLU ↑。原文无完整复现配置，实现为推定。

## 本机已完成 / 远程待做

已完成：插件、Hydra overlay、无权重单测、本手册。

待远程：`01_env_and_weights.md` → BLADE 冒烟/全量停一次 → GROM 五基准停一次 → BalDRO → ALTER。

## 硬约束

- 只用 `workspace/scripts/train.py` 和 `eval.py`（会注册插件并拒绝仓库根 `saves/`）。
- 必须带 `paths.output_dir=workspace/saves/unlearn/<task>`（实验 YAML 已写默认值）。
- 汇总表：`workspace/results/*.md`（先原论文表，再实测）。原始 JSON：`workspace/saves/`。
- 缺跑格子写「未复现」，禁止把论文数字填进实测列。
