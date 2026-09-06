# 三方法复现总览

本目录是**远程训练手册**，不是官方 `docs/`。代码在 `workspace/src`，入口在 `workspace/scripts/train.py`。

## Pin

| 方法 | 论文 | 官方代码 | 本仓库接入 |
|------|------|----------|------------|
| BLADE | arXiv:2608.22557 | github.com/tzpranto/blade `@84f7504` | `workspace/src/open_unlearning_plugins/blade/` |
| BalDRO | WWW 2026, arXiv:2601.09172v3, nxZhai/BalDRO `@5a93ede` | 六个 NPO/SimNPO/SatImp 包装 | `workspace/src/open_unlearning_plugins/baldro/` |
| ALTER | AAAI 2026, arXiv:2603.01792 | GitHub 仅空 README `@f664526` | **推定重建** `workspace/src/open_unlearning_plugins/alter/` |

## 口径（不要混用）

- **BLADE 论文表**：TOFU `HM = hmean(MU, 1−Prob, 1−RG)`。标准 Forget Quality（KS p 值）只作旁注。
- **BalDRO 论文表**：标准 TOFU FQ + MU。
- **ALTER**：WMDP-Bio/Cyber accuracy ↓ + MMLU ↑。实现为推定，不能写成「官方数字复现」。

## 本机已完成 / 远程待做

已完成：插件、Hydra overlay、无权重单测、本手册。

待远程：`01_env_and_weights.md` → 冒烟 → BLADE 全量停一次 → BalDRO 停一次 → ALTER。

## 硬约束

- 只用 `workspace/scripts/train.py` 和 `eval.py`（会注册插件并拒绝仓库根 `saves/`）。
- 必须带 `paths.output_dir=workspace/saves/unlearn/<task>`（实验 YAML 已写默认值）。
- 汇总表：`workspace/results/*.csv`。原始 JSON：`workspace/saves/`。
- 缺跑格子写「未复现」，禁止填论文数字。
