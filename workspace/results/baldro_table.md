# BalDRO

论文：*BalDRO: A Distributionally Robust Optimization based Framework for Large Language Model Unlearning*（WWW ’26，DOI 10.1145/3774904.3792975，arXiv:2601.09172）。原文：[workspace/docs/papers/【168】BalDRO-A Distributionally Robust Optimization based Framework for Large Language Model Unlearning.pdf](../docs/papers/【168】BalDRO-A Distributionally Robust Optimization based Framework for Large Language Model Unlearning.pdf)。官方代码 pin：`nxZhai/BalDRO @5a93ede`。

## 协议（论文）

- 主骨干：Llama-2-7B；8×A800。TOFU forget 1% / 5% / 10%；MUSE 固定 Books / News split。
- TOFU 主指标：**FQ**（相对 retain-only 的 Truth Ratio KS）、**MU**（retain/holdout 上 Probability、ROUGE、Truth Ratio 的调和平均）。
- 两个插件：BalDRO-G（batch 内高 loss 子集，文中 top-50%）、BalDRO-DV（Donsker–Varadhan log-sum-exp）。DV 峰值常见 **β=2.0、λ=1.0**（Figure 4）；只对 forget loss 做 DRO，retain 不加 DRO（Figure 5）。
- 超参搜索：lr ∈ {1e-5, 2e-5, 5e-5, 1e-4}，batch ∈ {8, 16, 32}，β ∈ {1, 2, 5, 10}，λ ∈ {0.25, 0.5, 1, 2}。
- forget 5%/10% 的完整表在 Figure 3，正文只给出部分 FQ；10% 的逐点数字原文未列表，实测勿填臆造值。

## 原论文结果

### Table 1 · TOFU forget ratio = 1%

| Method | FQ↑ | MU↑ | Fluency↑ | EM↓ | ES↓ | F-TR↑ | Ra-TR↑ | R-TR↑ | Rw-TR↑ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Original | 0.0013 | 0.6276 | 0.8889 | 1.0000 | 1.0000 | 0.5306 | 0.6120 | 0.4596 | 0.5521 |
| Retain | 1.0000 | 0.6268 | 0.9222 | 0.7121 | 0.0719 | 0.6777 | 0.6087 | 0.4621 | 0.5612 |
| GradAscent | 0.2657 | 0.5313 | 0.5326 | 0.5572 | 0.0361 | 0.0019 | 0.6049 | 0.4357 | 0.5848 |
| GradDiff | 0.5786 | 0.6064 | 0.3976 | 0.5397 | 0.0481 | 0.6359 | 0.5595 | 0.4523 | 0.5720 |
| NPO | 0.7659 | 0.5775 | 0.8158 | 0.6842 | 0.0982 | 0.7125 | 0.5807 | 0.4360 | 0.5489 |
| NPO+BalDRO-G | 0.9188 | 0.6126 | 0.8220 | 0.7634 | 0.0630 | 0.6859 | 0.6187 | 0.4495 | 0.5741 |
| **NPO+BalDRO-DV** | **0.9900** | **0.5815** | **0.8227** | **0.6659** | **0.0593** | **0.7238** | **0.6000** | **0.4148** | **0.5610** |
| SimNPO | 0.4046 | 0.5643 | 0.8422 | 0.7383 | 0.0850 | 0.6768 | 0.6025 | 0.3990 | 0.5377 |
| SimNPO+BalDRO-G | 0.5786 | 0.5651 | 0.8894 | 0.7189 | 0.0633 | 0.7034 | 0.5877 | 0.4307 | 0.5712 |
| SimNPO+BalDRO-DV | 0.5786 | 0.5917 | 0.8479 | 0.6926 | 0.0521 | 0.7257 | 0.6345 | 0.4215 | 0.5726 |
| SatImp | 0.0013 | 0.5342 | 0.8272 | 0.9466 | 0.2041 | 0.5117 | 0.5493 | 0.4315 | 0.5338 |
| SatImp+BalDRO-G | 0.0971 | 0.6003 | 0.8520 | 0.8879 | 0.1609 | 0.6108 | 0.6131 | 0.4654 | 0.5480 |
| SatImp+BalDRO-DV | 0.0068 | 0.5480 | 0.7588 | 0.8646 | 0.4522 | 0.5515 | 0.5764 | 0.4485 | 0.4973 |

### Figure 3 正文摘录 · TOFU forget 5%（FQ）

| Method | FQ↑ |
| --- | --- |
| NPO | 0.6284 |
| NPO+BalDRO-G | 0.7125 |
| NPO+BalDRO-DV | 0.9646 |
| SimNPO | 0.4662 |
| SimNPO+BalDRO-G | 0.7125 |
| SimNPO+BalDRO-DV | 0.8655 |

对应 MU 原文写「与原模型相当」，无逐点表。forget 10% 仅有定性描述。

### Table 2 · MUSE

| Method | News KM-Dr↑ | KM-Df↓ | VM-Df↓ | PL→0 | Books KM-Dr↑ | KM-Df↓ | VM-Df↓ | PL→0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Original | 0.5552 | 0.6443 | 0.5789 | −99.8111 | 0.6913 | 0.4712 | 0.9970 | −57.3410 |
| Retain | 0.5602 | 0.3279 | 0.2016 | −4.7200 | 0.6874 | 0.3029 | 0.1445 | 8.1600 |
| GradAscent | 0.0000 | 0.0000 | 0.0000 | 25.1259 | 0.0000 | 0.0000 | 0.0000 | −22.8180 |
| GradDiff | 0.2519 | 0.2938 | 0.0029 | 108.9840 | 0.0678 | 0.0089 | 0.0035 | −37.0562 |
| NPO | 0.4552 | 0.5978 | 0.4255 | −90.8480 | 0.6424 | 0.4414 | 0.6011 | −55.7692 |
| NPO+BalDRO-G | 0.4626 | 0.5805 | 0.3826 | −65.7011 | 0.6486 | 0.4376 | 0.5465 | −54.2160 |
| NPO+BalDRO-DV | 0.4589 | 0.5754 | 0.3934 | −69.5214 | 0.6520 | 0.4107 | 0.5230 | −55.2515 |
| SimNPO | 0.4121 | 0.5806 | 0.3829 | −99.8951 | 0.5969 | 0.3009 | 0.2364 | −51.7018 |
| SimNPO+BalDRO-G | 0.4272 | 0.5940 | 0.4193 | −99.8950 | 0.6151 | 0.2841 | 0.2264 | −51.2944 |
| SimNPO+BalDRO-DV | 0.4571 | 0.5670 | 0.1829 | 100.4139 | 0.5393 | 0.3009 | 0.1935 | −49.3898 |
| SatImp | 0.3797 | 0.5902 | 0.4403 | −99.8951 | 0.6026 | 0.4017 | 0.8730 | −58.3395 |
| SatImp+BalDRO-G | 0.3805 | 0.5053 | 0.4197 | −99.8531 | 0.6037 | 0.3802 | 0.4955 | −54.5858 |
| SatImp+BalDRO-DV | 0.3967 | 0.4568 | 0.3552 | −99.8321 | 0.6013 | 0.3672 | 0.5535 | −57.2485 |

### Table 3 · TOFU 1% 扩展 MIA（↓）

| Method | LOSS | ZLib | MinK | MinK++ |
| --- | --- | --- | --- | --- |
| Original | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Retain | 0.4981 | 0.5513 | 0.5038 | 0.6100 |
| NPO | 0.4681 | 0.4594 | 0.4912 | 0.4900 |
| NPO+BalDRO-G | 0.4632 | 0.3784 | 0.5056 | 0.3266 |
| NPO+BalDRO-DV | 0.3481 | 0.3250 | 0.3613 | 0.4875 |
| SimNPO | 0.4925 | 0.4700 | 0.4981 | 0.2738 |
| SimNPO+BalDRO-G | 0.2468 | 0.2188 | 0.2344 | 0.1314 |
| SimNPO+BalDRO-DV | 0.1769 | 0.2525 | 0.1744 | 0.1075 |
| SatImp | 0.9956 | 0.9906 | 0.9900 | 0.9544 |
| SatImp+BalDRO-G | 0.9493 | 0.9451 | 0.9496 | 0.8004 |
| SatImp+BalDRO-DV | 0.9613 | 0.9587 | 0.9613 | 0.8493 |

优先对照 Table 1 的 NPO / NPO+G / NPO+DV（forget01），以及正文给出的 forget05 NPO FQ。Seed **0**。`DrNPO.yaml` 默认 `beta_dv_forget=2.0`。

## 本仓库实测

<!-- MEASURED:START -->
| run | split | method | FQ | MU | FQ_paper | MU_paper | status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 未复现 | TOFU-01 | NPO |  |  | 0.7659 | 0.5775 | 未复现 |
| workspace/saves/unlearn/baldro_npo_g_tofu01/checkpoint-0/evals | TOFU-01 | NPO+G |  | 0.6276969401677462 | 0.9188 | 0.6126 | 实测 |
| workspace/saves/unlearn/baldro_npo_dv_tofu01/checkpoint-0/evals | TOFU-01 | NPO+DV |  | 0.6276969401677462 | 0.9900 | 0.5815 | 实测 |
| 未复现 | TOFU-05 | NPO |  |  | 0.6284 |  | 未复现 |
| 未复现 | TOFU-05 | NPO+DV |  |  | 0.9646 |  | 未复现 |
| workspace/saves/unlearn/baldro_simnpo_dv_tofu01/checkpoint-0/evals | TOFU-01 | SimNPO+DV |  | 0.6276969401677462 | 0.5786 | 0.5917 | 实测 |
| 未复现 | TOFU-01 | SimNPO+G |  |  | 0.5786 | 0.5651 | 未复现 |
| workspace/saves/unlearn/baldro_npo_dv_tofu01/checkpoint-10/evals | TOFU-01 | NPO+DV | 0.7659314523482239 | 0.6064246990123185 |  |  | 实测 |
| workspace/saves/unlearn/baldro_npo_dv_tofu01/checkpoint-2/evals | TOFU-01 | NPO+DV |  | 0.6146435332705501 |  |  | 实测 |
| workspace/saves/unlearn/baldro_npo_dv_tofu01/checkpoint-4/evals | TOFU-01 | NPO+DV |  | 0.5999116937829211 |  |  | 实测 |
| workspace/saves/unlearn/baldro_npo_dv_tofu01/checkpoint-6/evals | TOFU-01 | NPO+DV |  | 0.598813519805116 |  |  | 实测 |
| workspace/saves/unlearn/baldro_npo_dv_tofu01/checkpoint-8/evals | TOFU-01 | NPO+DV |  | 0.6025626686037175 |  |  | 实测 |
| workspace/saves/unlearn/baldro_npo_g_tofu01/checkpoint-10/evals | TOFU-01 | NPO+G | 0.5786001416508443 | 0.5858237811836555 |  |  | 实测 |
| workspace/saves/unlearn/baldro_npo_g_tofu01/checkpoint-2/evals | TOFU-01 | NPO+G |  | 0.6207531792496547 |  |  | 实测 |
| workspace/saves/unlearn/baldro_npo_g_tofu01/checkpoint-4/evals | TOFU-01 | NPO+G |  | 0.581535735433889 |  |  | 实测 |
| workspace/saves/unlearn/baldro_npo_g_tofu01/checkpoint-6/evals | TOFU-01 | NPO+G |  | 0.573965826876621 |  |  | 实测 |
| workspace/saves/unlearn/baldro_npo_g_tofu01/checkpoint-8/evals | TOFU-01 | NPO+G |  | 0.5801844231957501 |  |  | 实测 |
| workspace/saves/unlearn/baldro_simnpo_dv_tofu01/checkpoint-2/evals | TOFU-01 | SimNPO+DV |  | 0.6130731253049051 |  |  | 实测 |
| workspace/saves/unlearn/baldro_simnpo_dv_tofu01/checkpoint-4/evals | TOFU-01 | SimNPO+DV |  |  |  |  | 实测 |
<!-- MEASURED:END -->
