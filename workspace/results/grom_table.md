# GROM

论文：*GROM: Gradient-Free Rapid One-Shot Machine Unlearning*（arXiv:2608.05783）。官方代码 pin：`Batorskq/GROM @6dd9591`。闭式编辑，**不是**同作者的 EvoMU。

Seed 0。不报标准 Forget Quality（KS p 值）。TOFU-10% 官方基座是 Llama-3.2-**1B**，与 BLADE 3B 矩阵不对齐。

## 协议（论文）

- TOFU Final = 均值(1−Rouge-L, 1−Prob, 1−Extr, MU)
- WMDP Final = ½((1−AccBio) + MMLU)
- MUSE Final = ½(KnowMem_Dr + ForgetAvg)，ForgetAvg = [(100−VerbMem_Df)+(100−KnowMem_Df)+(100−|PrivLeak|)] / 3
- Time：单卡 H100 上闭式编辑 wall-clock，不含加载与评测

## 原论文结果

### Table 2 · TOFU（节选）

TOFU-5%（LLaMA2-7B-Chat）：

| Method | 1-Rouge-L↑ | 1-Prob↑ | 1-Extr↑ | MU↑ | Final↑ | Time (m)↓ |
| --- | --- | --- | --- | --- | --- | --- |
| Retain | 0.61 | 0.85 | 0.93 | 0.62 | 0.71 | — |
| SimNPO | 0.74 | 0.97 | 0.92 | 0.58 | 0.73 | 2.6 |
| **GROM** | **0.95** | **1.00** | **0.97** | **0.62** | **0.79** | **1.8** |

TOFU-10%（LLaMA3.2-1B-Instruct）：

| Method | 1-Rouge-L↑ | 1-Prob↑ | 1-Extr↑ | MU↑ | Final↑ | Time (m)↓ |
| --- | --- | --- | --- | --- | --- | --- |
| Retain | 0.62 | 0.88 | 0.94 | 0.59 | 0.70 | — |
| SimNPO | 0.65 | 0.94 | 0.94 | 0.56 | 0.70 | 2.5 |
| **GROM** | **0.78** | **1.00** | **0.94** | **0.60** | **0.75** | **0.5** |

全文方法行见原文 Table 2。

### Table 4 · WMDP-Bio（Llama-3-8B-Instruct）

| Method | 1−AccBio↑ | MMLU↑ | Final↑ | Time (m)↓ |
| --- | --- | --- | --- | --- |
| Original | 0.27 | 0.65 | 0.46 | — |
| SimNPO | 0.75 | 0.44 | 0.60 | 21.6 |
| NPO | 0.73 | 0.51 | 0.62 | 23.9 |
| **GROM** | **0.71** | **0.55** | **0.63** | **0.2** |

### Table 5 · MUSE

News（LLaMA2-7B）：

| Method | VerbMem Df↓ | KnowMem Df↓ | PrivLeak→0 | KnowMem Dr↑ | Final↑ | Time (m)↓ |
| --- | --- | --- | --- | --- | --- | --- |
| Retain | 20.75 | 33.32 | 0.00 | 53.79 | 67.88 | — |
| SimNPO | 2.34 | 44.84 | 72.93 | 39.65 | 49.81 | 36.3 |
| **GROM** | **15.68** | **24.01** | **−3.80** | 26.37 | **55.93** | **0.3** |

Books（论文写 ICLM-7B；复现跟 yaml 用 `MUSE-Books_target`）：

| Method | VerbMem Df↓ | KnowMem Df↓ | PrivLeak→0 | KnowMem Dr↑ | Final↑ | Time (m)↓ |
| --- | --- | --- | --- | --- | --- | --- |
| Retain | 14.30 | 28.90 | 0.00 | 74.50 | 80.05 | — |
| SimNPO | 0.00 | 0.00 | −19.82 | 48.27 | 70.83 | 53.8 |
| **GROM** | **2.40** | **36.89** | **−0.22** | **65.56** | **76.20** | **0.3** |

4-bit 攻击见原文 Table 6（可选后续）。ZsRE 本仓库不做。

## 本仓库实测

<!-- MEASURED:START -->
| run | bench | note | 1-Rouge | 1-Prob | 1-Extr | MU | Final | Final_paper | Time_min | Time_paper | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| workspace/saves/unlearn/blade_tofu_1b_05_s42/checkpoint-0/evals | TOFU-05-7B |  | 0.9482675484554387 | 0.9832960905853816 | 0.9582546957861974 | 0.5997517722833354 | 0.8723925267775883 | 0.79 |  | 1.8 | 实测 |
| workspace/saves/unlearn/baldro_npo_dv_tofu01/checkpoint-0/evals | TOFU-10-1B | 官方基座 1B，非 BLADE 3B | 0.050658602150537524 | 0.004886130988597914 | 0.0 | 0.6276969401677462 | 0.17081041832672042 | 0.75 |  | 0.5 | 实测 |
| 未复现 | MUSE-News |  |  |  |  |  |  | 55.93 |  | 0.3 | 未复现 |
| 未复现 | MUSE-Books | 跟 yaml MUSE-Books_target |  |  |  |  |  | 76.20 |  | 0.3 | 未复现 |
| 未复现 | WMDP-Bio | Bio forget 门控 |  |  |  |  |  | 0.63 |  | 0.2 | 未复现 |
| workspace/saves/unlearn/baldro_npo_dv_tofu01/checkpoint-10/evals | TOFU-10-1B |  | 0.8493092245444084 | 0.9555130568565801 | 0.939182097309254 | 0.6064246990123185 | 0.8376072694306402 |  |  |  | 实测 |
| workspace/saves/unlearn/baldro_npo_dv_tofu01/checkpoint-2/evals | TOFU-10-1B |  | 0.602749974640025 | 0.7345294781029225 | 0.8588778083263456 | 0.6146435332705501 | 0.7027001985849608 |  |  |  | 实测 |
| workspace/saves/unlearn/baldro_npo_dv_tofu01/checkpoint-4/evals | TOFU-10-1B |  | 0.7445501286162608 | 0.9344825807842426 | 0.9087324629541542 | 0.5999116937829211 | 0.7969192165343947 |  |  |  | 实测 |
| workspace/saves/unlearn/baldro_npo_dv_tofu01/checkpoint-6/evals | TOFU-10-1B |  | 0.8365920645657168 | 0.9512425572436769 | 0.9324087535080303 | 0.598813519805116 | 0.8297642237806351 |  |  |  | 实测 |
| workspace/saves/unlearn/baldro_npo_dv_tofu01/checkpoint-8/evals | TOFU-10-1B |  | 0.8496918418888176 | 0.9549351412337274 | 0.9376660790026179 | 0.6025626686037175 | 0.8362139326822201 |  |  |  | 实测 |
| workspace/saves/unlearn/baldro_npo_g_tofu01/checkpoint-0/evals | TOFU-10-1B |  | 0.050658602150537524 | 0.004886130988597914 | 0.0 | 0.6276969401677462 | 0.17081041832672042 |  |  |  | 实测 |
| workspace/saves/unlearn/baldro_npo_g_tofu01/checkpoint-10/evals | TOFU-10-1B |  | 0.8072468526716035 | 0.9740122300281655 | 0.9662383900342294 | 0.5858237811836555 | 0.8333303134794134 |  |  |  | 实测 |
| workspace/saves/unlearn/baldro_npo_g_tofu01/checkpoint-2/evals | TOFU-10-1B |  | 0.4773605314890972 | 0.5028969142585993 | 0.7856286961369764 | 0.6207531792496547 | 0.5966598302835819 |  |  |  | 实测 |
| workspace/saves/unlearn/baldro_npo_g_tofu01/checkpoint-4/evals | TOFU-10-1B |  | 0.6839369933700918 | 0.9264606899232604 | 0.9361712827672758 | 0.581535735433889 | 0.7820261753736293 |  |  |  | 实测 |
| workspace/saves/unlearn/baldro_npo_g_tofu01/checkpoint-6/evals | TOFU-10-1B |  | 0.8030161285166217 | 0.9703369839102379 | 0.9654187179030819 | 0.573965826876621 | 0.8281844143016406 |  |  |  | 实测 |
| workspace/saves/unlearn/baldro_npo_g_tofu01/checkpoint-8/evals | TOFU-10-1B |  | 0.8063148569785648 | 0.974593607118004 | 0.9662383900342294 | 0.5801844231957501 | 0.8318328193316371 |  |  |  | 实测 |
| workspace/saves/unlearn/blade_tofu01_smoke/checkpoint-0/evals | TOFU-10-1B |  | 0.13838719360042406 | 0.0981155797839165 | 0.2760314503371496 | 0.5985964083179959 | 0.2777826580098715 |  |  |  | 实测 |
| workspace/saves/unlearn/blade_tofu_1b_01_s123/checkpoint-0/evals | TOFU-10-1B |  | 0.9655786147824705 | 0.9986193228277784 | 0.9709405917660814 | 0.6022361119874987 | 0.8843436603409573 |  |  |  | 实测 |
| workspace/saves/unlearn/blade_tofu_1b_01_s42/checkpoint-0/evals | TOFU-10-1B |  | 0.9618838992423204 | 0.9987060747321493 | 0.9709405917660814 | 0.601882284515455 | 0.8833532125640015 |  |  |  | 实测 |
<!-- MEASURED:END -->
