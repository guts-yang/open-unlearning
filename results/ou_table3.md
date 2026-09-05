# OU Table 3 复现汇总（TOFU forget10 · Llama-3.2-1B-Instruct）

- 生成时间：2026-09-03T10:33:13+08:00
- 评估器 HEAD：`1466baac849f90d3ade525e18af3136ecc22637c`
- 数据来源：`results/ou_table3_runs.jsonl`（124 条，source=all）
- 口径：Mem=HM(1−ES,1−EM,1−ParaProb,1−TR)；Priv=HM(s_LOSS,s_ZLib,s_MinK,s_MinK++)；Utility=HM(MU,Fluency)；Agg=HM(Mem,Priv,Utility)；全部按 init-finetuned 归一化。负分量（常见 1−TR）clamp 到 0 且不参与 HM，带 ※
- 命中判定：四维全部落在论文靶值 ±0.02 内

## 〇、官方 ckpt 每方法最佳（按 Agg，写入本表）

| 方法 | 超参串 | Agg↑ | Mem↑ | Priv↑ | Utility↑ | 命中 |
|---|---|---|---|---|---|---|
| SimNPO | `lr5e-05_b3.5_a1_d1_g0.125_ep10` | 0.4716 | 0.3020 | 0.4880 | 0.9992 | ❌ |
| RMU | `lr5e-05_layer5_scoeff1_epoch5` | 0.4987 | 0.3874 | 0.4856 | 0.7271 | ❌ |
| UNDIAL | `lr0.0003_beta30_alpha2_epoch5` | 0.2746 | 0.2283 | 0.2877 | 0.3256 | — |
| AltPO | `lr2e-05_beta0.05_alpha1_epoch10` | 0.5826 | 0.4284 | 0.5724 | 0.9364 | — |
| NPO | — | — | — | — | — | 尚未评 |
| GradDiff | — | — | — | — | — | 尚未评 |
| IdkDPO | — | — | — | — | — | 尚未评 |
| IdkNLL | — | — | — | — | — | 尚未评 |
| SpecDiff | — | — | — | — | — | 无官方 ckpt |


## 〇（本机）、自训每方法最佳（按 Agg，每方法只留最高一组）

| 方法 | 超参串 | 运行名 | Agg↑ | Mem↑ | Priv↑ | Utility↑ | 命中 |
|---|---|---|---|---|---|---|---|
| SpecDiff | `lr5e-05_lam1_b0.1_k0.3_t0.02_ep10` | `tofu_1B_SpecDiff_forget10_seed1` | 0.5992 | 0.4148 | 0.7794 | 0.7617 | — |
| NPO | `lr1e-05_beta0.1_alpha1_epoch10` | `tofu_1B_NPO_forget10_N1` | 0.5122 | 0.3120 | 0.6165 | 0.9700 | — |
| SimNPO | `lr5e-05_b3.5_a1_d1_g0.125_ep10` | `tofu_1B_SimNPO_forget10_S_p1best` | 0.5085 | 0.3339 | 0.5654 | 0.8805 | ❌ |
| AltPO | `lr2e-05_beta0.05_alpha1_epoch10` | `tofu_1B_AltPO_forget10_A1` | 0.3664 | 0.5103 | 0.1914 | 0.9958 | — |
| GradDiff | `lr1e-05_alpha10_epoch10` ※ | `tofu_1B_GradDiff_forget10_GD2` | 0.3519 | 0.9381 | 0.1623 | 0.7714 | — |
| UNDIAL | `lr0.0001_beta10_alpha1_epoch10` | `tofu_1B_UNDIAL_forget10_U1` | 0.2350 | 0.1219 | 0.2955 | 0.8451 | — |
| RMU | `lr5e-05_layer5_scoeff1_epoch5` | `tofu_1B_RMU_forget10_R_best` | 0.2302 | 0.6043 | 0.1874 | 0.1655 | ❌ |
| TPO | `lr1e-5_beta0.19_alpha0_gpt_ep10` | `tofu_1B_TPO_forget10_TPO1` | 0.0007 | 0.3994 | 0.3086 | 0.0002 | — |
| GradAscent | `default_lr1e-05_ep10` ※ | `tofu_1B_GradAscent_forget10_GA1` | 0.0000 | 0.9724 | 0.4809 | 0.0000 | — |


<!-- specdiff-ou-table-begin -->
## 〇、SpecDiff（本机自训，HM(Mem,Utility) 选点）

写入 `results/ou_table3_runs.jsonl`（source=selftrain）。网格未完成时此表会随 `specdiff_table.py` 刷新。

**当前 HM 最高超参：** `lr5e-05_lam1_b0.1_k0.5_t0.02_ep10`（HM=0.6237，Agg=0.3635）。
**当前 Agg 最高单次：** `lr5e-05_lam1_b0.1_k0.3_t0.02_ep10` seed=1（Agg=0.5992，Mem=0.4148，Priv=0.7794，Utility=0.7617）。

| 超参串 | N | Mem | Priv | Utility | Agg | HM(Mem,Utility) |
|---|---:|---:|---:|---:|---:|---:|
| `lr5e-05_lam1_b0.1_k0.5_t0.02_ep10` ← 选中 | 3 | 0.5491 | 0.1984 | 0.7218 | 0.3635 | 0.6237 |
| `lr5e-05_lam1_b0_k0.3_t0.02_ep10` | 3 | 0.4403 | 0.5809 | 0.7131 | 0.5437 | 0.5444 |
| `lr5e-05_lam1_b0.1_k0.3_t0.02_ep10` | 3 | 0.4257 | 0.6878 | 0.7424 | 0.5809 | 0.5410 |
| `lr5e-05_lam1_b0.1_k0.3_t0.02_ep5` | 3 | 0.4215 | 0.6507 | 0.7472 | 0.5716 | 0.5389 |
| `lr5e-05_lam0.5_b0.1_k0.3_t0.02_ep10` | 1 | 0.4215 | 0.7368 | 0.7310 | 0.5885 | 0.5347 |
| `lr2e-05_lam1_b0.1_k0.3_t0.02_ep10` | 3 | 0.3664 | 0.5491 | 0.9207 | 0.5299 | 0.5240 |
| `lr5e-05_lam1_b0.3_k0.3_t0.02_ep10` | 1 | 0.3990 | 0.5883 | 0.7312 | 0.5383 | 0.5163 |
| `lr5e-05_lam2_b0.1_k0.3_t0.02_ep10` | 1 | 0.4031 | 0.5873 | 0.7120 | 0.5369 | 0.5148 |
| `lr2e-05_lam1_b0.1_k0.3_t0.02_ep5` | 1 | 0.3531 | 0.4495 | 0.9086 | 0.4872 | 0.5086 |
| `lr5e-05_lam1_b0.1_k0.2_t0.02_ep10` | 1 | 0.3645 | 0.3275 | 0.7320 | 0.4188 | 0.4867 |
| `lr5e-05_lam3_b0.5_k0.3_t0.02_ep10` | 3 | 0.3510 | 0.3194 | 0.7355 | 0.4079 | 0.4750 |
| `lr1e-05_lam1_b0.1_k0.3_t0.02_ep10` | 1 | 0.3116 | 0.2575 | 0.9703 | 0.3693 | 0.4718 |
| `lr1e-05_lam1_b0.1_k0.3_t0.02_ep5` | 1 | 0.2823 | 0.2188 | 0.9700 | 0.3281 | 0.4373 |
| `lr5e-05_lam1_b0.1_k0.3_t0.02_ep10_wup2_wudpo` | 1 | 0.2769 | 0.1857 | 0.9776 | 0.2994 | 0.4316 |
<!-- specdiff-ou-table-end -->

## 一、各方法 Top-K（按 Agg 降序）

### AltPO（3 个 ckpt）

| 来源 | 超参串 | Mem | Priv | Utility | Agg | HM(Mem,Utility) | 命中 | 与靶值偏差 |
|---|---|---|---|---|---|---|---|---|
| official | `lr2e-05_beta0.05_alpha1_epoch10` | 0.4284 | 0.5724 | 0.9364 | 0.5826 | 0.5879 | — | |
| selftrain | `lr2e-05_beta0.05_alpha1_epoch10` | 0.5103 | 0.1914 | 0.9958 | 0.3664 | 0.6748 | — | |
| official | `lr1e-05_beta0.05_alpha1_epoch10` | 0.3038 | 0.1611 | 0.5639 | 0.2661 | 0.3948 | — | |

（论文未给该方法靶值，不做命中判定）

### GradAscent（1 个 ckpt）

| 来源 | 超参串 | Mem | Priv | Utility | Agg | HM(Mem,Utility) | 命中 | 与靶值偏差 |
|---|---|---|---|---|---|---|---|---|
| selftrain | `default_lr1e-05_ep10` ※ | 0.9724 | 0.4809 | 0.0000 | 0.0000 | 0.0000 | — | |

（论文未给该方法靶值，不做命中判定）

### GradDiff（2 个 ckpt）

| 来源 | 超参串 | Mem | Priv | Utility | Agg | HM(Mem,Utility) | 命中 | 与靶值偏差 |
|---|---|---|---|---|---|---|---|---|
| selftrain | `lr1e-05_alpha10_epoch10` ※ | 0.9381 | 0.1623 | 0.7714 | 0.3519 | 0.8466 | — | |
| selftrain | `lr1e-05_alpha5_epoch10` ※ | 0.9461 | 0.1612 | 0.7605 | 0.3499 | 0.8432 | — | |

（论文未给该方法靶值，不做命中判定）

### NPO（1 个 ckpt）

| 来源 | 超参串 | Mem | Priv | Utility | Agg | HM(Mem,Utility) | 命中 | 与靶值偏差 |
|---|---|---|---|---|---|---|---|---|
| selftrain | `lr1e-05_beta0.1_alpha1_epoch10` | 0.3120 | 0.6165 | 0.9700 | 0.5122 | 0.4722 | — | |

（论文未给该方法靶值，不做命中判定）

### RMU（56 个 ckpt，靶值 Agg=0.52 / Mem=0.47 / Priv=0.50 / Utility=0.61）

| 来源 | 超参串 | Mem | Priv | Utility | Agg | HM(Mem,Utility) | 命中 | 与靶值偏差 |
|---|---|---|---|---|---|---|---|---|
| official | `lr5e-05_layer5_scoeff1_epoch5` | 0.3874 | 0.4856 | 0.7271 | 0.4987 | 0.5055 | ❌ | Agg-0.02 Mem-0.08 Priv-0.01 Utility+0.12 |
| official | `lr2e-05_layer5_scoeff10_epoch10` | 0.4153 | 0.5370 | 0.4484 | 0.4615 | 0.4312 | ❌ | Agg-0.06 Mem-0.05 Priv+0.04 Utility-0.16 |
| official | `lr2e-05_layer5_scoeff10_epoch5` | 0.3827 | 0.4074 | 0.5107 | 0.4270 | 0.4375 | ❌ | Agg-0.09 Mem-0.09 Priv-0.09 Utility-0.10 |
| official | `lr5e-05_layer5_scoeff1_epoch10` | 0.4792 | 0.2947 | 0.6042 | 0.4205 | 0.5345 | ❌ | Agg-0.10 Mem+0.01 Priv-0.21 Utility-0.01 |
| official | `lr2e-05_layer10_scoeff10_epoch10` | 0.2933 | 0.1683 | 0.4164 | 0.2553 | 0.3442 | ❌ | Agg-0.26 Mem-0.18 Priv-0.33 Utility-0.19 |
| selftrain | `lr5e-05_layer5_scoeff1_epoch5` | 0.6043 | 0.1874 | 0.1655 | 0.2302 | 0.2598 | ❌ | Agg-0.29 Mem+0.13 Priv-0.31 Utility-0.44 |
| official | `lr5e-05_layer10_scoeff1_epoch10` | 0.2288 | 0.1352 | 0.7380 | 0.2286 | 0.3493 | ❌ | Agg-0.29 Mem-0.24 Priv-0.36 Utility+0.13 |
| official | `lr2e-05_layer10_scoeff100_epoch5` | 0.2298 | 0.1554 | 0.3196 | 0.2156 | 0.2674 | ❌ | Agg-0.30 Mem-0.24 Priv-0.34 Utility-0.29 |
| official | `lr5e-05_layer10_scoeff1_epoch5` | 0.1989 | 0.1207 | 0.7787 | 0.2055 | 0.3169 | ❌ | Agg-0.31 Mem-0.27 Priv-0.38 Utility+0.17 |
| official | `lr2e-05_layer10_scoeff10_epoch5` | 0.1954 | 0.1260 | 0.6410 | 0.2053 | 0.2995 | ❌ | Agg-0.31 Mem-0.27 Priv-0.37 Utility+0.03 |

命中靶值的 ckpt 数：**0 / 56**

### SimNPO（50 个 ckpt，靶值 Agg=0.53 / Mem=0.32 / Priv=0.63 / Utility=1.00）

| 来源 | 超参串 | Mem | Priv | Utility | Agg | HM(Mem,Utility) | 命中 | 与靶值偏差 |
|---|---|---|---|---|---|---|---|---|
| selftrain | `lr5e-05_b3.5_a1_d1_g0.125_ep10` | 0.3339 | 0.5654 | 0.8805 | 0.5085 | 0.4841 | ❌ | Agg-0.02 Mem+0.01 Priv-0.06 Utility-0.12 |
| official | `lr5e-05_b3.5_a1_d1_g0.125_ep10` | 0.3020 | 0.4880 | 0.9992 | 0.4716 | 0.4638 | ❌ | Agg-0.06 Mem-0.02 Priv-0.14 Utility-0.00 |
| official | `lr5e-05_b4.5_a1_d1_g0.25_ep5` | 0.3388 | 0.3606 | 0.9912 | 0.4456 | 0.5050 | ❌ | Agg-0.08 Mem+0.02 Priv-0.27 Utility-0.01 |
| official | `lr5e-05_b3.5_a1_d1_g0.125_ep5` | 0.2941 | 0.4168 | 0.9874 | 0.4404 | 0.4532 | ❌ | Agg-0.09 Mem-0.03 Priv-0.21 Utility-0.01 |
| official | `lr5e-05_b4.5_a1_d1_g0.25_ep10` | 0.3395 | 0.3228 | 1.0115 | 0.4266 | 0.5083 | ❌ | Agg-0.10 Mem+0.02 Priv-0.31 Utility+0.01 |
| official | `lr5e-05_b4.5_a1_d1_g0.125_ep10` | 0.2870 | 0.3770 | 0.9997 | 0.4203 | 0.4460 | ❌ | Agg-0.11 Mem-0.03 Priv-0.25 Utility-0.00 |
| official | `lr5e-05_b4.5_a1_d1_g0.125_ep5` | 0.2826 | 0.3425 | 0.9907 | 0.4017 | 0.4398 | ❌ | Agg-0.13 Mem-0.04 Priv-0.29 Utility-0.01 |
| official | `lr5e-05_b3.5_a1_d1_g0.25_ep5` | 0.3574 | 0.2529 | 0.9954 | 0.3868 | 0.5260 | ❌ | Agg-0.14 Mem+0.04 Priv-0.38 Utility-0.00 |
| official | `lr5e-05_b3.5_a1_d1_g0.25_ep10` | 0.3608 | 0.2303 | 1.0088 | 0.3701 | 0.5315 | ❌ | Agg-0.16 Mem+0.04 Priv-0.40 Utility+0.01 |
| official | `lr2e-05_b3.5_a1_d1_g0.25_ep10` | 0.1708 | 0.3139 | 0.9865 | 0.2984 | 0.2912 | ❌ | Agg-0.23 Mem-0.15 Priv-0.32 Utility-0.01 |

命中靶值的 ckpt 数：**0 / 50**

### TPO（1 个 ckpt）

| 来源 | 超参串 | Mem | Priv | Utility | Agg | HM(Mem,Utility) | 命中 | 与靶值偏差 |
|---|---|---|---|---|---|---|---|---|
| selftrain | `lr1e-5_beta0.19_alpha0_gpt_ep10` | 0.3994 | 0.3086 | 0.0002 | 0.0007 | 0.0004 | — | |

（论文未给该方法靶值，不做命中判定）

### UNDIAL（10 个 ckpt）

| 来源 | 超参串 | Mem | Priv | Utility | Agg | HM(Mem,Utility) | 命中 | 与靶值偏差 |
|---|---|---|---|---|---|---|---|---|
| official | `lr0.0003_beta30_alpha2_epoch5` | 0.2283 | 0.2877 | 0.3256 | 0.2746 | 0.2684 | — | |
| selftrain | `lr0.0001_beta10_alpha1_epoch10` | 0.1219 | 0.2955 | 0.8451 | 0.2350 | 0.2131 | — | |
| official | `lr0.0001_beta10_alpha5_epoch5` ※ | 0.2078 | 0.1433 | 0.8964 | 0.2325 | 0.3374 | — | |
| official | `lr0.0003_beta10_alpha2_epoch10` | 0.1595 | 0.2036 | 0.4488 | 0.2238 | 0.2354 | — | |
| selftrain | `lr0.0003_beta30_alpha2_epoch5` | 0.3130 | 0.3183 | 0.0866 | 0.1678 | 0.1357 | — | |
| official | `lr0.0001_beta3_alpha1_epoch10` ※ | 0.0693 | 0.1038 | 0.9355 | 0.1194 | 0.1291 | — | |
| official | `lr1e-05_beta30_alpha2_epoch10` ※ | 0.0235 | 0.1010 | 1.0051 | 0.0562 | 0.0460 | — | |
| official | `lr1e-05_beta10_alpha1_epoch10` ※ | 0.0162 | 0.1003 | 1.0114 | 0.0413 | 0.0319 | — | |
| official | `lr1e-05_beta3_alpha2_epoch5` | 0.0016 | 0.1000 | 1.0061 | 0.0048 | 0.0033 | — | |
| official | `lr0.0003_beta3_alpha5_epoch5` | 0.1305 | 0.1293 | 0.0000 | 0.0000 | 0.0000 | — | |

（论文未给该方法靶值，不做命中判定）

## 附：SimNPO 全量 50 点（按 Agg 降序，用于反推 model selection）

| # | 来源 | 超参串 | Mem | Priv | Utility | Agg | HM(Mem,Utility) | 命中 | 与靶值偏差 |
|---|---|---|---|---|---|---|---|---|---|
| 1  selftrain | `lr5e-05_b3.5_a1_d1_g0.125_ep10` | 0.3339 | 0.5654 | 0.8805 | 0.5085 | 0.4841 | ❌ | Agg-0.02 Mem+0.01 Priv-0.06 Utility-0.12 |
| 2  official | `lr5e-05_b3.5_a1_d1_g0.125_ep10` | 0.3020 | 0.4880 | 0.9992 | 0.4716 | 0.4638 | ❌ | Agg-0.06 Mem-0.02 Priv-0.14 Utility-0.00 |
| 3  official | `lr5e-05_b4.5_a1_d1_g0.25_ep5` | 0.3388 | 0.3606 | 0.9912 | 0.4456 | 0.5050 | ❌ | Agg-0.08 Mem+0.02 Priv-0.27 Utility-0.01 |
| 4  official | `lr5e-05_b3.5_a1_d1_g0.125_ep5` | 0.2941 | 0.4168 | 0.9874 | 0.4404 | 0.4532 | ❌ | Agg-0.09 Mem-0.03 Priv-0.21 Utility-0.01 |
| 5  official | `lr5e-05_b4.5_a1_d1_g0.25_ep10` | 0.3395 | 0.3228 | 1.0115 | 0.4266 | 0.5083 | ❌ | Agg-0.10 Mem+0.02 Priv-0.31 Utility+0.01 |
| 6  official | `lr5e-05_b4.5_a1_d1_g0.125_ep10` | 0.2870 | 0.3770 | 0.9997 | 0.4203 | 0.4460 | ❌ | Agg-0.11 Mem-0.03 Priv-0.25 Utility-0.00 |
| 7  official | `lr5e-05_b4.5_a1_d1_g0.125_ep5` | 0.2826 | 0.3425 | 0.9907 | 0.4017 | 0.4398 | ❌ | Agg-0.13 Mem-0.04 Priv-0.29 Utility-0.01 |
| 8  official | `lr5e-05_b3.5_a1_d1_g0.25_ep5` | 0.3574 | 0.2529 | 0.9954 | 0.3868 | 0.5260 | ❌ | Agg-0.14 Mem+0.04 Priv-0.38 Utility-0.00 |
| 9  official | `lr5e-05_b3.5_a1_d1_g0.25_ep10` | 0.3608 | 0.2303 | 1.0088 | 0.3701 | 0.5315 | ❌ | Agg-0.16 Mem+0.04 Priv-0.40 Utility+0.01 |
| 10  official | `lr2e-05_b3.5_a1_d1_g0.25_ep10` | 0.1708 | 0.3139 | 0.9865 | 0.2984 | 0.2912 | ❌ | Agg-0.23 Mem-0.15 Priv-0.32 Utility-0.01 |
| 11  official | `lr2e-05_b4.5_a1_d1_g0.25_ep10` | 0.1771 | 0.2530 | 0.9994 | 0.2830 | 0.3008 | ❌ | Agg-0.25 Mem-0.14 Priv-0.38 Utility-0.00 |
| 12  official | `lr2e-05_b3.5_a1_d1_g0.25_ep5` | 0.1686 | 0.2622 | 0.9955 | 0.2791 | 0.2884 | ❌ | Agg-0.25 Mem-0.15 Priv-0.37 Utility-0.00 |
| 13  official | `lr2e-05_b4.5_a1_d1_g0.25_ep5` | 0.1738 | 0.2321 | 1.0006 | 0.2712 | 0.2961 | ❌ | Agg-0.26 Mem-0.15 Priv-0.40 Utility+0.00 |
| 14  official | `lr2e-05_b3.5_a1_d1_g0.125_ep10` | 0.1556 | 0.2212 | 0.9925 | 0.2509 | 0.2690 | ❌ | Agg-0.28 Mem-0.16 Priv-0.41 Utility-0.01 |
| 15  official | `lr2e-05_b4.5_a1_d1_g0.125_ep10` | 0.1465 | 0.2015 | 0.9918 | 0.2345 | 0.2554 | ❌ | Agg-0.30 Mem-0.17 Priv-0.43 Utility-0.01 |
| 16  official | `lr5e-05_b3.5_a1_d0_g0.25_ep10` | 0.1890 | 0.1347 | 0.9985 | 0.2187 | 0.3178 | ❌ | Agg-0.31 Mem-0.13 Priv-0.50 Utility-0.00 |
| 17  official | `lr2e-05_b3.5_a1_d1_g0.125_ep5` | 0.1339 | 0.1837 | 1.0025 | 0.2157 | 0.2363 | ❌ | Agg-0.31 Mem-0.19 Priv-0.45 Utility+0.00 |
| 18  official | `lr2e-05_b4.5_a1_d1_g0.125_ep5` | 0.1290 | 0.1756 | 0.9982 | 0.2076 | 0.2285 | ❌ | Agg-0.32 Mem-0.19 Priv-0.45 Utility-0.00 |
| 19  official | `lr5e-05_b3.5_a1_d0_g0.25_ep5` | 0.1750 | 0.1272 | 0.9853 | 0.2056 | 0.2973 | ❌ | Agg-0.32 Mem-0.14 Priv-0.50 Utility-0.01 |
| 20  official | `lr5e-05_b4.5_a1_d0_g0.25_ep10` | 0.1538 | 0.1175 | 0.9964 | 0.1873 | 0.2664 | ❌ | Agg-0.34 Mem-0.17 Priv-0.51 Utility-0.00 |
| 21  official | `lr1e-05_b4.5_a1_d1_g0.125_ep5` ※ | 0.1587 | 0.1057 | 0.9992 | 0.1790 | 0.2739 | ❌ | Agg-0.35 Mem-0.16 Priv-0.52 Utility-0.00 |
| 22  official | `lr5e-05_b3.5_a1_d0_g0.125_ep10` | 0.1444 | 0.1122 | 0.9969 | 0.1781 | 0.2522 | ❌ | Agg-0.35 Mem-0.18 Priv-0.52 Utility-0.00 |
| 23  official | `lr5e-05_b4.5_a1_d0_g0.25_ep5` | 0.1385 | 0.1146 | 0.9873 | 0.1769 | 0.2429 | ❌ | Agg-0.35 Mem-0.18 Priv-0.52 Utility-0.01 |
| 24  official | `lr1e-05_b3.5_a1_d1_g0.125_ep5` ※ | 0.1547 | 0.1052 | 1.0025 | 0.1768 | 0.2680 | ❌ | Agg-0.35 Mem-0.17 Priv-0.52 Utility+0.00 |
| 25  official | `lr5e-05_b3.5_a1_d0_g0.125_ep5` | 0.1181 | 0.1091 | 0.9802 | 0.1608 | 0.2108 | ❌ | Agg-0.37 Mem-0.20 Priv-0.52 Utility-0.02 |
| 26  official | `lr5e-05_b4.5_a1_d0_g0.125_ep10` | 0.1171 | 0.1062 | 0.9970 | 0.1583 | 0.2096 | ❌ | Agg-0.37 Mem-0.20 Priv-0.52 Utility-0.00 |
| 27  official | `lr1e-05_b3.5_a1_d0_g0.25_ep10` ※ | 0.1185 | 0.1021 | 1.0086 | 0.1561 | 0.2121 | ❌ | Agg-0.37 Mem-0.20 Priv-0.53 Utility+0.01 |
| 28  official | `lr1e-05_b4.5_a1_d0_g0.25_ep10` ※ | 0.1026 | 0.1012 | 1.0093 | 0.1455 | 0.1862 | ❌ | Agg-0.38 Mem-0.22 Priv-0.53 Utility+0.01 |
| 29  official | `lr1e-05_b3.5_a1_d0_g0.25_ep5` ※ | 0.0985 | 0.1014 | 1.0088 | 0.1428 | 0.1795 | ❌ | Agg-0.39 Mem-0.22 Priv-0.53 Utility+0.01 |
| 30  official | `lr1e-05_b4.5_a1_d0_g0.25_ep5` ※ | 0.0858 | 0.1008 | 1.0106 | 0.1329 | 0.1581 | ❌ | Agg-0.40 Mem-0.23 Priv-0.53 Utility+0.01 |
| 31  official | `lr2e-05_b3.5_a1_d0_g0.25_ep10` | 0.0677 | 0.1196 | 1.0137 | 0.1244 | 0.1270 | ❌ | Agg-0.41 Mem-0.25 Priv-0.51 Utility+0.01 |
| 32  selftrain | `lr1e-05_b4.5_a1_d0_g0.125_ep10` | 0.0709 | 0.1040 | 1.0047 | 0.1214 | 0.1325 | ❌ | Agg-0.41 Mem-0.25 Priv-0.53 Utility+0.00 |
| 33  official | `lr5e-05_b4.5_a1_d0_g0.125_ep5` | 0.0700 | 0.1055 | 0.9830 | 0.1211 | 0.1307 | ❌ | Agg-0.41 Mem-0.25 Priv-0.52 Utility-0.02 |
| 34  official | `lr2e-05_b3.5_a1_d0_g0.25_ep5` | 0.0616 | 0.1124 | 1.0169 | 0.1148 | 0.1161 | ❌ | Agg-0.42 Mem-0.26 Priv-0.52 Utility+0.02 |
| 35  official | `lr1e-05_b3.5_a1_d0_g0.125_ep10` ※ | 0.0594 | 0.1004 | 1.0056 | 0.1079 | 0.1121 | ❌ | Agg-0.42 Mem-0.26 Priv-0.53 Utility+0.01 |
| 36  official | `lr2e-05_b4.5_a1_d0_g0.25_ep10` | 0.0546 | 0.1121 | 1.0180 | 0.1063 | 0.1036 | ❌ | Agg-0.42 Mem-0.27 Priv-0.52 Utility+0.02 |
| 37  official | `lr1e-05_b4.5_a1_d0_g0.125_ep10` ※ | 0.0553 | 0.1002 | 1.0075 | 0.1032 | 0.1048 | ❌ | Agg-0.43 Mem-0.26 Priv-0.53 Utility+0.01 |
| 38  official | `lr2e-05_b4.5_a1_d0_g0.25_ep5` | 0.0502 | 0.1079 | 1.0133 | 0.0995 | 0.0957 | ❌ | Agg-0.43 Mem-0.27 Priv-0.52 Utility+0.01 |
| 39  official | `lr1e-05_b3.5_a1_d0_g0.125_ep5` ※ | 0.0518 | 0.1003 | 1.0059 | 0.0991 | 0.0986 | ❌ | Agg-0.43 Mem-0.27 Priv-0.53 Utility+0.01 |
| 40  official | `lr1e-05_b4.5_a1_d0_g0.125_ep5` ※ | 0.0489 | 0.1002 | 1.0074 | 0.0955 | 0.0933 | ❌ | Agg-0.43 Mem-0.27 Priv-0.53 Utility+0.01 |
| 41  official | `lr1e-05_b4.5_a1_d1_g0.25_ep10` | 0.0412 | 0.1265 | 0.9887 | 0.0904 | 0.0791 | ❌ | Agg-0.44 Mem-0.28 Priv-0.50 Utility-0.01 |
| 42  official | `lr1e-05_b3.5_a1_d1_g0.25_ep10` | 0.0399 | 0.1266 | 0.9896 | 0.0884 | 0.0768 | ❌ | Agg-0.44 Mem-0.28 Priv-0.50 Utility-0.01 |
| 43  official | `lr2e-05_b3.5_a1_d0_g0.125_ep10` | 0.0276 | 0.1054 | 1.0210 | 0.0643 | 0.0538 | ❌ | Agg-0.47 Mem-0.29 Priv-0.52 Utility+0.02 |
| 44  official | `lr2e-05_b3.5_a1_d0_g0.125_ep5` | 0.0274 | 0.1030 | 1.0109 | 0.0636 | 0.0534 | ❌ | Agg-0.47 Mem-0.29 Priv-0.53 Utility+0.01 |
| 45  official | `lr2e-05_b4.5_a1_d0_g0.125_ep5` | 0.0219 | 0.1020 | 1.0111 | 0.0531 | 0.0428 | ❌ | Agg-0.48 Mem-0.30 Priv-0.53 Utility+0.01 |
| 46  official | `lr1e-05_b4.5_a1_d1_g0.25_ep5` | 0.0206 | 0.1183 | 0.9905 | 0.0517 | 0.0403 | ❌ | Agg-0.48 Mem-0.30 Priv-0.51 Utility-0.01 |
| 47  official | `lr2e-05_b4.5_a1_d0_g0.125_ep10` | 0.0211 | 0.1034 | 1.0166 | 0.0516 | 0.0413 | ❌ | Agg-0.48 Mem-0.30 Priv-0.53 Utility+0.02 |
| 48  official | `lr1e-05_b3.5_a1_d1_g0.25_ep5` | 0.0194 | 0.1180 | 0.9963 | 0.0492 | 0.0381 | ❌ | Agg-0.48 Mem-0.30 Priv-0.51 Utility-0.00 |
| 49  official | `lr1e-05_b4.5_a1_d1_g0.125_ep10` | 0.0077 | 0.1097 | 1.0043 | 0.0215 | 0.0153 | ❌ | Agg-0.51 Mem-0.31 Priv-0.52 Utility+0.00 |
| 50  official | `lr1e-05_b3.5_a1_d1_g0.125_ep10` | 0.0057 | 0.1090 | 1.0091 | 0.0160 | 0.0112 | ❌ | Agg-0.51 Mem-0.31 Priv-0.52 Utility+0.01 |

## 二、补算备注（※ = 1−norm 为负，clamp 0 且不参与 HM）

原先 `1−TR<0` 会 `ValueError`，评完也进不了 jsonl。`ou_aggregate.py` 改为 clamp 后由 `ou_backfill_aggregate.sh` 补算。下表不进 Top-K 排名逻辑之外的额外口径；自训行仅作对照，不参与官方命中判定。

| 来源 | 方法 | 超参串 | Mem | Priv | Utility | Agg | 备注 |
|---|---|---|---|---|---|---|---|
| official | UNDIAL | `lr0.0001_beta10_alpha5_epoch5` | 0.2078 | 0.1433 | 0.8964 | 0.2325 | 1-norm(ParaProb) raw=-0.022727<0 → clamp 0 且不参与 HM（target 略高于 init） |
| official | RMU | `lr1e-05_layer5_scoeff100_epoch5` | 0.2340 | 0.1024 | 0.9671 | 0.1991 | 1-norm(TR) raw=-0.002559<0 → clamp 0 且不参与 HM（target 略高于 init） |
| official | SimNPO | `lr1e-05_b4.5_a1_d1_g0.125_ep5` | 0.1587 | 0.1057 | 0.9992 | 0.1790 | 1-norm(TR) raw=-0.001672<0 → clamp 0 且不参与 HM（target 略高于 init） |
| official | SimNPO | `lr1e-05_b3.5_a1_d1_g0.125_ep5` | 0.1547 | 0.1052 | 1.0025 | 0.1768 | 1-norm(TR) raw=-0.002244<0 → clamp 0 且不参与 HM（target 略高于 init） |
| official | RMU | `lr1e-05_layer5_scoeff10_epoch10` | 0.1475 | 0.1010 | 0.9836 | 0.1695 | 1-norm(TR) raw=-0.000387<0 → clamp 0 且不参与 HM（target 略高于 init） |
| official | SimNPO | `lr1e-05_b3.5_a1_d0_g0.25_ep10` | 0.1185 | 0.1021 | 1.0086 | 0.1561 | 1-norm(TR) raw=-0.001850<0 → clamp 0 且不参与 HM（target 略高于 init） |
| official | SimNPO | `lr1e-05_b4.5_a1_d0_g0.25_ep10` | 0.1026 | 0.1012 | 1.0093 | 0.1455 | 1-norm(TR) raw=-0.002852<0 → clamp 0 且不参与 HM（target 略高于 init） |
| official | SimNPO | `lr1e-05_b3.5_a1_d0_g0.25_ep5` | 0.0985 | 0.1014 | 1.0088 | 0.1428 | 1-norm(TR) raw=-0.002768<0 → clamp 0 且不参与 HM（target 略高于 init） |
| official | RMU | `lr1e-05_layer5_scoeff10_epoch5` | 0.0991 | 0.1003 | 0.9948 | 0.1424 | 1-norm(TR) raw=-0.000104<0 → clamp 0 且不参与 HM（target 略高于 init） |
| official | SimNPO | `lr1e-05_b4.5_a1_d0_g0.25_ep5` | 0.0858 | 0.1008 | 1.0106 | 0.1329 | 1-norm(TR) raw=-0.003499<0 → clamp 0 且不参与 HM（target 略高于 init） |
| official | UNDIAL | `lr0.0001_beta3_alpha1_epoch10` | 0.0693 | 0.1038 | 0.9355 | 0.1194 | 1-norm(ParaProb) raw=-0.419432<0 → clamp 0 且不参与 HM（target 略高于 init） |
| official | SimNPO | `lr1e-05_b3.5_a1_d0_g0.125_ep10` | 0.0594 | 0.1004 | 1.0056 | 0.1079 | 1-norm(TR) raw=-0.004282<0 → clamp 0 且不参与 HM（target 略高于 init） |
| official | SimNPO | `lr1e-05_b4.5_a1_d0_g0.125_ep10` | 0.0553 | 0.1002 | 1.0075 | 0.1032 | 1-norm(TR) raw=-0.004183<0 → clamp 0 且不参与 HM（target 略高于 init） |
| official | SimNPO | `lr1e-05_b3.5_a1_d0_g0.125_ep5` | 0.0518 | 0.1003 | 1.0059 | 0.0991 | 1-norm(TR) raw=-0.004186<0 → clamp 0 且不参与 HM（target 略高于 init） |
| official | SimNPO | `lr1e-05_b4.5_a1_d0_g0.125_ep5` | 0.0489 | 0.1002 | 1.0074 | 0.0955 | 1-norm(TR) raw=-0.004088<0 → clamp 0 且不参与 HM（target 略高于 init） |
| official | RMU | `lr1e-05_layer15_scoeff100_epoch10` | 0.0449 | 0.1001 | 0.9940 | 0.0902 | 1-norm(TR) raw=-0.010240<0 → clamp 0 且不参与 HM（target 略高于 init） |
| official | RMU | `lr1e-05_layer10_scoeff100_epoch10` | 0.0431 | 0.0998 | 0.9980 | 0.0876 | 1-norm(TR) raw=-0.006862<0 → clamp 0 且不参与 HM（target 略高于 init） |
| official | RMU | `lr1e-05_layer10_scoeff10_epoch5` | 0.0287 | 0.0999 | 1.0039 | 0.0654 | 1-norm(TR) raw=-0.000188<0 → clamp 0 且不参与 HM（target 略高于 init） |
| official | RMU | `lr1e-05_layer15_scoeff100_epoch5` | 0.0270 | 0.1000 | 0.9983 | 0.0625 | 1-norm(TR) raw=-0.008320<0 → clamp 0 且不参与 HM（target 略高于 init） |
| official | RMU | `lr1e-05_layer10_scoeff100_epoch5` | 0.0249 | 0.0998 | 1.0024 | 0.0586 | 1-norm(TR) raw=-0.006543<0 → clamp 0 且不参与 HM（target 略高于 init） |
| official | UNDIAL | `lr1e-05_beta30_alpha2_epoch10` | 0.0235 | 0.1010 | 1.0051 | 0.0562 | 1-norm(ParaProb) raw=-0.020248<0 → clamp 0 且不参与 HM（target 略高于 init） |
| official | UNDIAL | `lr1e-05_beta10_alpha1_epoch10` | 0.0162 | 0.1003 | 1.0114 | 0.0413 | 1-norm(ParaProb) raw=-0.153496<0 → clamp 0 且不参与 HM（target 略高于 init） |
| selftrain | GradDiff | `lr1e-05_alpha10_epoch10` | 0.9381 | 0.1623 | 0.7714 | 0.3519 | 1-norm(TR) raw=-0.309963<0 → clamp 0 且不参与 HM（target 略高于 init） |
| selftrain | GradDiff | `lr1e-05_alpha5_epoch10` | 0.9461 | 0.1612 | 0.7605 | 0.3499 | 1-norm(TR) raw=-0.360110<0 → clamp 0 且不参与 HM（target 略高于 init） |
| selftrain | GradAscent | `default_lr1e-05_ep10` | 0.9724 | 0.4809 | 0.0000 | 0.0000 | 1-norm(TR) raw=-0.460032<0 → clamp 0 且不参与 HM（target 略高于 init） |

## 三、待复现超参（官方评测选出，本机尚未重训）

| 方法 | 超参串 | 官方 Agg | Mem | Priv | Utility | HF repo |
|---|---|---|---|---|---|---|
| SimNPO | `lr5e-05_b3.5_a1_d1_g0.125_ep10` | 0.4716 | 0.3020 | 0.4880 | 0.9992 | `open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_SimNPO_lr5e-05_b3.5_a1_d1_g0.125_ep10` |
| AltPO | `lr2e-05_beta0.05_alpha1_epoch10` | 0.5826 | 0.4284 | 0.5724 | 0.9364 | `open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_AltPO_lr2e-05_beta0.05_alpha1_epoch10` |

- **SimNPO `lr5e-05_b3.5_a1_d1_g0.125_ep10`**：官方 48 点最好（未命中论文 0.53/0.32/0.63/1.00，差在 Priv）。本机复现：tofu_unlearn_one SimNPO + 该串（tag S_p1best），由 scripts/ou_repro_p1_winners.sh 在 P1 结束后串行开训。
- **AltPO `lr2e-05_beta0.05_alpha1_epoch10`**：官方 ckpt 评测 Agg 0.5826，高于论文 SimNPO 靶 0.53 / Retain 靶 0.58。本机复现：bash community/methods/AltPO/run.sh（已钉死该串；DPO + alt5_seed_0，lr=2e-5, beta=0.05, alpha=1, ep=10，双卡 4×4×2、sdpa、adamw_torch）。论文 Table 3 无 AltPO 行，不替代 SimNPO 命中门槛。

## 四、未命中 / 缺失项归因

- **GradAscent**：官方 HF 仓库没有 GradAscent 的 forget10 ckpt：398 个 unlearn_tofu_Llama-3.2-1B-Instruct_forget10_* 里不存在该方法（分布：AltPO/NPO/RMU/UNDIAL/IdkDPO 各 54、SimNPO 48、GradDiff/IdkNLL 各 40）。P1 该格改由 P2 自训补齐。
- **Retain Mem**：论文 0.31 vs 本实现 0.3462（+0.036）：在官方 ES/EM/ParaProb 数值与 HM 结构下，四个非负分量无法给出 0.31（需要 TR_norm>1）。见 docs/zh/ou-table3-p0.md §三。
- **GradDiff Priv**：论文 3.27e-03：对称映射下过遗忘侧 s(dev=+0.6)≈0.15，与论文不符；可用 ou_aggregate.py --s-mia-mode piecewise 做归因实验。
- **Mem HM clamp**：1−norm(*)<0（常见 TR 略高于 init）时夹到 0 且不参与 Mem HM，避免报错或整维塌成 0。带 ※ 的行见 jsonl `notes`。
- **Retain**：靶值存在但 `results/ou_table3_runs.jsonl` 里还没有该方法的任何记录（需先跑 ou_eval_batch.sh --only Retain，或 P0-3 的 retain90/full 本地评测）

