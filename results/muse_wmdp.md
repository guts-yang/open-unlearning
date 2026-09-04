# MUSE / WMDP 本机复现

更新时间：2026-09-04T10:23:58+08:00

口径见 `docs/zh/【3】muse-wmdp-repro.md`。**不进** `ou_table3_runs.jsonl`。

## MUSE 自测

### News

| 方法 | KnowMem_Df↓ | KnowMem_Dr↑ | VerbMem_Df↓ | PrivLeak→0 |
|---|---:|---:|---:|---:|
| GradAscent | 0.0000 | 0.0000 | 0.0000 | 54.6809 |
| GradDiff | 0.3268 | 0.2607 | 0.1007 | 36.4190 |
| NPO | — | — | — | — |
| SimNPO | 0.5550 | 0.3841 | 0.3468 | -99.8321 |
| RMU | — | — | — | — |
| CEU | 0.0000 | 0.0000 | 0.0000 | 21.1797 |
| PDU | 0.0000 | 0.0000 | 0.0000 | — |
| DPO | — | — | — | — |
| UNDIAL | — | — | — | — |
| WGA | — | — | — | — |
| SatImp | — | — | — | — |
| Finetuned（对照） | 0.6443 | 0.5552 | 0.5789 | -99.8111 |
| Retain（对照） | 0.3279 | 0.5602 | 0.2016 | -4.7200 |

### Books

| 方法 | KnowMem_Df↓ | KnowMem_Dr↑ | VerbMem_Df↓ | PrivLeak→0 |
|---|---:|---:|---:|---:|
| GradAscent | 0.0000 | 0.0000 | 0.0000 | -14.7744 |
| GradDiff | 0.0000 | 0.0000 | 0.0000 | -14.4786 |
| NPO | — | — | — | — |
| SimNPO | 0.4149 | 0.6933 | 0.3386 | -62.0932 |
| RMU | — | — | — | — |
| DPO | — | — | — | — |
| UNDIAL | — | — | — | — |
| WGA | — | — | — | — |
| SatImp | — | — | — | — |
| Finetuned（对照） | 0.4712 | 0.6913 | 0.9970 | -57.3410 |
| Retain（对照） | 0.3029 | 0.6874 | 0.1445 | 8.1600 |

## WMDP 自测

| 方法 | Bio Acc↓ | MMLU↑ | Cyber Acc↓ | MT-Bench↑ |
|---|---:|---:|---:|---:|
| RMU | — | — | 0.2778 | — |

本机 WMDP 只评了 cyber；Bio、MT-Bench 未跑。MMLU 本轮补评尚未写入 summary。Cyber 为 `wmdp_cyber/acc=0.2778`（jsonl 里曾误记 stderr 0.0101）。

NPO / RMU / DPO 等空行是训练或评测未出数，不是指标为 0。
