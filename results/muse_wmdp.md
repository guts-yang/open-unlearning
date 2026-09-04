# MUSE / WMDP 复现结果

口径与 TOFU Table 3 四维不同，**不进** `ou_table3_runs.jsonl`。

## MUSE

| split | method | forget KnowMem | retain KnowMem | VerbMem | PrivLeak | ES | HM(KM,Priv≈0) | note |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Books | DPO | — | — | — | — | — | — | train failed |
| Books | Finetuned | 0.4712 | 0.6913 | 0.9970 | -57.3410 | 0.9163 | 0.4478 | 官方对照日志，非本轮自训 |
| Books | GradAscent | 0.0000 | 0.0000 | 0.0000 | -14.7744 | 0.0079 | 2e-12 | ok |
| Books | GradDiff | 0.0000 | 0.0000 | 0.0000 | -14.4786 | 0.0079 | 2e-12 | ok |
| Books | NPO | — | — | — | — | — | — | train failed |
| Books | RMU | — | — | — | — | — | — | train failed |
| Books | Retain | 0.3029 | 0.6874 | 0.1445 | 8.1600 | 0.0107 | 0.4555 | 官方对照日志，非本轮自训 |
| Books | SatImp | — | — | — | — | — | — | train failed |
| Books | SimNPO | 0.4149 | 0.6933 | 0.3386 | -62.0932 | 0.2017 | 0.3962 | ok |
| Books | UNDIAL | — | — | — | — | — | — | train failed |
| Books | WGA | — | — | — | — | — | — | train failed |
| News | CEU | 0.0000 | 0.0000 | 0.0000 | 21.1797 | 0.0079 | 2e-12 | ok |
| News | DPO | — | — | — | — | — | — | train failed |
| News | Finetuned | 0.6443 | 0.5552 | 0.5789 | -99.8111 | 0.2954 | 0.0038 | 官方对照日志，非本轮自训 |
| News | GradAscent | — | — | — | — | — | — | skip_eval |
| News | GradAscent | 0.0000 | 0.0000 | 0.0000 | 54.6809 | 0.0079 | 2e-12 | ok |
| News | GradDiff | 0.3268 | 0.2607 | 0.1007 | 36.4190 | 0.0109 | 0.4317 | ok |
| News | NPO | — | — | — | — | — | — | train failed |
| News | PDU | 0.0000 | 0.0000 | 0.0000 | — | — | — | eval failed |
| News | RMU | — | — | — | — | — | — | train failed |
| News | Retain | 0.3279 | 0.5602 | 0.2016 | -4.7200 | 0.0244 | 0.4879 | 官方对照日志，非本轮自训 |
| News | SatImp | — | — | — | — | — | — | train failed |
| News | SimNPO | 0.5550 | 0.3841 | 0.3468 | -99.8321 | 0.0995 | 0.0033 | ok |
| News | UNDIAL | — | — | — | — | — | — | train failed |
| News | WGA | — | — | — | — | — | — | train failed |

## WMDP-cyber

| method | wmdp_cyber acc | mmlu acc | note |
|---|---:|---:|---|
| RMU | — | — | train failed |
