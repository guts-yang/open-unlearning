# MUSE / WMDP 还需要复现什么

> 2026-09-04 09:33 已停调度。本机结果：`results/muse_wmdp.md`、`results/muse_wmdp_runs.jsonl`。  
> 口径说明见 [`【3】muse-wmdp-repro.md`](【3】muse-wmdp-repro.md)。**不进** `ou_table3_runs.jsonl`。

本机（2×V100 32GB，Docker cgroup **50GiB**）只能单卡 fp16 + Adafactor。官方是 A100 + bf16 + 2 卡 ZeRO3 + AdamW。下表「本机已有」**不能**当成与 `docs/repro.md` 完全同实现。

---

## 一、本机已经有数（不必重跑，除非换官方口径）

| 基准 | 方法 | forget KM | retain KM | PrivLeak | 备注 |
|---|---|---:|---:|---:|---|
| News | GradAscent | 0 | 0 | 54.7 | 与 repro GA 数量级一致 |
| News | GradDiff | 0.327 | 0.261 | 36.4 | retain 也掉了 |
| News | SimNPO | 0.555 | 0.384 | -99.8 | 几乎没忘掉 |
| News | CEU | 0 | 0 | 21.2 | 全塌 |
| Books | GradAscent | 0 | 0 | -14.8 | 全塌 |
| Books | GradDiff | 0 | 0 | -14.5 | 全塌 |
| Books | SimNPO | 0.415 | 0.693 | -62.1 | forget 仍高，retain 接近官方 Retain |

对照日志（非自训）：News/Books 的 Finetuned、Retain 已在表里。

News **PDU**：训完了，评测 PrivLeak/MIA 报 NaN（`eval_fail`）。若要完整一行，只需 **重评** 已有 ckpt；但默认 `KEEP_CKPT=0`，权重可能已删，需重训或从 `saves/unlearn/muse_Llama-2-7b-hf_News_PDU` 看是否还在。

---

## 二、停的时候正在跑（优先补）

单模，这台机器能跑完：

| 优先级 | 组 | 命令 | 说明 |
|---|---|---|---|
| P1 | MUSE-Books **CEU** | `bash scripts/muse_unlearn_one.sh CEU Books` | 09:19 起训，约 2/170 被停 |
| P1 | MUSE-Books **PDU** | `bash scripts/muse_unlearn_one.sh PDU Books` | 刚 load；News PDU 评测曾 NaN |
| P1 | WMDP-cyber **GradAscent** | `bash scripts/wmdp_unlearn_one.sh GradAscent` | 从未开训 |
| P1 | WMDP-cyber **GradDiff** | `bash scripts/wmdp_unlearn_one.sh GradDiff` | 从未开训 |
| P1 | WMDP-cyber **SimNPO** | `bash scripts/wmdp_unlearn_one.sh SimNPO` | 从未开训 |
| P1 | WMDP-cyber **CEU** | `bash scripts/wmdp_unlearn_one.sh CEU` | 从未开训 |

WMDP：`max_steps=80`，lr=5e-5，zephyr-7b-beta，只报 `wmdp_cyber` acc + `mmlu`。

---

## 三、本机 OOM / 缺字段（要复现必须换机器或改加载）

这些方法会 `deepcopy` 一份 7B 参考模型，单卡 32GB 必爆。官方 P0 里的 **NPO、RMU** 都在这档，**和 repro.md 对齐必须补**。

需要：**≥80–100GiB 容器内存** 的 2 卡（或 A100 + ZeRO3），或改 `train.py` 不要每个 rank 整模 `from_pretrained`。

| 组 | 失败原因 | 是否论文/官方必补 |
|---|---|---|
| News/Books **NPO** | ref 模型 OOM | **是**（repro P0） |
| News/Books **RMU** | ref + EMBED_DIFF OOM | **是**（repro P0） |
| WMDP-cyber **RMU** | 同上 | **是**（官方 WMDP 主方法） |
| News/Books **UNDIAL / WGA / SatImp** | ref OOM | 仓库扩展，有卡再补 |
| News/Books **DPO** | 要 ref + forget `alternate`；MUSE 语料没有 | 可不做，或先改数据 |
| WMDP **NPO / UNDIAL / WGA / SatImp** | ref OOM | 扩展 |

---

## 四、计划里本来就不做

- MUSE scalability / sustainability
- WMDP bio（forget gated）、chem（无 jsonl）
- TPO / AltPO / IdkDPO / IdkNLL（要 TOFU 式标注）
- Appendix F.2 的 TOFU-1B 网格套到 7B
- 写入 Table 3 四维

---

## 五、建议复现顺序（换卡后）

1. **对齐 repro.md**：News+Books 的 NPO、RMU；WMDP-cyber RMU（官方 AdamW / 有效 batch / bf16）。  
2. **补本机没跑完的单模**：Books CEU、Books PDU、WMDP × {GA, GradDiff, SimNPO, CEU}。  
3. **选做扩展**：UNDIAL / WGA / SatImp / DPO（DPO 先解决 alternate）。  
4. 结果只追加 `results/muse_wmdp_runs.jsonl`，刷新 `results/muse_wmdp.md`。

单卡续跑（本机，仅 P1 单模）：

```bash
source /root/autodl-tmp/env_hf.sh
source /root/autodl-tmp/envs/unlearning/bin/activate
# 先清 claim：rm -f /root/autodl-tmp/logs/muse_wmdp_claim/*.claimed
bash scripts/muse_unlearn_one.sh CEU Books
bash scripts/muse_unlearn_one.sh PDU Books
bash scripts/wmdp_unlearn_one.sh GradAscent
bash scripts/wmdp_unlearn_one.sh GradDiff
bash scripts/wmdp_unlearn_one.sh SimNPO
bash scripts/wmdp_unlearn_one.sh CEU
```

双模请不要在本机 `NUM_PROCESSES=2` 硬开，会再被 50GiB cgroup SIGKILL。
