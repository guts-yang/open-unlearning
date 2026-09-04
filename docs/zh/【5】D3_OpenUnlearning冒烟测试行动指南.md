# D3 冒烟测试行动指南 v2：SpecGap 跑分代码实现（模型已就位版）

> 前提更新：三基准所需模型**已全部下载到本地**，环境已装好。本指南只剩一件事：**写好 SpecProbe 探针代码并跑分**。
> 版本：v2.0 · 2026-09-04 · 配套：`D3投机解码分歧遗忘_开题报告.md` §6.1

---

## 1. 算法实现思路（先讲清楚，再上代码）

### 1.1 数学目标（一行）

对探针语料的每个序列位置 $t$，计算两模型全词表分布的重合度：

$$
s_t \;=\; \sum_{v \in \mathcal{V}} \min\!\big(p_t(v),\, q_\theta(v)\big) \;=\; 1 - D_{TV}(p_t, q_\theta) \;\in\; [0,1]
$$

- $p$：冻结 draft（遗忘前 checkpoint）；$q_\theta$：target（遗忘后模型）；
- 样本级重合度 $\alpha_i = \operatorname{mean}_t s_t$，**SpecGap** $= 1 - \alpha_i$；
- 集合级：遗忘集 SpecGap_f 与保留集 SpecGap_r 各报 mean ± bootstrap CI，核心判据是二者效应量 Cohen's d。

**一句话记忆：这就是一个"双分布逐位置重合度积分"，名字借自投机解码，实现上是纯前向计算，无采样、无反传。**

### 1.2 六步数据流（张量形状全部标出）

```
① 分词        question + answer 拼接 → input_ids [B, T]，另算 prompt 长度 len_p
② 双前向      draft 与 target 各一次 teacher-forcing → logits_p, logits_q [B, T, V]
③ 位置对齐    logits[:, t] 预测的是 input_ids[:, t+1]
              → 错一位：logits[:, :-1] vs targets = input_ids[:, 1:]，掩码同步错位
④ 答案区掩码  只在 t ≥ len_p 的答案 token 上累计（知识所在位置），prompt 区丢弃
⑤ 分块积分    s_t += Σ_v min(p, q)，按词表分块（每块 8192）逐块累加，防 OOM
⑥ 聚合报告    s_t → α_i (mean_t) → SpecGap_i = 1 − α_i → 集合 mean ± CI + Cohen's d
```

### 1.3 五个决定正确性的实现要点

**要点 1：分块 softmax 必须先算全局 logsumexp（最容易写错的点）**
softmax 的分母是**全词表**的。直接对 `logits[:, :, v0:v1]` 切片做 `log_softmax` 得到的是切片内归一化的错误结果。正确做法：

```python
lse = torch.logsumexp(logits, dim=-1, keepdim=True)   # 全词表 [B, T, 1]，fp32
# 分块内：
p_chunk = (logits[:, :, v0:v1] - lse).exp()            # = 全局 softmax 的对应切片
s += torch.minimum(p_chunk, q_chunk).sum(-1)
```

logsumexp 本身也可以分块累加（数值稳定），但 32k 词表在 A800 上一次算完毫无压力，不必过度工程。

**要点 2：精度——logits 转 fp32 再 softmax**
bf16 的 softmax 在小概率尾部（恰好是 min(p,q) 的主要贡献区）误差不可忽略。模型权重保持 bf16 推理，logits upcast 成 fp32 后再算。显存账：B=4、T=512、V=32k 的 fp32 logits ≈ 262MB/模型，双模型 ≈ 524MB，80G 卡完全无压力。

**要点 3：答案区掩码（SpecGap 的"定位"能力来源）**
只对答案 token 位置累计 $s_t$。这样输出的不只是一个集合级标量，还有**逐位置 $\alpha_t$ 剖面**——$\arg\min_t s_t$ 就是"改得最狠的 token 位置"，与作者事实 token 对齐后即假遗忘检测的可视化证据（论文 Figure 3）。边界注意：question 与 answer 拼接后的分词结果 ≠ 两者分别分词的拼接，len_p 有 ±1 token 的边界误差，冒烟阶段可接受，正式实验用"答案首 token 对齐复查"消掉。

**要点 4：draft logits 离线缓存（E0 批量审计的省钱开关）**
draft 是冻结的，同一探针集的 logits_p 只需算一次，存盘（bf16，1k 样本 ≈ 数 GB）。之后每来一个新 target checkpoint，只做一次前向 + 读缓存积分——E0 扫 8 个方法 × 3 splits 时，成本从 16 次前向降为 8 次。缓存键 = (模型路径, 数据集, split, n, max_len, 分词器) 的哈希，防串味。

**要点 5：与"冒烟近似版"的边界（诚实纪律）**
此前冒烟版用采样路径概率 $p_t(y_t)$ 做重合度代理——那是 $\sum_v \min(p,q)$ 的**下界代理**，方向正确但数值不可比。本版起一切跑分以全词表 min 积分为准，两版数值**不混用、不进同一张表**。

### 1.4 双模型同词表前置断言

恒等式要求 p、q 在**同一词表空间**上做 min。TOFU 同家族模型（LLaMA-2-7B-chat 与其 GA 遗忘版）词表天然一致；换基座抽查时先断言：

```python
assert tok_p.vocab_size == tok_q.vocab_size
assert tok_p.encode("test") == tok_q.encode("test")   # 关键采样路径一致性
```

---

## 2. 完整代码 `specprobe.py`（可直接跑）

```python
"""
SpecProbe: 双模型逐位置分布重合度 (Σ min(p,q)) 跑分
用法:
  python specprobe.py --draft <遗忘前模型路径> --target <遗忘后模型路径> \
      --splits forget05 retain95 --n 200 --max-len 512 --out result.json
  python specprobe.py --draft PATH --target PATH --self-test   # sanity: 自探应≈1
"""
import argparse, json, hashlib, os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

CHUNK = 8192          # 词表分块大小
CACHE_DIR = "saves/specprobe_cache"

def load_model(path):
    tok = AutoTokenizer.from_pretrained(path)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()
    return model, tok

@torch.no_grad()
def answer_mask_and_ids(tok, question, answer, max_len):
    """拼接分词 + 答案区掩码。返回 input_ids [1,T] 与 bool 掩码（True=答案 token）。"""
    full = tok(question + " " + answer, truncation=True, max_length=max_len,
               return_tensors="pt").input_ids
    len_p = tok(question, truncation=True, max_length=max_len).input_ids.shape[1]
    mask = torch.zeros_like(full, dtype=torch.bool)
    mask[:, len_p:] = True
    return full, mask

@torch.no_grad()
def forward_logits(model, input_ids):
    out = model(input_ids=input_ids)
    return out.logits[:, :-1].float()          # [1, T-1, V]，预测 input_ids[:,1:]

@torch.no_grad()
def min_overlap(logits_p, logits_q, mask_tgt, chunk=CHUNK):
    """逐位置 Σ_v min(p,q)。logits_*: [1, T-1, V] fp32; mask_tgt: [1, T-1] bool。"""
    lse_p = torch.logsumexp(logits_p, dim=-1, keepdim=True)   # 全词表归一化分母
    lse_q = torch.logsumexp(logits_q, dim=-1, keepdim=True)
    V = logits_p.shape[-1]
    s = torch.zeros(logits_p.shape[:2], device=logits_p.device)   # [1, T-1]
    for v0 in range(0, V, chunk):
        v1 = min(v0 + chunk, V)
        p = (logits_p[:, :, v0:v1] - lse_p).exp()
        q = (logits_q[:, :, v0:v1] - lse_q).exp()
        s += torch.minimum(p, q).sum(-1)
    s = s[mask_tgt]                              # 只留答案区位置 → [n_answer]
    return s                                     # 逐位置重合度，∈[0,1]

@torch.no_grad()
def draft_logits_cached(model, tok, items, max_len, cache_key):
    """draft 前向 + 磁盘缓存（E0 批量审计复用）"""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, cache_key + ".pt")
    if os.path.exists(path):
        return torch.load(path)
    outs = []
    for q, a in items:
        ids, _ = answer_mask_and_ids(tok, q, a, max_len)
        outs.append(forward_logits(model, ids.to("cuda:0")).squeeze(0).bfloat16().cpu())
    torch.save(outs, path)
    return outs

def cohens_d(a, b):
    a, b = np.asarray(a), np.asarray(b)
    pooled = np.sqrt(((len(a)-1)*a.std()**2 + (len(b)-1)*b.std()**2) / (len(a)+len(b)-2))
    return (a.mean() - b.mean()) / (pooled + 1e-12)

def bootstrap_ci(x, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    means = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n_boot)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--splits", nargs="+", default=["forget05", "retain95"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--out", default="specprobe_result.json")
    ap.add_argument("--self-test", action="store_true")   # sanity: target=draft
    args = ap.parse_args()

    p_model, tok_p = load_model(args.draft)
    target_path = args.draft if args.self_test else args.target
    q_model, tok_q = load_model(target_path)

    # 前置断言：同词表空间
    assert tok_p.vocab_size == tok_q.vocab_size, "两模型词表不一致，min(p,q) 无意义"

    results = {}
    for split in args.splits:
        ds = load_dataset("locuslab/TOFU", split)["train"].select(range(args.n))
        items = [(x["question"], x["answer"]) for x in ds]

        key = hashlib.md5(json.dumps(
            [args.draft, split, args.n, args.max_len, tok_p.name_or_path]).encode()).hexdigest()[:16]
        logits_list = draft_logits_cached(p_model, tok_p, items, args.max_len, key)

        gaps, profiles = [], []
        for (q, a), lp in zip(items, logits_list):
            ids, mask = answer_mask_and_ids(tok_q, q, a, args.max_len)
            ids, mask = ids.to("cuda:0"), mask.to("cuda:0")
            lq = forward_logits(q_model, ids).squeeze(0).cpu()
            s_t = min_overlap(lp.float().unsqueeze(0), lq.unsqueeze(0),
                              mask[:, 1:].cpu())          # 逐位置重合度
            gaps.append(1.0 - s_t.mean().item())          # 样本级 SpecGap
            profiles.append(s_t.tolist())                 # 逐位置 α_t 剖面（假遗忘证据源）

        lo, hi = bootstrap_ci(gaps)
        results[split] = {"specgap_mean": float(np.mean(gaps)),
                          "ci95": [lo, hi], "profiles": profiles,
                          "argmin_positions": [int(np.argmin(p)) for p in profiles]}

        if args.self_test:
            assert np.mean(gaps) < 0.001, "自探 SpecGap≠0，探针实现有 bug"
        print(f"[{split}] SpecGap = {np.mean(gaps):.4f}  CI95=[{lo:.4f}, {hi:.4f}]")

    if len(args.splits) == 2 and not args.self_test:
        f, r = results[args.splits[0]]["specgap_mean"], results[args.splits[1]]["specgap_mean"]
        # 效应量用逐样本 gap 数组重算（此处简化，正式版把 gaps 数组也存盘）
        print(f"\n生死判据: forget({f:.4f}) vs retain({r:.4f})  →  详细 d 值见存盘数据")

    json.dump(results, open(args.out, "w"), indent=2)
    print(f"已写入 {args.out}")

if __name__ == "__main__":
    main()
```

---

## 3. 跑分流程（三步）

```bash
# Step 1 自检（1 分钟，必做）：draft 探自身，SpecGap 必须 ≈ 0
python specprobe.py --draft <本地遗忘前模型路径> --target 同路径 --self-test --n 20

# Step 2 遗忘集 vs 保留集（P0 核心跑分）
python specprobe.py --draft <遗忘前> --target <GA遗忘后> \
  --splits forget05 retain95 --n 200 --out p0_result.json

# Step 3 判据
# forget SpecGap 显著 > retain SpecGap，Cohen's d > 0.8 → 项目 go
# 顺手人工抽查 argmin_positions 是否落在作者事实 token 上（10 条足够）
```

**判读表**：

| 观测 | 结论 |
|---|---|
| 自探 SpecGap ≈ 0（<1e-3） | 探针实现正确 ✓ |
| forget ≫ retain，d > 0.8 | H1 成立，信号存在 → 进入主训练 |
| forget ≈ retain | 信号不存在 → 触发开题报告 §8 备选路线（逐层 logit lens 替代输出层） |
| forget > retain 但 d < 0.8 | 信号弱 → 增大 n / 检查答案掩码边界 / 换更长探针语料 |

---

## 4. 精简坑表（本阶段相关的 4 条）

| 坑 | 解法 |
|---|---|
| 切片直接 `log_softmax` | 错误结果（切片内归一化）——必须全局 `logsumexp` 后再切片（代码已内置） |
| bf16 下算 softmax | 尾部概率误差大——logits upcast fp32（代码已内置） |
| prompt/answer 拼接分词边界 | len_p 有 ±1 误差——冒烟可接受，正式版做首 token 对齐复查 |
| 缓存串味 | 缓存键含模型路径+split+n+max_len 哈希，换任何一项自动失效（代码已内置） |

---

## 5. P0 之后的直接复用

同一份脚本改两个参数即升级为 E0 批量审计：把 `--target` 循环换成官方 8 方法 checkpoint 列表（draft logits 缓存命中，只跑 target 前向），输出即为"全 leaderboard 假遗忘审计表"（v2 实验设计 §3）。逐位置 `profiles` 字段就是论文 Figure 3（$\alpha_t$ 剖面对比）的原始数据。

---

## 6. 项目内正式实现（2026-09-04）

指南 §2 的单文件代码是算法草图；仓库正式实现已按 OpenUnlearning 的数据和实验约定拆分：

| 文件 | 作用 |
|---|---|
| `src/evals/specgap.py` | 分布重合度、统计量、版本化 draft 分片缓存 |
| `src/specprobe.py` | TOFU SpecProbe 命令行入口 |
| `scripts/specgap_p0.sh` | self-test + P0 生死门 |
| `configs/specgap/e0_tofu_forget10.yaml` | E0 八方法代表 checkpoint 清单 |
| `scripts/specgap_e0.sh` | E0 下载→审计→删本次下载权重，支持断点续跑 |
| `scripts/specgap_table.py` | 生成 `results/specgap_e0.jsonl` / `.md` |

### 6.1 相对草图的正确性修正

1. 不再用 `question + " " + answer` 单独分词。正式实现复用
   `data.utils.preprocess_chat_instance` 和 Llama-3.2 项目 chat template；用
   `labels[1:] != -100` 选择答案 token 对应的预测位置，消除 prompt/answer 边界误差。
2. draft cache 只保存答案位置的 `[A,V]` bf16 logits，每个样本一个 shard；
   积分时逐 shard 搬回 GPU。缓存键包含模型文件、tokenizer、数据 fingerprint、
   样本索引、chat template、max length 和 schema 版本。
3. target 与缓存 draft logits 在 GPU 上以 fp32 做全局 `logsumexp` 和分块积分；
   不采用草图中 `.cpu()` 后进行大矩阵积分的低效路径。
4. target checkpoint 自带 tokenizer 时必须与 draft 词表和特殊 token 完全一致；
   checkpoint 未携带 tokenizer 时明确记录 `draft_fallback`。

### 6.2 P0 命令与输出

```bash
cd /usr/local/open-unlearning
bash scripts/specgap_p0.sh
```

默认设置：

- draft：`open-unlearning/tofu_Llama-3.2-1B-Instruct_full`
- target：`$SAVES/unlearn/tofu_1B_SimNPO_forget10`
- forget10：全量 400 条
- retain90：`seed=0` 固定等量抽取 400 条
- 输出：`$SAVES/specgap/p0_self_test.json`、`p0_local_simnpo.json`
- 通过条件：self-test mean `<1e-3`，且正式 P0 满足
  `SpecGap_f > SpecGap_r`、Cohen's d `>0.8`

P0 JSON 顶层包含 `draft`、`target`、`settings`、`splits`、`comparison` 和
`gate`。每个 split 含 mean/std/CI、逐样本 gap、答案 token 数、argmin 位置和
逐 token overlap profile。

### 6.3 E0 命令与结果

P0 通过后运行：

```bash
bash scripts/specgap_e0.sh
# 小批试跑 / 断点调试
bash scripts/specgap_e0.sh --limit 1
bash scripts/specgap_e0.sh --only SimNPO
```

E0 固定扫描配置中的八个代表 checkpoint，并复用同一 draft cache。已存在结果默认
跳过；`--force` 可重跑。只有脚本本次下载的权重会在单条结束后删除。

汇总命令可独立重跑：

```bash
python scripts/specgap_table.py
```

结果表报告 SpecGap_f/r、各自 bootstrap CI、差值和 Cohen's d；仅按 checkpoint
名称关联 `results/ou_table3_runs.jsonl` 中已经存在的 Mem/Priv/Utility/Agg，
不会为了补齐四维触发额外评测。SpecGap 原始结果与 OU `TOFU_SUMMARY.json` 分开，
避免改变 Table 3 的既有口径。
