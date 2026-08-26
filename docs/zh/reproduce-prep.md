# 复现准备（针对 2×A800 云主机）

对应问题：在本仓库复现遗忘方法前要做什么；以及当前云主机配置是否够用。

本文按**本仓库实际入口**书写（`src/train.py`、`setup_data.py`、`scripts/tofu_unlearn.sh`）。对照数字见 [复现结果](./repro.md)；Hydra 覆盖见 [实验配置与运行](./experiments.md)。

---

## 结论（先看这张表）

| 项目 | 官方复现 | 当前云主机 | 判断 |
|------|----------|------------|------|
| GPU | 2× L40s 48GB | **2× NVIDIA A800 80GB** | **够用，且更宽裕** |
| 分布式 | Accelerate + DeepSpeed ZeRO-3，`num_processes: 2` | 正好 2 卡 | 可直接用默认 `configs/accelerate/default_config.yaml` |
| CPU / 内存 | 未写死 | 28 vCPU，240GB RAM | 足够 |
| Python | `>=3.11`，文档示例 3.11 | 镜像 **3.12** | 满足下限；更稳妥仍建议单独建 **3.11** 环境 |
| PyTorch | 钉死 `torch==2.4.1` | 镜像 **2.8.0** + CUDA **12.8** | **不要直接在镜像环境里 `pip install -r requirements.txt`**，会冲掉 2.8 或装上不匹配的 CUDA 轮子 |
| 系统盘 | — | **30GB** | **不够放模型和 checkpoint** |
| 数据盘 | — | 50GB 免费 SSD + 250GB 付费 | 代码、HF 缓存、`saves/` 必须放到数据盘；全量脚本建议用满 **250GB** |

**显存结论：** 双卡 A800 80GB 可以跑通仓库里的 TOFU（含 8B）和 MUSE（Llama-2 7B）默认遗忘脚本。不要先跑完整 `scripts/tofu_unlearn.sh`（3 模型 × 5 方法 × 3 划分 ≈ 45 次训练+评测），磁盘和时间都会爆。先做下方「冒烟实验」。

**不能对齐 `docs/repro.md` 数字的原因（即使卡更强）：** 换 GPU / CUDA / 是否 ZeRO-3 / 单卡还是双卡，官方也写明结果会漂。本机目标应是：**流程跑通、指标量级合理**；逐位对齐请尽量复刻 `torch==2.4.1` + 双卡 DeepSpeed + 脚本里的 batch。

---

## 一、上机后立刻做的检查

```bash
nvidia-smi                  # 确认 2 张 A800、驱动正常
python3 --version           # 镜像多为 3.12
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"

df -h                       # 看系统盘 / 数据盘挂载点（常见 /root、/data、/workspace）
echo $HF_HOME $HF_TOKEN
```

记下**数据盘挂载路径**（下文用 `$DATA` 表示，请换成真实路径，例如 `/data` 或 `/workspace`）。

建议目录：

```bash
export DATA=/data                    # 改成实际挂载点
export HF_HOME=$DATA/hf-cache
export HF_DATASETS_CACHE=$DATA/hf-cache/datasets
mkdir -p $HF_HOME $DATA/open-unlearning $DATA/saves
```

把仓库 clone 到 `$DATA/open-unlearning`，不要放在 30GB 系统盘。

---

## 二、账号与网络

1. **Hugging Face**
   - 评测 log：`open-unlearning/eval`（dataset）
   - 目标模型：`open-unlearning/tofu_*_full`、`muse-bench/MUSE-News_target` 等
   - 数据集：`locuslab/TOFU`、`muse-bench/MUSE-News` / `MUSE-Books`
   - 若下载 Meta 原版 Llama（自己微调时）：需要许可 + `huggingface-cli login`
   - 直接用 `open-unlearning/` 上已微调的 TOFU 模型，一般**不必** Llama 许可

```bash
pip install -U huggingface_hub
huggingface-cli login          # 若 Hub 限流或 gated 模型失败再登录
```

2. 机器需能访问 Hugging Face（或配镜像）。`setup_data.py --wmdp` 还要用 `wget` 拉 S3，并需要 `unzip`。
3. 可选：`wandb login`（配置里若开了 wandb）。

---

## 三、Python 环境（推荐做法）

镜像是 **PyTorch 2.8.0 + Python 3.12 + CUDA 12.8**。本仓库 `requirements.txt` 钉的是 **`torch==2.4.1`、`deepspeed==0.15.4`、`bitsandbytes==0.44.1`、`accelerate==0.34.2`**。混装很容易出现：DeepSpeed 编译失败、bitsandbytes 对不上、flash-attn 装不上。

### 推荐：独立 conda / venv（更接近官方）

```bash
# 若镜像有 conda；没有则用 python3.11 的 venv
conda create -n unlearning python=3.11 -y
conda activate unlearning
cd $DATA/open-unlearning

# 先装与 CUDA 匹配的 torch 2.4.1（按机器 CUDA 选 cu121 或 cu124 索引）
# 示例（CUDA 12.x 常见）：
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124

pip install -e ".[lm-eval]"          # 会再装 requirements.txt 里其余包
# 若 pip 试图重装 torch，先确认仍是 2.4.1 + CUDA 可用

pip install --no-build-isolation flash-attn==2.6.3
```

`flash-attn` 可能现场编译，需要 `nvcc` 与对应 CUDA toolkit。编译失败可先**去掉 flash attention** 跑 1B 冒烟：

```bash
python src/train.py --config-name=unlearn.yaml experiment=unlearn/tofu/default \
  "~model.model_args.attn_implementation" \
  task_name=SMOKE_GA forget_split=forget10 retain_split=retain90 trainer=GradAscent
```

zsh 里 `~` 必须加引号。云主机默认若是 **bash**，可直接写 `~model.model_args.attn_implementation`。

### 不推荐但可试：沿用镜像 PyTorch 2.8

只在「装 2.4.1 失败」时用：跳过 `requirements.txt` 里的 `torch==2.4.1`，保留镜像 2.8，再装 transformers / hydra / deepspeed / bitsandbytes。DeepSpeed、flash-attn、bnb **版本可能都要对 2.8 升级**，数字更难和 `docs/repro.md` 对齐，只适合先打通流程。

装完自检：

```bash
python -c "import torch; assert torch.cuda.device_count()>=2; print(torch.__version__, torch.cuda.get_device_name(0))"
python -c "import transformers, accelerate, deepspeed, datasets; print('ok')"
```

---

## 四、必须下载的数据与对照 log

**没有 retain 评测 log，`forget_quality`、`privleak` 等相对指标会失败或无意义。**

英文 README 写的是 `python setup_data.py --eval`。本仓库脚本实际参数是：

```bash
cd $DATA/open-unlearning
python setup_data.py --eval_logs    # → saves/eval/ 下 TOFU / MUSE 的 retain、finetuned JSON
python setup_data.py --idk          # DPO / Idk 需要 data/idk.jsonl
# python setup_data.py --wmdp       # 仅跑 WMDP 时需要；会 wget zip 并用密码 unzip
python setup_data.py --help
```

跑完后应能看到类似路径（名称以实际下载为准）：

- `saves/eval/tofu_Llama-3.2-1B-Instruct_retain90/TOFU_EVAL.json`
- `saves/eval/muse_Llama-2-7b-hf_News_retrain/MUSE_EVAL.json`（MUSE 脚本里用的是这个命名）

训练时把 `retain_logs_path=` 指到对应 JSON。

首次训练还会从 Hub **拉数据集和 target 模型**（例如 `open-unlearning/tofu_Llama-3.2-1B-Instruct_full`），请保证 `HF_HOME` 在数据盘。

---

## 五、磁盘怎么规划

粗略体量（数量级，含优化器状态时 checkpoint 会更大）：

| 内容 | 约占用 |
|------|--------|
| 评测 log（`--eval_logs`） | 较小（JSON） |
| Llama-3.2-1B target + 一次遗忘 ckpt | 数 GB～十几 GB |
| Llama-3.2-3B | 十几 GB 级 |
| Llama-3.1-8B / Llama-2-7B | 数十 GB 级（含 ZeRO 分片与保存） |
| `scripts/tofu_unlearn.sh` 全量 | **很容易超过 50GB，可能上百 GB** |

建议：

- 系统盘只放系统；`$HF_HOME`、仓库、`saves/` 全部在 **250GB 数据盘**
- 先只跑 **1B + SimNPO + forget10**（论文综合排名第一），不要一上来跑全量脚本
- 需要对照 7B 表时再下 Llama-2 / 8B，并随时删旧 `saves/unlearn/*`

若 `saves/` 仍写到仓库内，可在命令里覆盖：`paths.output_dir=$DATA/saves/unlearn/YOUR_TASK`。

---

## 六、推荐实验顺序（优秀方法优先）

默认 accelerate 已是 **2 进程 DeepSpeed ZeRO-3**，与双卡匹配，**不必改卡数**。脚本里 `per_device_train_batch_size=4`、`gradient_accumulation_steps=4`、2 卡 → 有效 batch **32**（以本仓库脚本为准）。

**排序依据：** OpenUnlearning 论文 Table 3（TOFU 上调参后，记忆 / 隐私 / 效用的调和平均）。**从高到低复现**，朴素基线放到最后。

| 优先级 | 方法 | 论文 Agg. | 本仓库怎么跑 | 说明 |
|--------|------|-----------|--------------|------|
| 1 | **SimNPO** | 0.53（第一） | `trainer=SimNPO` | 效用几乎不掉，综合最好；默认 yaml 已带论文常用 `beta=4.5` |
| 2 | **RMU** | 0.52 | `trainer=RMU` | 遗忘强，效用往往掉；层号等要按模型核对 |
| 3 | **UNDIAL** | 0.42 | `trainer=UNDIAL` | 已注册；社区 `community/methods/UNDIAL/run.sh` 含调参网格 |
| 4 | **AltPO** | 0.15 | `community/methods/AltPO/run.sh` | 不在 `TRAINER_REGISTRY`，走社区脚本 |
| 5 | **NPO** | 0.15 | `trainer=NPO` | 默认超参下 TOFU 的 `forget_quality` 往往最好（见 [复现结果](./repro.md)） |
| 6 | **IdkDPO** | 0.14 | `trainer=DPO` + `experiment=unlearn/tofu/idk.yaml` | 需 `setup_data.py --idk` |
| 7 | **GradDiff** | 9e-3 | `trainer=GradDiff` | 容易过遗忘 |
| 8 | **GradAscent** | 未进该表 | `trainer=GradAscent` | 朴素基线，forget10 上常把 utility 打到 0 |

论文也写了：若**不把隐私算进综合分**、只看记忆+效用，GradDiff 会排到前面——那是过遗忘，不是推荐优先复现的「好方法」。

PDU / SatImp / WGA / CE-U 未进入 Table 3；核心方法跑完再按社区 README 补。

> **默认超参 vs 论文调参：** `docs/repro.md` 未调参时，TOFU forget10 上 **NPO** 的 forget_quality 明显高于 SimNPO。若目标是「对齐未调参表」可把 NPO 提前；若目标是「复现论文认定的最强方法」，仍从 **SimNPO** 开始。

### 1）先复现 SimNPO（TOFU 1B + forget10）

```bash
cd $DATA/open-unlearning
conda activate unlearning

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
  --config_file configs/accelerate/default_config.yaml --main_process_port $MASTER_PORT \
  src/train.py --config-name=unlearn.yaml experiment=unlearn/tofu/default \
  trainer=SimNPO \
  task_name=tofu_1B_SimNPO_forget10 \
  model=Llama-3.2-1B-Instruct \
  forget_split=forget10 retain_split=retain90 \
  model.model_args.pretrained_model_name_or_path=open-unlearning/tofu_Llama-3.2-1B-Instruct_full \
  retain_logs_path=saves/eval/tofu_Llama-3.2-1B-Instruct_retain90/TOFU_EVAL.json \
  trainer.args.per_device_train_batch_size=4 \
  trainer.args.gradient_accumulation_steps=4 \
  trainer.args.ddp_find_unused_parameters=true \
  trainer.args.gradient_checkpointing=true
```

**多卡训练过程中不能跑自定义 TOFU eval。** 训完后单卡评测（把 `SimNPO` / `task_name` 换成后面各方法即可）：

```bash
CUDA_VISIBLE_DEVICES=0 python src/eval.py experiment=eval/tofu/default.yaml \
  forget_split=forget10 holdout_split=holdout10 \
  model=Llama-3.2-1B-Instruct \
  task_name=tofu_1B_SimNPO_forget10 \
  model.model_args.pretrained_model_name_or_path=saves/unlearn/tofu_1B_SimNPO_forget10 \
  paths.output_dir=saves/unlearn/tofu_1B_SimNPO_forget10/evals \
  retain_logs_path=saves/eval/tofu_Llama-3.2-1B-Instruct_retain90/TOFU_EVAL.json
```

对照 [复现结果](./repro.md) 里 **Llama-3.2-1B-Instruct / forget10 / SimNPO** 的量级。能出 `TOFU_SUMMARY.json` 即流水线通了。

### 2）按优先级接着换方法（同一 1B / forget10）

把上面训练命令里的 `trainer=` 和 `task_name=` 按表替换：

```bash
# 2. RMU
trainer=RMU  task_name=tofu_1B_RMU_forget10

# 3. UNDIAL
trainer=UNDIAL  task_name=tofu_1B_UNDIAL_forget10

# 4. AltPO（社区脚本，含调参网格）
bash community/methods/AltPO/run.sh

# 5. NPO
trainer=NPO  task_name=tofu_1B_NPO_forget10

# 6. IdkDPO（先 python setup_data.py --idk）
# 并把 experiment=unlearn/tofu/default 改成：
experiment=unlearn/tofu/idk.yaml trainer=DPO task_name=tofu_1B_DPO_forget10

# 7. GradDiff
trainer=GradDiff  task_name=tofu_1B_GradDiff_forget10

# 8. GradAscent（最后做基线）
trainer=GradAscent  task_name=tofu_1B_GA_forget10
```

RMU 的 `module_regex` 默认指向 `model.layers.7`，1B/3B/8B 层数不同，换模型时要改。NPO 会再拷一份参考模型，双卡 A800 80GB 一般够。

### 3）同一最强方法再放大规模

SimNPO（以及需要时的 RMU / NPO）在 1B forget10 跑通后：

- 同一方法：`forget05` / `forget01`（`retain_logs_path` 改成 retain95 / retain99）
- 再换 `model=Llama-3.2-3B-Instruct` 或 `Llama-3.1-8B-Instruct`，以及对应 `tofu_${model}_full`
- MUSE：确认 `saves/eval/muse_*_retrain/MUSE_EVAL.json` 后，优先 `trainer=SimNPO` / `NPO`，`experiment=unlearn/muse/default`，`data_split=News` 或 `Books`

全量循环磁盘很大，不要整份跑：

```bash
bash scripts/tofu_unlearn.sh    # 内含 GA/GradDiff/NPO/DPO/RMU，无 SimNPO
bash scripts/muse_unlearn.sh
```

`tofu_unlearn.sh` **没有 SimNPO**。要复现论文第一名，用本节命令或自己把 `SimNPO` 加进脚本循环。

---

## 七、方法与基准清单（复现前选范围）

复现顺序见上一节（SimNPO → RMU → UNDIAL → … → GradAscent）。已注册 trainer 见 `src/trainer/__init__.py`；AltPO 仅在 `community/methods/AltPO/`。

| 想复现 | 最少准备 |
|--------|----------|
| TOFU 遗忘 + 官方指标 | `--eval_logs` + Hub 上 `tofu_*_full` + 双卡训练 + 单卡 `src/eval.py` |
| TOFU + DPO/Idk | 上一项 + `--idk` |
| MUSE | `--eval_logs` 中 MUSE 部分 + `data_split=News` 或 `Books` |
| WMDP | `--wmdp` + `pip install ".[lm-eval]"` + 对应 experiment |
| 自己当 target | 先 `scripts/tofu_finetune.sh` 或 `experiment=finetune/tofu/default`，再遗忘 |

---

## 八、常见失败

| 现象 | 处理 |
|------|------|
| 系统盘 100% | `HF_HOME`、`saves` 改到数据盘；清缓存 |
| `forget_quality` 异常 / 缺文件 | 检查 `retain_logs_path` 是否指向已下载的 JSON，且 split 配对（forget10↔retain90） |
| flash-attn 编译失败 | 去掉 `attn_implementation`，或换与 CUDA 匹配的预编译轮子 |
| DeepSpeed / bitsandbytes 报错 | 不要混用镜像 torch 2.8 与钉死的 2.4.1 生态 |
| 多卡训练里 eval 被跳过 | 属预期；训完单卡 `src/eval.py` |
| NPO / SimNPO OOM | A800 80GB 一般够；若仍 OOM，减小 `per_device_train_batch_size` 并加大 `gradient_accumulation_steps`，保持有效 batch 32 |
| 数字和表对不齐 | 换卡、换 torch、单卡都会漂；先保证有效 batch、epoch、lr 与脚本一致 |

---

## 九、出门清单（复制勾选）

- [ ] 数据盘已挂载；`HF_HOME`、仓库、`saves` 不在 30GB 系统盘
- [ ] `nvidia-smi` 两张 A800；独立环境里 `torch.cuda.device_count()==2`
- [ ] 尽量 **Python 3.11 + torch 2.4.1（CUDA 轮子）**，而不是直接覆盖镜像 2.8
- [ ] `flash-attn` 已装，或已准备好去掉 flash attention 的命令
- [ ] `python setup_data.py --eval_logs`（DPO 再加 `--idk`）
- [ ] Hub 能下 `open-unlearning/tofu_Llama-3.2-1B-Instruct_full`
- [ ] 先完成 1B **SimNPO** forget10 训练 + 单卡评测，再按论文排名做 RMU → UNDIAL → … → GradAscent
- [ ] 全量脚本前估算磁盘；按需删减 `scripts/tofu_unlearn.sh` 循环
