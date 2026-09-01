# 复现准备（本机：AutoDL 内蒙 DC2 / nm-A1）

对应问题：在本仓库复现遗忘方法前要做什么；以及**这台容器当前实测**够不够。

本文按**本仓库实际入口**书写（`src/train.py`、`setup_data.py`、`scripts/tofu_unlearn.sh`）。对照数字见 [复现结果](./repro.md)；Hydra 覆盖见 [实验配置与运行](./experiments.md)。

盘点时间：**2026-08-26**。换实例或开机挂卡后，先重跑「一、上机检查」，不要沿用过期数字。

---

## 结论（先看这张表）

| 项目 | 官方复现 | 本机实测（2026-08-26） | 判断 |
|------|----------|------------------------|------|
| GPU | 2× L40s 48GB | **当前未挂卡**：无 `/dev/nvidia*`，`nvidia-smi` 无输出，`torch.cuda.is_available()==False`。驱动 `580.105.08` 在。同一 AutoDL 账号上 `Unlearn-Simple` 的历史记录是 **2× A800 80GB** | **先在控制台开机挂卡**，再 `nvidia-smi -L` 确认数量与型号 |
| 分布式 | Accelerate + DeepSpeed ZeRO-3，`num_processes: 2` | 默认 [`configs/accelerate/default_config.yaml`](../../configs/accelerate/default_config.yaml) 仍是 **2 进程** | 挂上 **2 卡** 才能直接用默认配置；1 卡必须改 `num_processes` 并重算有效 batch |
| CPU / 内存（关卡） | — | cgroup：**0.5 核、2GB RAM**（AutoDL 关 GPU 省钱） | 关卡时 **不能训、不能评**；开机后配额会放开 |
| CPU / 内存（宿主机可见） | — | `lscpu` 112 线程 Xeon Gold 6348；`free` 约 **1.0Ti** | 仅作参考；以开机后 cgroup / `nvidia-smi` 为准 |
| Python | `>=3.11`，文档示例 3.11 | conda **base：3.12.3**（`/root/miniconda3`），**没有** `unlearning` 环境，也没有 `python3.11` | 3.12 满足下限；**不要**在 base 里直接 `pip install -e .`（会冲掉下面这套已有栈） |
| PyTorch | 钉死 `torch==2.4.1` | 镜像 **2.8.0+cu128**；`nvcc` 在 `/usr/local/cuda/bin`（**不在默认 PATH**），CUDA toolkit **12.8** | 本机更稳的路径是 **沿用 2.8 + 已下好的 flash-attn 轮子**；硬装 2.4.1 没有现成 cu128 轮子，且会和已有实验环境冲突 |
| 系统盘 `/` | — | overlay **30GB**，已用约 4.9GB | **不够放模型和 checkpoint** |
| 数据盘 | — | **`/root/autodl-tmp`：300GB**，已用约 64GB，剩余约 **237GB** | 代码、HF 缓存、`saves/` **必须**在这里 |
| `/dev/shm` | — | **2.0GB** | DeepSpeed 多卡有时会踩共享内存；OOM / bus error 时再查这项 |
| 网络 GitHub | — | HTTPS 不稳定：`curl 16` HTTP/2、TLS 中断；已设 `git config --global http.version HTTP/1.1` | 全量 clone 易失败；用 `--depth 1` 或代理 |
| 网络 Hugging Face | — | `huggingface.co` SSL reset；**`https://hf-mirror.com` 可用** | 每个新 shell 都要 `source /root/autodl-tmp/env_hf.sh` |

**显存结论（以历史 2×A800 80GB 计）：** 挂上双卡后，仓库里的 TOFU（含 8B）和 MUSE（Llama-2 7B）默认遗忘脚本显存一般够。不要一上来跑完整 `scripts/tofu_unlearn.sh`（3 模型 × 5 方法 × 3 划分 ≈ 45 次训练+评测），磁盘和时间都会爆。先做下方「冒烟 / SimNPO 1B」。

**当前最大阻塞：** GPU 未挂载。未开机前，后面所有训练命令都会失败。

**不能对齐 `docs/repro.md` 数字的原因（即使卡更强）：** 换 GPU / CUDA / 是否 ZeRO-3 / 单卡还是双卡 / **torch 2.8 vs 2.4.1**，官方也写明结果会漂。本机目标应是：**流程跑通、指标量级合理**；逐位对齐请尽量复刻 `torch==2.4.1` + 双卡 DeepSpeed + 脚本里的 batch（本机不优先走这条）。

---

## 本机已经有的东西（不要重复下、不要乱覆盖）

| 路径 / 状态 | 说明 |
|-------------|------|
| `/usr/local/open-unlearning` | **本仓库唯一副本**（`repro/ou-table3` @ `e9e9ab2`）。位于 30GB 系统盘（剩约 16GB），所以**代码之外的一切都要写到 `/root/autodl-tmp`**：`HF_HOME`、`saves/`、`logs/`、每条命令的 `paths.output_dir`。数据盘上的重复副本 `/root/autodl-tmp/open-unlearning` 已于 2026-09-01 删除（删前已确认零独有内容），**不要再克隆第二份** |
| `/usr/local/Unlearn-Simple` | 另一套旧实验代码；conda **base** 正在给它用（`transformers==4.46.3`、`torch==2.8.0+cu128`） |
| `/root/autodl-tmp/env_hf.sh` | HF 镜像、缓存路径、token、修正 `OMP_NUM_THREADS` |
| `/root/autodl-tmp/huggingface` | `HF_HOME`，约 **51GB**。已有 `locuslab/TOFU` 数据集、`NousResearch/Llama-2-7b*`、`locuslab/tofu_ft_llama2-7b`。**没有** `open-unlearning/tofu_Llama-3.2-1B-Instruct_*`，首次跑本仓库 1B 仍要下模型 |
| `/root/autodl-tmp/flash_attn-2.8.3.post1+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl` | 给 **Python 3.12 + torch 2.8** 的预编译轮子（GitHub 直连很慢） |
| `/root/autodl-tmp/paper_models/` | 约 13GB 旧 NPO 产出，与本仓库目录约定无关；磁盘紧时再删 |
| Hugging Face token | `env_hf.sh` 从 `Unlearn-Simple/.env` 的 `HuggingFace_token=` 读取；数据盘也有 `huggingface/token`。`huggingface-cli whoami` 在未 source 时显示未登录，属预期 |

---

## 一、上机后立刻做的检查

```bash
# 1) AutoDL 关卡时下面会失败或为空，先去控制台开机
nvidia-smi -L
nvidia-smi

python3 --version           # 本机：3.12.3
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
# 关卡时期望：2.8.0+cu128 False 0
# 开机后期望：True，且 device_count 与 nvidia-smi 一致（历史为 2）

df -h / /root/autodl-tmp
echo "HF_HOME=$HF_HOME HF_ENDPOINT=$HF_ENDPOINT"
echo $PATH | tr : '\n' | grep cuda || echo "nvcc 不在 PATH，需要 export PATH=/usr/local/cuda/bin:\$PATH"
```

**数据盘挂载点（已确认，不要再用 `/data` 或 `/workspace`）：**

```bash
export DATA=/root/autodl-tmp
source $DATA/env_hf.sh          # 每个新 shell 必做：HF 镜像 + HF_HOME + token + OMP

mkdir -p $DATA/open-unlearning $DATA/saves $DATA/envs
```

`env_hf.sh` 会设置：

- `HF_ENDPOINT=https://hf-mirror.com`（直连 Hub 会 SSL reset）
- `HF_HOME=/root/autodl-tmp/huggingface`
- `HUGGINGFACE_HUB_CACHE=$HF_HOME/hub`（必须是 `hub` 子目录，否则会绕开已有 51GB 缓存）
- `OMP_NUM_THREADS=8`（覆盖镜像里过小的 OpenMP 值）

### 仓库位置：唯一副本 `/usr/local/open-unlearning`

数据盘上的重复副本已于 2026-09-01 删除（删前已确认两份 tracked 文件零差异、且无独有未跟踪文件）。**不要再克隆第二份**——两份副本必然导致「改 A 跑 B」。

一律：

```bash
cd /usr/local/open-unlearning
export DATA=/root/autodl-tmp       # 只放数据与产物，不放代码
```

因为仓库在系统盘（剩约 16GB），**代码之外的一切都写到 `$DATA`**：`HF_HOME=$DATA/huggingface`、`saves/`（仓库内 `saves` 已软链到 `$DATA/saves`）、`logs/`、以及每条命令里的 `paths.output_dir=$DATA/saves/...`。

浅克隆若以后需要完整历史：

```bash
git fetch --unshallow    # GitHub 不稳时可能再次 TLS 失败，多试或走代理
```

---

## 二、账号与网络

1. **Hugging Face（走镜像）**
   - 评测 log：`open-unlearning/eval`（dataset）
   - 目标模型：`open-unlearning/tofu_*_full`、`muse-bench/MUSE-News_target` 等
   - 数据集：`locuslab/TOFU`（本机缓存里已有）、`muse-bench/MUSE-News` / `MUSE-Books`
   - 直接用 `open-unlearning/` 上已微调的 TOFU 模型，一般**不必** Llama 许可
   - **不要** `pip install -U huggingface_hub` 装到 base：Unlearn-Simple 依赖 `huggingface_hub` 0.36.x；本仓库 `requirements.txt` 也是 `huggingface-hub==0.36.0`

```bash
source /root/autodl-tmp/env_hf.sh
# 需要时再：huggingface-cli login   # token 多数情况已由 env_hf.sh 注入
```

2. GitHub：`git clone` 全量曾报 `RPC failed; curl 16 Error in the HTTP2 framing layer`。已全局 `http.version=HTTP/1.1`。必要时浅克隆或 `ghproxy.net`。
3. `setup_data.py --wmdp` 还要用 `wget` 拉 S3，并需要 `unzip`（本机 `/usr/bin/wget`、`/usr/bin/unzip` 都在）。S3 是否可达未在关卡状态下测通。
4. 可选：`wandb login`（配置里若开了 wandb）。

---

## 三、Python 环境（按本机改）

镜像是 **PyTorch 2.8.0+cu128 + Python 3.12 + CUDA 12.8**。本仓库 `requirements.txt` 钉的是 **`torch==2.4.1`、`deepspeed==0.15.4`、`bitsandbytes==0.44.1`、`accelerate==0.34.2`、`transformers==4.51.3`**。

conda 只有 **base**，且 base 已给 `/usr/local/Unlearn-Simple` 使用（实测 `transformers==4.46.3`、`accelerate==1.14.0`）。清华 conda 频道在本机 `conda search python=3.11` 会报 channel 不可用，**不要指望轻松新建 3.11 conda 环境**。

### 推荐：数据盘上单独 venv（Python 3.12），保留镜像 torch 2.8

只在「对齐 `docs/repro.md` 逐位数字」失败、且你愿意重建环境时，再考虑 torch 2.4.1。

```bash
source /root/autodl-tmp/env_hf.sh
export DATA=/root/autodl-tmp
cd /usr/local/open-unlearning

python3 -m venv $DATA/envs/unlearning
source $DATA/envs/unlearning/bin/activate

# 先把镜像同款 torch 装进 venv（版本以开机后 `python -c "import torch; print(torch.__version__)"` 为准）
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128

# 跳过 requirements 里的 torch 钉死，再装其余依赖
grep -v '^torch==' requirements.txt > /tmp/req-notorch.txt
pip install -r /tmp/req-notorch.txt
pip install -e ".[lm-eval]" --no-deps   # 或按报缺的包补；避免 pip 把 torch 降回 2.4.1

# flash-attn：用已经下好的轮子，不要现场编译（PATH 里默认没有 nvcc）
pip install /root/autodl-tmp/flash_attn-2.8.3.post1+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
```

装完自检（**必须在 GPU 已挂载**时）：

```bash
source /root/autodl-tmp/envs/unlearning/bin/activate
source /root/autodl-tmp/env_hf.sh
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
python -c "import transformers, accelerate, deepspeed, datasets; print('ok')"
python /root/autodl-tmp/verify_flash_attn.py
```

`flash-attn` 编译失败或未装时，可先去掉 flash attention 跑 1B 冒烟：

```bash
python src/train.py --config-name=unlearn.yaml experiment=unlearn/tofu/default \
  "~model.model_args.attn_implementation" \
  task_name=SMOKE_GA forget_split=forget10 retain_split=retain90 trainer=GradAscent
```

本机默认 **bash**，`~model...` 可不加引号；zsh 必须加引号。

### 不推荐：在 conda base 里 `pip install -e .`

会把 Unlearn-Simple 的 `transformers 4.46.3` 等升级/降级掉。两套代码共用 base 迟早互相踩脚。

### 更接近官方、但本机成本高：Python 3.11 + torch 2.4.1

需要自己解决：conda 3.11 频道、cu124/cu121 轮子与本机 **CUDA 12.8 驱动** 是否兼容、以及 **没有** 对应 flash-attn 现成轮子（现成的是 cp312 + torch2.8）。只在必须对齐官方表时再做。

---

## 四、必须下载的数据与对照 log

**没有 retain 评测 log，`forget_quality`、`privleak` 等相对指标会失败或无意义。**

英文 README 写的是 `python setup_data.py --eval`。本仓库脚本实际参数是：

```bash
source /root/autodl-tmp/env_hf.sh
cd /usr/local/open-unlearning
python setup_data.py --eval_logs    # → saves/eval/ 下 TOFU / MUSE 的 retain、finetuned JSON
python setup_data.py --idk          # DPO / Idk 需要 data/idk.jsonl
# python setup_data.py --wmdp       # 仅跑 WMDP 时需要；会 wget zip 并用密码 unzip
python setup_data.py --help
```

跑完后应能看到类似路径（名称以实际下载为准）：

- `saves/eval/tofu_Llama-3.2-1B-Instruct_retain90/TOFU_EVAL.json`
- `saves/eval/muse_Llama-2-7b-hf_News_retrain/MUSE_EVAL.json`（MUSE 脚本里用的是这个命名）

训练时把 `retain_logs_path=` 指到对应 JSON。建议同时覆盖输出目录到数据盘，例如 `paths.output_dir=$DATA/saves/unlearn/YOUR_TASK`。

首次训练还会从 Hub **拉 target 模型**（例如 `open-unlearning/tofu_Llama-3.2-1B-Instruct_full`）。本机 HF 缓存里目前 **没有** 这些 `open-unlearning/tofu_*` 权重，请保证已 `source env_hf.sh`。

---

## 五、磁盘怎么规划

粗略体量（数量级，含优化器状态时 checkpoint 会更大）：

| 内容 | 约占用 | 本机 |
|------|--------|------|
| 评测 log（`--eval_logs`） | 较小（JSON） | 尚未下载 |
| Llama-3.2-1B target + 一次遗忘 ckpt | 数 GB～十几 GB | 需新下载 |
| Llama-3.2-3B | 十几 GB 级 | — |
| Llama-3.1-8B / Llama-2-7B | 数十 GB 级 | Llama-2 相关权重缓存已有一部分（给旧仓库用） |
| `scripts/tofu_unlearn.sh` 全量 | **很容易超过剩余空间里的安全余量** | 数据盘剩约 237GB，全量仍不建议一次跑完 |
| 已占用 | — | HF 缓存 ~51GB + `paper_models` ~13GB + 系统盘仓库 |

建议：

- 系统盘只放系统；`$HF_HOME`、仓库、`saves/` 全部在 **`/root/autodl-tmp`**
- 先只跑 **1B + SimNPO + forget10**，不要一上来跑全量脚本
- 需要对照 7B 表时再下 Llama-2 / 8B 的 **open-unlearning** target（不要和缓存里的 `locuslab/tofu_ft_llama2-7b` 混用路径）
- 随时删旧 `saves/unlearn/*`；磁盘紧时再考虑删 `/root/autodl-tmp/paper_models`

若 `saves/` 仍写到仓库内，命令里覆盖：`paths.output_dir=$DATA/saves/unlearn/YOUR_TASK`。

---

## 六、推荐实验顺序（优秀方法优先）

**先确认 `nvidia-smi` 为 2 张卡**，再使用默认 accelerate（2 进程 DeepSpeed ZeRO-3）。脚本里 `per_device_train_batch_size=4`、`gradient_accumulation_steps=4`、2 卡 → 有效 batch **32**（以本仓库脚本为准）。若开机后只有 1 卡：把 `num_processes` 改为 1，并把 `gradient_accumulation_steps` 加倍以保持有效 batch 32。

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
source /root/autodl-tmp/env_hf.sh
source /root/autodl-tmp/envs/unlearning/bin/activate
cd /usr/local/open-unlearning

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
export PATH=/usr/local/cuda/bin:$PATH

CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
  --config_file configs/accelerate/default_config.yaml --main_process_port $MASTER_PORT \
  src/train.py --config-name=unlearn.yaml experiment=unlearn/tofu/default \
  trainer=SimNPO \
  task_name=tofu_1B_SimNPO_forget10 \
  model=Llama-3.2-1B-Instruct \
  forget_split=forget10 retain_split=retain90 \
  model.model_args.pretrained_model_name_or_path=open-unlearning/tofu_Llama-3.2-1B-Instruct_full \
  retain_logs_path=saves/eval/tofu_Llama-3.2-1B-Instruct_retain90/TOFU_EVAL.json \
  paths.output_dir=/root/autodl-tmp/saves/unlearn/tofu_1B_SimNPO_forget10 \
  trainer.args.per_device_train_batch_size=4 \
  trainer.args.gradient_accumulation_steps=4 \
  trainer.args.ddp_find_unused_parameters=true \
  trainer.args.gradient_checkpointing=true
```

长时间训练请用 AutoDL 文档里的 `screen` / `tmux`，避免 SSH 断开杀进程。

**多卡训练过程中不能跑自定义 TOFU eval。** 训完后单卡评测（把 `SimNPO` / `task_name` 换成后面各方法即可）：

```bash
CUDA_VISIBLE_DEVICES=0 python src/eval.py experiment=eval/tofu/default.yaml \
  forget_split=forget10 holdout_split=holdout10 \
  model=Llama-3.2-1B-Instruct \
  task_name=tofu_1B_SimNPO_forget10 \
  model.model_args.pretrained_model_name_or_path=/root/autodl-tmp/saves/unlearn/tofu_1B_SimNPO_forget10 \
  paths.output_dir=/root/autodl-tmp/saves/unlearn/tofu_1B_SimNPO_forget10/evals \
  retain_logs_path=saves/eval/tofu_Llama-3.2-1B-Instruct_retain90/TOFU_EVAL.json
```

对照 [复现结果](./repro.md) 里 **Llama-3.2-1B-Instruct / forget10 / SimNPO** 的量级。能出 `TOFU_SUMMARY.json` 即流水线通了。数字不必逐位相同。

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
| TOFU 遗忘 + 官方指标 | 开机双卡 + `--eval_logs` + Hub 上 `tofu_*_full` + 双卡训练 + 单卡 `src/eval.py` |
| TOFU + DPO/Idk | 上一项 + `--idk` |
| MUSE | `--eval_logs` 中 MUSE 部分 + `data_split=News` 或 `Books` |
| WMDP | `--wmdp` + `pip install ".[lm-eval]"` + 对应 experiment |
| 自己当 target | 先 `scripts/tofu_finetune.sh` 或 `experiment=finetune/tofu/default`，再遗忘 |

---

## 八、常见失败（含本机已踩过的）

| 现象 | 处理 |
|------|------|
| `nvidia-smi` 无输出 / `torch.cuda.is_available()==False` / 内存只有 2GB | AutoDL **关卡**。控制台开机挂 GPU 后再查 |
| 系统盘 100% | 仓库和 `HF_HOME`、`saves` 改到 `/root/autodl-tmp` |
| 删大文件后 `df` 不变 | AutoDL 回收站 `/root/autodl-tmp/.Trash-0/` 滞留（`du -sh /root/autodl-tmp/*` 看不到隐藏目录）。清空 `.Trash-0/files/` 才真正释放；批量删除超 500 条目会被 safe-delete 拦截，需 `ls \| xargs -n 200 rm -rf` 分批 |
| `huggingface.co` SSL reset / 超时 | `source /root/autodl-tmp/env_hf.sh`，不要直连 Hub |
| `git clone`：`curl 16` HTTP/2 或 GnuTLS -110 | `git config --global http.version HTTP/1.1`，改浅克隆 |
| `forget_quality` 异常 / 缺文件 | 检查 `retain_logs_path` 是否指向已下载的 JSON，且 split 配对（forget10↔retain90） |
| flash-attn 编译失败 | 用数据盘上现成 `.whl`；或去掉 `attn_implementation` |
| DeepSpeed / bitsandbytes 报错 | 不要在同一环境混用镜像 torch 2.8 与钉死的 2.4.1 |
| 多卡训练里 eval 被跳过 | 属预期；训完单卡 `src/eval.py` |
| NPO / SimNPO OOM | A800 80GB 一般够；若仍 OOM，减小 `per_device_train_batch_size` 并加大 `gradient_accumulation_steps`，保持有效 batch 32 |
| DeepSpeed 共享内存错误 | `/dev/shm` 本机仅 2GB，需按 AutoDL/Docker 方式加大 shm |
| 数字和表对不齐 | 换卡、换 torch 2.8、单卡都会漂；先保证有效 batch、epoch、lr 与脚本一致 |
| conda 新建 python=3.11 失败 | 本机清华频道 `conda search` 不可用；用 3.12 venv |
| 在 base 里装本仓库依赖 | 会破坏 Unlearn-Simple；用 `$DATA/envs/unlearning` |

---

## 九、出门清单（复制勾选）

- [ ] AutoDL **已开机挂卡**；`nvidia-smi -L` 能看到 GPU（历史预期 2× A800）；`torch.cuda.device_count()` 与卡数一致
- [ ] 每个新 shell：`source /root/autodl-tmp/env_hf.sh`
- [ ] 仓库唯一副本在 `/usr/local/open-unlearning`（数据盘重复副本已删除，不要再克隆第二份）；所有产物写到 `/root/autodl-tmp`
- [ ] 独立 venv `$DATA/envs/unlearning`，**没有**在 conda base 里覆盖 Unlearn-Simple
- [ ] `torch` 仍为 **2.8.x + CUDA**；`flash-attn` 用数据盘 wheel 或已准备去掉 flash attention 的命令
- [ ] `python setup_data.py --eval_logs`（DPO 再加 `--idk`）
- [ ] 镜像能下 `open-unlearning/tofu_Llama-3.2-1B-Instruct_full`（当前缓存里还没有）
- [ ] 先完成 1B **SimNPO** forget10 训练 + 单卡评测，再按论文排名做 RMU → UNDIAL → … → GradAscent
- [ ] 全量脚本前看 `df -h /root/autodl-tmp`；按需删减 `scripts/tofu_unlearn.sh` 循环
- [ ] 长时间任务用 `screen`/`tmux`
