#!/bin/bash
# SimNPO 调参与复现脚本（TOFU · Llama-3.2-1B-Instruct · forget10）
#
# 完整分阶段搜索（冒烟→强度粗扫→学习率→停止点→保效用）由本仓库的
# scripts/tofu_tune_simnpo.sh 驱动，支持断点续跑与评测后自动删权重：
#   bash scripts/tofu_tune_simnpo.sh 0      # 单卡轨迹冒烟（复现默认）
#   bash scripts/tofu_tune_simnpo.sh a1     # beta×gamma 强度粗扫
#   bash scripts/tofu_tune_simnpo.sh a2     # 学习率（基于 a1 top-3）
#   bash scripts/tofu_tune_simnpo.sh b      # 停止点（top-3 单卡轨迹）
#   bash scripts/tofu_tune_simnpo.sh c      # 保效用（alpha/delta/KL）
#   bash scripts/tofu_tune_simnpo.sh final  # 最优配置 2 卡完整重跑
# 汇总：python scripts/summarize_tune_trials.py
#
# 下面给出符合 community 约定的两段式骨架：段一为代表性粗扫网格，段二为最终最优配置复现命令。

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

MODEL="Llama-3.2-1B-Instruct"
FORGET_SPLIT="forget10"
RETAIN_SPLIT="retain90"
HOLDOUT_SPLIT="holdout10"
MODEL_HF="open-unlearning/tofu_${MODEL}_full"
RETAIN_LOG="saves/eval/tofu_${MODEL}_${RETAIN_SPLIT}/TOFU_EVAL.json"

########################################################################################################################
########################################### Hyper parameter tuning #####################################################
########################################################################################################################

betas=(0.5 1.0 2.5 4.5)
gammas=(0.125 1.0)
lrs=(1e-5 2e-5 5e-5)

for beta in "${betas[@]}"; do
  for gamma in "${gammas[@]}"; do
    for lr in "${lrs[@]}"; do
      task_name="tofu_${MODEL}_${FORGET_SPLIT}_SimNPO_b${beta}_g${gamma}_a1.0_d0.0_lr${lr}_e10"
      ckpt="/root/autodl-tmp/saves/tune/simnpo_forget10/${task_name}"
      # 断点续跑：已评测则跳过
      [ -f "$ckpt/evals/TOFU_SUMMARY.json" ] && { echo "[SKIP] $task_name"; continue; }

      # 2 卡终点训练（in-training eval 跳过，只看终点指标）
      CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
        --config_file configs/accelerate/default_config.yaml --main_process_port "$MASTER_PORT" \
        src/train.py --config-name=unlearn.yaml experiment=unlearn/tofu/default \
        trainer=SimNPO task_name="${task_name}" model="${MODEL}" \
        forget_split="${FORGET_SPLIT}" retain_split="${RETAIN_SPLIT}" \
        model.model_args.pretrained_model_name_or_path="${MODEL_HF}" \
        model.tokenizer_args.pretrained_model_name_or_path="${MODEL_HF}" \
        retain_logs_path="${RETAIN_LOG}" paths.output_dir="${ckpt}" \
        trainer.args.per_device_train_batch_size=4 \
        trainer.args.gradient_accumulation_steps=4 \
        trainer.args.ddp_find_unused_parameters=true \
        trainer.args.gradient_checkpointing=true \
        trainer.args.eval_strategy=no trainer.args.eval_on_start=False \
        trainer.args.num_train_epochs=10 trainer.args.learning_rate="${lr}" \
        trainer.method_args.beta="${beta}" trainer.method_args.gamma="${gamma}" \
        trainer.method_args.alpha=1.0 trainer.method_args.delta=0.0 \
        trainer.method_args.retain_loss_type=NLL

      # 终点评测
      CUDA_VISIBLE_DEVICES=0 python src/eval.py experiment=eval/tofu/default.yaml \
        forget_split="${FORGET_SPLIT}" holdout_split="${HOLDOUT_SPLIT}" model="${MODEL}" \
        task_name="${task_name}" \
        model.model_args.pretrained_model_name_or_path="${ckpt}" \
        model.tokenizer_args.pretrained_model_name_or_path="${MODEL_HF}" \
        paths.output_dir="${ckpt}/evals" retain_logs_path="${RETAIN_LOG}"

      # 评测后删权重（仅留 evals 与配置）
      rm -f "${ckpt}/model.safetensors"
      rm -rf "${ckpt}"/checkpoint-*
    done
  done
done

########################################################################################################################
########################################### Final best parameters #####################################################
########################################################################################################################

# TODO(final): 将下方超参替换为 scripts/summarize_tune_trials.py 选出的最优配置
# （Stage final 结束后回填；当前为占位：默认配置，仅用于演示复现链路）
BETA=4.5
GAMMA=0.125
ALPHA=1.0
DELTA=0.0
LR=1e-5
BEST_TASK="tofu_${MODEL}_${FORGET_SPLIT}_SimNPO_b${BETA}_g${GAMMA}_a${ALPHA}_d${DELTA}_lr${LR}_e10"
BEST_CKPT="/root/autodl-tmp/saves/tune/simnpo_forget10/${BEST_TASK}"

CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
  --config_file configs/accelerate/default_config.yaml --main_process_port "$MASTER_PORT" \
  src/train.py --config-name=unlearn.yaml experiment=unlearn/tofu/default \
  trainer=SimNPO task_name="${BEST_TASK}" model="${MODEL}" \
  forget_split="${FORGET_SPLIT}" retain_split="${RETAIN_SPLIT}" \
  model.model_args.pretrained_model_name_or_path="${MODEL_HF}" \
  model.tokenizer_args.pretrained_model_name_or_path="${MODEL_HF}" \
  retain_logs_path="${RETAIN_LOG}" paths.output_dir="${BEST_CKPT}" \
  trainer.args.per_device_train_batch_size=4 \
  trainer.args.gradient_accumulation_steps=4 \
  trainer.args.ddp_find_unused_parameters=true \
  trainer.args.gradient_checkpointing=true \
  trainer.args.eval_strategy=no trainer.args.eval_on_start=False \
  trainer.args.num_train_epochs=10 trainer.args.learning_rate="${LR}" \
  trainer.method_args.beta="${BETA}" trainer.method_args.gamma="${GAMMA}" \
  trainer.method_args.alpha="${ALPHA}" trainer.method_args.delta="${DELTA}" \
  trainer.method_args.retain_loss_type=NLL

CUDA_VISIBLE_DEVICES=0 python src/eval.py experiment=eval/tofu/default.yaml \
  forget_split="${FORGET_SPLIT}" holdout_split="${HOLDOUT_SPLIT}" model="${MODEL}" \
  task_name="${BEST_TASK}" \
  model.model_args.pretrained_model_name_or_path="${BEST_CKPT}" \
  model.tokenizer_args.pretrained_model_name_or_path="${MODEL_HF}" \
  paths.output_dir="${BEST_CKPT}/evals" retain_logs_path="${RETAIN_LOG}"

# 回写结果表（以 SimNPO-tuned 另开一行，与未调参 SimNPO 并列）
python scripts/record_tofu_result.py \
  --summary "${BEST_CKPT}/evals/TOFU_SUMMARY.json" \
  --method "SimNPO-tuned" \
  --forget-split "${FORGET_SPLIT}" \
  --model "${MODEL}" \
  --ckpt "${BEST_CKPT}"
