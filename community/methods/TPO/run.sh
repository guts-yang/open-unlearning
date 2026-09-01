#!/bin/bash
# TPO (Targeted Preference Optimization) on TOFU, aligned with this repo's benchmark pipeline.
# Paper: "Not All Tokens Are Meant to Be Forgotten" (AAAI 2026, arXiv:2506.03142)
# Official code: https://github.com/guts-yang/Unlearning-TPO
#
# Prerequisites:
#   1. Retain-model eval logs (for forget_quality / privleak):
#        python setup_data.py --eval_logs
#      -> saves/eval/tofu_Llama-3.2-1B-Instruct_retain{90,95,99}/TOFU_EVAL.json
#   2. Annotated forget data is vendored under community/methods/TPO/data/ (see README.md).

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

########################################################################################################################
########################################### Hyper parameter tuning #####################################################
########################################################################################################################

# Optional: search grid for beta (no official 1B values; official 3B range is 0.19-0.30).
# Uncomment the two lines below to grid-search beta instead of running the final values.
# betas_search=(0.19 0.23 0.27 0.30 0.32)
# use_search_grid=true

########################################################################################################################
########################################### Final best parameters #####################################################
########################################################################################################################

# Required to replicate your results.
# Official best beta values (paper, Llama-3.2-3B reference; Llama-2-7B uses 0.32/0.32/0.23):
#   forget01=0.30, forget05=0.27, forget10=0.19
# UW/GW annotation source: 'gpt' (official default) or 'bert'.
classifier=gpt

per_device_train_batch_size=16
gradient_accumulation_steps=2   # effective batch 32 = official 4 x 4 x 2 GPUs
learning_rate=1e-5
num_train_epochs=10

models=(
    "Llama-3.2-1B-Instruct"
)
forget_retain_splits=(
    "forget01 retain99 0.30"
    "forget05 retain95 0.27"
    "forget10 retain90 0.19"
)

for split in "${forget_retain_splits[@]}"; do
    forget_split=$(echo $split | cut -d' ' -f1)
    retain_split=$(echo $split | cut -d' ' -f2)
    beta=$(echo $split | cut -d' ' -f3)

    if [ -n "${use_search_grid:-}" ]; then
        beta_list=("${betas_search[@]}")
    else
        beta_list=(${beta})
    fi

    for model in "${models[@]}"; do
        for beta in "${beta_list[@]}"; do
            task_name=tofu_${model}_${forget_split}_TPO_lr${learning_rate}_beta${beta}_${classifier}
            model_path=open-unlearning/tofu_${model}_full
            echo ${task_name}: Unlearning ${model_path} using TPO

            # Unlearn
            CUDA_VISIBLE_DEVICES=0 \
            python src/train.py --config-name=unlearn.yaml \
            experiment=unlearn/tofu/default.yaml \
            trainer=TPO \
            task_name=${task_name} \
            model=${model} \
            forget_split=${forget_split} \
            retain_split=${retain_split} \
            model.model_args.pretrained_model_name_or_path=${model_path} \
            retain_logs_path=saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json \
            trainer.args.per_device_train_batch_size=$per_device_train_batch_size \
            trainer.args.gradient_accumulation_steps=$gradient_accumulation_steps \
            trainer.args.learning_rate=$learning_rate \
            trainer.args.num_train_epochs=$num_train_epochs \
            trainer.args.eval_strategy=no \
            trainer.args.eval_on_start=False \
            trainer.method_args.beta=$beta \
            trainer.method_args.alpha=0.0 \
            data.forget.TOFU_QA_forget.handler=QAwithCommonWordsDataset \
            ~data.forget.TOFU_QA_forget.args.hf_args.name \
            data.forget.TOFU_QA_forget.args.hf_args.path=json \
            +data.forget.TOFU_QA_forget.args.hf_args.data_files=community/methods/TPO/data/${forget_split}_with_common_words_${classifier}.json \
            data.forget.TOFU_QA_forget.args.hf_args.split=train

            # Eval
            CUDA_VISIBLE_DEVICES=0 python src/eval.py \
            experiment=eval/tofu/default.yaml \
            forget_split=${forget_split} \
            model=${model} \
            task_name=${task_name} \
            model.model_args.pretrained_model_name_or_path=saves/unlearn/${task_name} \
            paths.output_dir=saves/unlearn/${task_name}/evals \
            retain_logs_path=saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json
        done
    done
done
