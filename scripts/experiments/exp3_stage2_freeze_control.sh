#!/usr/bin/env bash

set -euo pipefail

# Exp3: Stage2 freeze strategy control (freeze vs unfreeze stage1 backbone).
# Goal: check whether full fine-tuning improves ENSO skill over frozen-base perturbation training.

CUDA_GPUS="${CUDA_GPUS:-0,1,2,3}"
if [ -n "$CUDA_GPUS" ]; then
  export CUDA_VISIBLE_DEVICES="$CUDA_GPUS"
  NUM_GPUS=$(echo "$CUDA_GPUS" | tr ',' '\n' | wc -l)
else
  NUM_GPUS=1
fi

echo "[Exp3] Using GPUs: ${CUDA_GPUS} (num_gpus=${NUM_GPUS})"

SEED="${SEED:-1}"
LR="${LR:-2e-4}"
BATCH_SIZE="${BATCH_SIZE:-4}"
EPOCHS="${EPOCHS:-5}"
INPUT_STEPS="${INPUT_STEPS:-1}"
PREDICT_STEPS="${PREDICT_STEPS:-1}"
MAX_T="${MAX_T:-6}"
SAVE_EVAL_STEPS="${SAVE_EVAL_STEPS:-800}"

INPUT_VAR_LIST="${INPUT_VAR_LIST:-so thetao tos uo vo zos}"
DATA_DIR="${DATA_DIR:-./data/train_data/godas_split}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-./output/train_stage1/exp4/checkpoint-75}"
EXP_ROOT="${EXP_ROOT:-./output/train_stage2/exp_freeze_control}"

if [ ! -d "${BASE_MODEL_PATH}" ]; then
  echo "[Exp3] BASE_MODEL_PATH does not exist: ${BASE_MODEL_PATH}"
  exit 1
fi

run_one() {
  local freeze_flag="$1"
  local name="$2"
  local out_dir="${EXP_ROOT}/${name}"
  local dist_port=$((12345 + RANDOM % 12345))

  echo "[Exp3] Run setting=${name}, freeze_base_model=${freeze_flag}"

  torchrun --nproc_per_node="${NUM_GPUS}" train_stage2_GODAS.py \
    --max_t "${MAX_T}" \
    --atmo_var_list tauu tauv \
    --atmo_dims 2 \
    --ignore_mismatched_sizes True \
    --do_train \
    --dist_port "${dist_port}" \
    --data_dir "${DATA_DIR}" \
    --input_var_list ${INPUT_VAR_LIST} \
    --input_steps "${INPUT_STEPS}" \
    --predict_steps "${PREDICT_STEPS}" \
    --output_dir "${out_dir}" \
    --base_model_path "${BASE_MODEL_PATH}" \
    --freeze_base_model "${freeze_flag}" \
    --seed "${SEED}" \
    --report_to tensorboard \
    --log_level info \
    --logging_dir "${out_dir}/log" \
    --logging_steps 5 \
    --log_on_each_node False \
    --save_strategy steps \
    --save_steps "${SAVE_EVAL_STEPS}" \
    --save_total_limit 3 \
    --ddp_find_unused_parameters False \
    --num_train_epochs "${EPOCHS}" \
    --per_device_train_batch_size "${BATCH_SIZE}" \
    --per_device_eval_batch_size "${BATCH_SIZE}" \
    --gradient_accumulation_steps 1 \
    --dataloader_num_workers 8 \
    --gradient_checkpointing False \
    --learning_rate "${LR}" \
    --weight_decay 0.1 \
    --max_grad_norm 0.0 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --adam_epsilon 1e-6 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1
}

run_one True freeze_true
run_one False freeze_false

echo "[Exp3] Freeze strategy control completed."
