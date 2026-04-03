#!/usr/bin/env bash

set -euo pipefail

# Exp2: Data source control (CMIP6-trained Stage1 vs GODAS-trained Stage1).
# Goal: isolate domain shift effect from training source.

CUDA_GPUS="${CUDA_GPUS:-0,1,2,3}"
if [ -n "$CUDA_GPUS" ]; then
  export CUDA_VISIBLE_DEVICES="$CUDA_GPUS"
  NUM_GPUS=$(echo "$CUDA_GPUS" | tr ',' '\n' | wc -l)
else
  NUM_GPUS=1
fi

echo "[Exp2] Using GPUs: ${CUDA_GPUS} (num_gpus=${NUM_GPUS})"

SEED="${SEED:-1}"
LR="${LR:-2e-4}"
BATCH_SIZE="${BATCH_SIZE:-4}"
EPOCHS="${EPOCHS:-5}"
INPUT_STEPS="${INPUT_STEPS:-1}"
PREDICT_STEPS="${PREDICT_STEPS:-1}"
MAX_T="${MAX_T:-6}"
SAVE_EVAL_STEPS="${SAVE_EVAL_STEPS:-800}"

# Keep same variable set for fair comparison.
INPUT_VAR_LIST="${INPUT_VAR_LIST:-so thetao uo vo}"
IN_CHANS="${IN_CHANS:-16 16 16 16}"
OUT_CHANS="${OUT_CHANS:-16 16 16 16}"

CMIP6_DATA_DIR="${CMIP6_DATA_DIR:-./data/train_data}"
GODAS_DATA_DIR="${GODAS_DATA_DIR:-./data/train_data/godas_split}"
EXP_ROOT="${EXP_ROOT:-./output/ctrl_data_source}"

run_stage1_cmip6() {
  local out_dir="${EXP_ROOT}/stage1_cmip6"
  local dist_port=$((12345 + RANDOM % 12345))
  echo "[Exp2] Run CMIP6 stage1 -> ${out_dir}"

  torchrun --nproc_per_node="${NUM_GPUS}" train_stage1.py \
    --in_chans ${IN_CHANS} \
    --out_chans ${OUT_CHANS} \
    --max_t "${MAX_T}" \
    --atmo_var_list tauu tauv \
    --atmo_dims 2 \
    --ignore_mismatched_sizes True \
    --do_train \
    --dist_port "${dist_port}" \
    --data_dir "${CMIP6_DATA_DIR}" \
    --input_var_list ${INPUT_VAR_LIST} \
    --input_steps "${INPUT_STEPS}" \
    --predict_steps "${PREDICT_STEPS}" \
    --output_dir "${out_dir}" \
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
    --fsdp "full_shard auto_wrap" \
    --learning_rate "${LR}" \
    --weight_decay 0.1 \
    --max_grad_norm 0.0 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --adam_epsilon 1e-6 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1
}

run_stage1_godas() {
  local out_dir="${EXP_ROOT}/stage1_godas"
  local dist_port=$((12345 + RANDOM % 12345))
  echo "[Exp2] Run GODAS stage1 -> ${out_dir}"

  torchrun --nproc_per_node="${NUM_GPUS}" train_stage1_GODAS.py \
    --in_chans ${IN_CHANS} \
    --out_chans ${OUT_CHANS} \
    --max_t "${MAX_T}" \
    --atmo_var_list tauu tauv \
    --atmo_dims 2 \
    --ignore_mismatched_sizes True \
    --do_train \
    --dist_port "${dist_port}" \
    --data_dir "${GODAS_DATA_DIR}" \
    --input_var_list ${INPUT_VAR_LIST} \
    --input_steps "${INPUT_STEPS}" \
    --predict_steps "${PREDICT_STEPS}" \
    --output_dir "${out_dir}" \
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

run_stage1_cmip6
run_stage1_godas

echo "[Exp2] Data source control completed."
