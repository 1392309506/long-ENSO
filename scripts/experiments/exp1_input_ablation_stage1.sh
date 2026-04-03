#!/usr/bin/env bash

set -euo pipefail

# Exp1: Input variable ablation for Stage1 (CMIP6 training).
# Goal: quantify contribution of salinity/current/surface variables to ENSO prediction skill.

CUDA_GPUS="${CUDA_GPUS:-0,1,2,3}"
if [ -n "$CUDA_GPUS" ]; then
  export CUDA_VISIBLE_DEVICES="$CUDA_GPUS"
  NUM_GPUS=$(echo "$CUDA_GPUS" | tr ',' '\n' | wc -l)
else
  NUM_GPUS=1
fi

echo "[Exp1] Using GPUs: ${CUDA_GPUS} (num_gpus=${NUM_GPUS})"

SEED="${SEED:-1}"
LR="${LR:-2e-4}"
BATCH_SIZE="${BATCH_SIZE:-4}"
EPOCHS="${EPOCHS:-5}"
INPUT_STEPS="${INPUT_STEPS:-1}"
PREDICT_STEPS="${PREDICT_STEPS:-1}"
MAX_T="${MAX_T:-6}"
SAVE_EVAL_STEPS="${SAVE_EVAL_STEPS:-800}"

DATA_DIR="${DATA_DIR:-./data/train_data}"
SODA_DIR="${SODA_DIR:-}"
ORAS5_DIR="${ORAS5_DIR:-}"
EXP_ROOT="${EXP_ROOT:-./output/train_stage1/exp_input_ablation}"

# 4 controlled settings: only input variable set changes.
NAMES=("full6" "no_surface" "no_currents" "no_salinity")
VAR_LISTS=(
  "so thetao tos uo vo zos"
  "so thetao uo vo"
  "so thetao tos zos"
  "thetao tos uo vo zos"
)
IN_CHANS_LISTS=(
  "16 16 1 16 16 1"
  "16 16 16 16"
  "16 16 1 1"
  "16 1 16 16 1"
)
OUT_CHANS_LISTS=(
  "16 16 1 16 16 1"
  "16 16 16 16"
  "16 16 1 1"
  "16 1 16 16 1"
)

for i in "${!NAMES[@]}"; do
  name="${NAMES[$i]}"
  input_var_list="${VAR_LISTS[$i]}"
  in_chans="${IN_CHANS_LISTS[$i]}"
  out_chans="${OUT_CHANS_LISTS[$i]}"

  out_dir="${EXP_ROOT}/${name}"
  dist_port=$((12345 + RANDOM % 12345))

  echo "[Exp1] Running setting=${name}, output=${out_dir}"

  cmd=(
    torchrun --nproc_per_node="${NUM_GPUS}" train_stage1.py
      --in_chans ${in_chans}
      --out_chans ${out_chans}
      --max_t "${MAX_T}"
      --atmo_var_list tauu tauv
      --atmo_dims 2
      --ignore_mismatched_sizes True
      --do_train
      --dist_port "${dist_port}"
      --data_dir "${DATA_DIR}"
      --input_var_list ${input_var_list}
      --input_steps "${INPUT_STEPS}"
      --predict_steps "${PREDICT_STEPS}"
      --output_dir "${out_dir}"
      --seed "${SEED}"
      --report_to tensorboard
      --log_level info
      --logging_dir "${out_dir}/log"
      --logging_steps 5
      --log_on_each_node False
      --save_strategy steps
      --save_steps "${SAVE_EVAL_STEPS}"
      --save_total_limit 3
      --ddp_find_unused_parameters False
      --num_train_epochs "${EPOCHS}"
      --per_device_train_batch_size "${BATCH_SIZE}"
      --per_device_eval_batch_size "${BATCH_SIZE}"
      --gradient_accumulation_steps 1
      --dataloader_num_workers 8
      --gradient_checkpointing False
      --fsdp "full_shard auto_wrap"
      --learning_rate "${LR}"
      --weight_decay 0.1
      --max_grad_norm 0.0
      --adam_beta1 0.9
      --adam_beta2 0.95
      --adam_epsilon 1e-6
      --lr_scheduler_type cosine
      --warmup_ratio 0.1
  )

  # Enable eval only when both validation dirs are provided.
  if [ -n "${SODA_DIR}" ] && [ -n "${ORAS5_DIR}" ]; then
    cmd+=(
      --do_eval
      --valid_data_dir "${SODA_DIR}" "${ORAS5_DIR}"
      --evaluation_strategy steps
      --eval_steps "${SAVE_EVAL_STEPS}"
      --load_best_model_at_end True
    )
  fi

  "${cmd[@]}"
done

echo "[Exp1] Input ablation completed."
