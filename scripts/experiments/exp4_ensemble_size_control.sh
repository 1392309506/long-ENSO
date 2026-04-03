#!/usr/bin/env bash

set -euo pipefail

# Exp4: Ensemble size control at inference (K=1/3/5 checkpoints).
# Goal: measure stability/skill gain from checkpoint ensemble.

input_var_list="${INPUT_VAR_LIST:-so thetao tos uo vo zos}"
input_steps="${INPUT_STEPS:-1}"
predict_steps="${PREDICT_STEPS:-1}"

data_dir="${DATA_DIR:-./data/test_data/godas_split}"
base_out="${BASE_OUT:-./output/predict/exp_ensemble_size}"
batch_size="${BATCH_SIZE:-8}"
seed="${SEED:-1}"

# Fill with your candidate stage2 checkpoints (at least 5 for full comparison).
CKPT_DIRS=(
  "./output/train_stage2/exp2/checkpoint-60"
  "./output/train_stage2/exp2/checkpoint-120"
  "./output/train_stage2/exp2/checkpoint-180"
  "./output/train_stage2/exp4/checkpoint-60"
  "./output/train_stage2/exp4/checkpoint-120"
)

if (( ${#CKPT_DIRS[@]} < 5 )); then
  echo "[Exp4] Need at least 5 checkpoints in CKPT_DIRS for K=1/3/5 control."
  exit 1
fi

for ckpt in "${CKPT_DIRS[@]}"; do
  if [ ! -d "${ckpt}" ]; then
    echo "[Exp4] Checkpoint dir does not exist: ${ckpt}"
    exit 1
  fi
done

# Reuse one config path; predict.py supports broadcasting one config to all checkpoints.
config_path="${CONFIG_PATH:-${CKPT_DIRS[0]}/config.json}"
if [ ! -f "${config_path}" ]; then
  echo "[Exp4] Config file not found: ${config_path}"
  exit 1
fi

run_k() {
  local k="$1"
  local out_dir="${base_out}/k${k}"
  local dist_port=$((31234 + RANDOM % 100))

  local ckpt_subset=("${CKPT_DIRS[@]:0:${k}}")

  echo "[Exp4] Run ensemble size K=${k}, output=${out_dir}"

  python -u predict.py \
    --save_preds True \
    --save_vars tos \
    --ckpt_list "${ckpt_subset[@]}" \
    --config_path_list "${config_path}" \
    --dist_port "${dist_port}" \
    --data_dir "${data_dir}" \
    --input_var_list ${input_var_list} \
    --input_steps "${input_steps}" \
    --predict_steps "${predict_steps}" \
    --atmo_var_list tauu tauv \
    --output_dir "${out_dir}" \
    --overwrite_output_dir \
    --seed "${seed}" \
    --log_level info \
    --dataloader_num_workers 8 \
    --per_device_eval_batch_size "${batch_size}"
}

run_k 1
run_k 3
run_k 5

echo "[Exp4] Ensemble-size control completed."
