#!/usr/bin/env bash

set -euo pipefail

input_var_list='so thetao uo vo'
input_steps=1
predict_steps=1

data_dir=./data/test_data/godas
stage1_ckpt_dir=./output/train_stage1/exp10
stage1_config=$stage1_ckpt_dir/config.json

batch_size=8
base_out=./output/predict_stage1/exp10

# GPU configuration: set usable GPUs here. Example: "0,2,4,7"
CUDA_GPUS="1,2,3,4,5,6,7"
IFS=',' read -r -a GPU_LIST <<< "$CUDA_GPUS"
max_parallel=${#GPU_LIST[@]}

if (( max_parallel == 0 )); then
  echo "Error: CUDA_GPUS is empty. Please set at least one GPU id."
  exit 1
fi

echo "Using GPUs: ${CUDA_GPUS} (total: ${max_parallel})"
jobs_in_batch=0

for lead in $(seq 1 21); do
  out_dir=${base_out}/lead_${lead}
  dist_port=$[31234+$[$RANDOM%100]]

  gpu_index=$(( (lead - 1) % max_parallel ))
  gpu=${GPU_LIST[$gpu_index]}

  echo "Running stage1 lead_time=${lead}, output=${out_dir}"

  CUDA_VISIBLE_DEVICES=$gpu python -u predict_stage1.py \
    --save_preds True \
    --save_vars so thetao uo vo \
    --ckpt_list $stage1_ckpt_dir \
    --config_path_list $stage1_config \
    --dist_port $dist_port \
    --data_dir $data_dir \
    --input_var_list $input_var_list \
    --input_steps $input_steps \
    --predict_steps $predict_steps \
    --fixed_lead_time $lead \
    --output_dir $out_dir \
    --overwrite_output_dir \
    --seed 1 \
    --log_level info \
    --dataloader_num_workers 8 \
    --per_device_eval_batch_size $batch_size &

  jobs_in_batch=$((jobs_in_batch + 1))
  if (( jobs_in_batch == max_parallel )); then
    wait
    jobs_in_batch=0
  fi

done

wait
echo "All stage1 lead-time jobs finished."
