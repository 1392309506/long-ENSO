seed=1
lr='2e-4'
batch_size=8
epoch=5
input_steps=1
predict_steps=1
max_t=6
input_var_list='so thetao uo vo'

save_eval_steps=800

dist_port=$[12345+$[$RANDOM%12345]]

output_dir=./output/train_stage1/exp1 # configure your output directory
data_dir=./data/godas # replace with your GODAS data directory

# If you use SLURM to launch the training script, you can use the following command:
# node_num=1
# gpu_per_node=4
# srun -p YOUR_PARTITION_NAME --ntasks-per-node=$gpu_per_node -N $node_num --gres=gpu:$gpu_per_node --async \
#     python -u train_stage1.py

# Otherwise, you can use torchrun to launch the training script

torchrun --nproc_per_node=4 \
    train_stage1.py \
        --in_chans 16 16 16 16 \
        --out_chans 16 16 16 16 \
        --max_t $max_t \
        --atmo_var_list tauu tauv \
        --atmo_dims 2 \
        --ignore_mismatched_sizes True \
        --do_train \
        --dist_port $dist_port \
        --data_dir $data_dir \
        --input_var_list $input_var_list \
        --input_steps $input_steps \
        --predict_steps $predict_steps \
        --output_dir $output_dir \
        --seed $seed \
        --report_to tensorboard \
        --log_level info \
        --logging_dir $output_dir/log \
        --logging_steps 5 \
        --log_on_each_node False \
        --save_strategy steps \
        --save_steps $save_eval_steps \
        --save_total_limit 3 \
        --ddp_find_unused_parameters False \
        --num_train_epochs $epoch \
        --per_device_train_batch_size $batch_size \
        --per_device_eval_batch_size $batch_size \
        --gradient_accumulation_steps 1 \
        --dataloader_num_workers 8 \
        --gradient_checkpointing False \
        --fsdp "full_shard auto_wrap" \
        --learning_rate $lr \
        --weight_decay 0.1 \
        --max_grad_norm 0.0 \
        --adam_beta1 0.9 \
        --adam_beta2 0.95 \
        --adam_epsilon 1e-6 \
        --lr_scheduler_type cosine \
        --warmup_ratio 0.1 \
        --evaluation_strategy steps \
        --eval_steps $save_eval_steps \
        --load_best_model_at_end True

#        --do_eval \
