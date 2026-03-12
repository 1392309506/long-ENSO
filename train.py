import os
import json
import logging

from transformers import set_seed, HfArgumentParser
from transformers.trainer_utils import get_last_checkpoint
from transformers.trainer_pt_utils import get_model_param_count

from dataset import DataArguments, Cmip6Dataset, ReanalyCombinedDataset
from model import ModelArguments, ORCADLConfig, BaseModel

from trainer import (
    Trainer, TrainingArguments,
    get_default_callbacks, setup_logger, collate_fn
)

logger = logging.getLogger(__name__)

def main():
    # 使用 Hugging Face 的 ArgumentParser 解析三类参数：训练、数据、模型
    parser = HfArgumentParser((TrainingArguments, DataArguments, ModelArguments))
    training_args, data_args, model_args = parser.parse_args_into_dataclasses()

    # 如果提供了外部数据配置文件（JSON），则覆盖 data_args
    if data_args.data_config_path is not None:
        with open(data_args.data_config_path, 'r') as f:
            data_dict = json.load(f)
        data_args = type("DataArguments", (), data_dict)

    # Setup logging
    setup_logger(training_args, logger)
    training_args._setup_devices
    logger.warning(
        f"Process global rank: {training_args.global_rank}, local rank: {training_args.local_rank}, "
        + f"device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed: {bool(training_args.local_rank != -1)}, 16-bits: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # 检查输出目录是否存在检查点，用于恢复训练
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # 设置种子
    set_seed(training_args.seed)

    # 加载数据集（CMIP6 气候模拟数据）
    train_dataset = Cmip6Dataset(data_args, split='train')
    # 可选：加载验证数据集（再分析数据 + CMIP6 组合）
    eval_dataset = None
    if training_args.do_eval:
        eval_dataset = ReanalyCombinedDataset(data_args, data_args.valid_data_dir, split='valid')

    # 获取输入变量列表及其在数据中的索引（用于模型配置）
    var_list = train_dataset.get_input_var_list_cmip6()
    var_index = [train_dataset.get_var_index(v) for v in var_list]

    # 根据是否提供预训练模型路径，决定从头训练还是加载预训练模型
    if model_args.model_path is None:
        logger.warning("Trying to train a model from scratch")
        # 若提供模型配置文件，则加载；否则使用默认配置
        if model_args.model_config_path is not None:
            logger.warning(f"Using model config defined in {model_args.model_config_path}")
            config = ORCADLConfig.from_json_file(model_args.model_config_path)
        else:
            logger.warning("Using default model config")
            config = ORCADLConfig()

        # 更新配置：注入数据相关参数（变量、时间步等）
        config.update({
            'var_list': var_list,
            'var_index': var_index,
            'max_t': data_args.max_t,
            'predict_time_steps': data_args.predict_steps,
        })
        # 用命令行参数进一步覆盖配置（如学习率、层数等）
        config.update_from_args(model_args)

        # 从配置构建新模型
        model = BaseModel(config)
    else:
        # 从预训练路径加载配置和模型权重
        config = ORCADLConfig.from_pretrained(model_args.model_path)
        config.update_from_args(model_args)  # 允许微调时覆盖部分配置
        model = BaseModel.from_pretrained(
            model_args.model_path,
            config=config,
            ignore_mismatched_sizes=model_args.ignore_mismatched_sizes  # 允许部分权重不匹配（如分类头变化）
        )
        # 动态更新预测时间步（因可能与预训练时不一致）
        model.config.update({
            'predict_time_steps': data_args.predict_steps,
        })

    logger.info(f"Model Config {model.config}")

    # 初始化自定义 Trainer（支持气候数据特有逻辑）
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        callbacks=get_default_callbacks(),  # 默认回调（如早停、日志等）
        data_collator=collate_fn  # 自定义 batch 整理函数
    )

    # ===== 训练阶段 =====
    if training_args.do_train:
        # 确定恢复训练的检查点路径
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        # 启动训练
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        # 添加额外指标：样本数、模型参数量
        metrics["train_samples"] = len(train_dataset)
        metrics["params"] = get_model_param_count(model)

        # 保存模型、分词器（如有）、指标和训练状态
        trainer.save_model()  # Saves the tokenizer too for easy upload
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # ===== 评估阶段 =====
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        metrics["eval_samples"] = len(eval_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # 保存所有参数到 output_dir，便于复现实验
    with open(os.path.join(training_args.output_dir, 'args.json'), 'w') as fp:
        json.dump({
            'data_args': data_args.to_dict(),
            'model_args': model_args.to_dict(),
            'training_args': training_args.to_dict(),
        }, fp, indent=2)


if __name__ == "__main__":
    main()