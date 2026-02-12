import os
import json
import logging

from dataclasses import dataclass, field
from transformers import set_seed, HfArgumentParser
from transformers.trainer_utils import get_last_checkpoint
from transformers.trainer_pt_utils import get_model_param_count

from dataset import DataArguments, Cmip6Dataset, ReanalyCombinedDataset
from model import ModelArguments, ORCADLConfig, ORCADLModel, ORCADLPerturbationModel

from trainer import (
    Trainer, TrainingArguments,
    get_default_callbacks, setup_logger, collate_fn
)

logger = logging.getLogger(__name__)


@dataclass
class Stage2Arguments:
    base_model_path: str = field(default=None, metadata={"help": "Stage1 model path (save_pretrained directory)."})
    freeze_base_model: bool = field(default=True, metadata={"help": "Freeze stage1 model during diffusion training."})


def main():
    parser = HfArgumentParser((TrainingArguments, DataArguments, ModelArguments, Stage2Arguments))
    training_args, data_args, model_args, stage2_args = parser.parse_args_into_dataclasses()

    if data_args.data_config_path is not None:
        with open(data_args.data_config_path, 'r') as f:
            data_dict = json.load(f)
        data_args = type("DataArguments", (), data_dict)

    setup_logger(training_args, logger)
    training_args._setup_devices
    logger.warning(
        f"Process local rank: {training_args.local_rank}, "
        + f"device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed: {bool(training_args.local_rank != -1)}, 16-bits: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # 如输出目录已有 checkpoint，则自动从最新的继续训练。
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

    # 设定随机种子，保证可复现。
    set_seed(training_args.seed)

    # 构建训练/评估数据集。
    train_dataset = Cmip6Dataset(data_args, split='train')
    eval_dataset = None
    if training_args.do_eval:
        eval_dataset = ReanalyCombinedDataset(data_args, data_args.valid_data_dir, split='valid')

    # 将输入变量名映射为索引，写入模型配置。
    var_list = train_dataset.get_input_var_list_cmip6()
    var_index = [train_dataset.get_var_index(v) for v in var_list]

    if stage2_args.base_model_path is None:
        raise ValueError("Stage2 requires --base_model_path pointing to stage1 checkpoint directory.")

    # 从指定配置文件或 base model checkpoint 加载配置。
    if model_args.model_config_path is not None:
        logger.warning(f"Using model config defined in {model_args.model_config_path}")
        config = ORCADLConfig.from_json_file(model_args.model_config_path)
    else:
        config = ORCADLConfig.from_pretrained(stage2_args.base_model_path)

    base_var_list = list(config.var_list)
    depth_count = config.in_chans[0] if config.in_chans else 16
    single_lev_set = set(train_dataset.single_lev_vars)
    stage2_var_chans = [1 if v in single_lev_set else depth_count for v in var_list]

    config.update({
        'var_list': var_list,
        'var_index': var_index,
        'max_t': data_args.max_t,
        'predict_time_steps': data_args.predict_steps,
    })
    config.update_from_args(model_args)

    # Stage-2 extra metadata for conditioning and target selection.
    config.stage2_base_vars = base_var_list
    config.stage2_full_vars = var_list
    config.stage2_surface_vars = ['tos', 'zos']
    config.stage2_target_vars = ['tos']
    config.stage2_var_chans = stage2_var_chans

    # Stage2 从已训练的 stage1 base model 开始。
    base_model = ORCADLModel.from_pretrained(
        stage2_args.base_model_path,
        config=config,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes
    )

    # 在 base model 上叠加扰动头用于扩散训练。
    model = ORCADLPerturbationModel(
        config=config,
        base_model=base_model,
        freeze_base_model=stage2_args.freeze_base_model
    )

    logger.info(f"Model Config {model.config}")

    # Trainer 负责训练流程与评估/保存。
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        callbacks=get_default_callbacks(),
        data_collator=collate_fn
    )

    # 执行训练（可断点续训），并保存指标与状态。
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)
        metrics["params"] = get_model_param_count(model)

        trainer.save_model()
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # 如需评估则执行。
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        metrics["eval_samples"] = len(eval_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # 保存参数配置便于复现实验。
    with open(os.path.join(training_args.output_dir, 'args.json'), 'w') as fp:
        json.dump({
            'data_args': data_args.to_dict(),
            'model_args': model_args.to_dict(),
            'stage2_args': {
                'base_model_path': stage2_args.base_model_path,
                'freeze_base_model': stage2_args.freeze_base_model,
            },
            'training_args': training_args.to_dict(),
        }, fp, indent=2)


if __name__ == "__main__":
    main()
