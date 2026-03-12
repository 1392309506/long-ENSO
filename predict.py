import os
import json
import torch
import logging
import numpy as np
from tqdm import tqdm
from typing import List
from itertools import accumulate
from dataclasses import dataclass, field
from transformers import set_seed, HfArgumentParser
from torch.utils.data import DataLoader

from trainer import (
    TrainingArguments,
    setup_logger,
    collate_fn,
)

from dataset import DataArguments, GodasDataset
from model import PerturbationModel, ORCADLConfig, ModelArguments

logger = logging.getLogger(__name__)


def _resolve_output_vars_and_chans(model_config, fallback_vars):
    """Resolve output variable names and channel sizes for stage2 or fallback config."""
    target_vars = getattr(model_config, 'stage2_target_vars', None)
    full_vars = getattr(model_config, 'stage2_full_vars', None)
    var_chans = getattr(model_config, 'stage2_var_chans', None)

    if target_vars:
        output_vars = list(target_vars)
        if full_vars and var_chans:
            chan_map = {v: c for v, c in zip(full_vars, var_chans)}
            output_chans = [chan_map[v] for v in output_vars]
        else:
            output_chans = [sum(model_config.out_chans)]
        return output_vars, output_chans

    return fallback_vars, list(model_config.out_chans)


@dataclass
class TestingArguments(TrainingArguments):
    """Inference-time arguments, including checkpoint ensemble and save controls."""
    num_test_samples: int = field(default=None, metadata={"help": "How many samples used for testing."})
    test_data_indices: List[int] = field(default_factory=list, metadata={"help": "Samples indices in test dataset. If set, will ignore num test samples."})
    save_preds: bool = field(default=False, metadata={"help": "If True, will save model predictions and true labels."})
    test_start_year: int = field(default=1983)
    test_end_year: int = field(default=2020)
    pred_start_year: int = field(default=1980)
    fixed_lead_time: int = field(default=None, metadata={"help": "Fixed lead time for inference (1-based)."})
    ckpt_list: List[str] = field(default_factory=list, metadata={"help": "模型权重路径列表（用于集成）"})
    config_path_list: List[str] = field(default_factory=list, metadata={"help": "模型配置路径列表"})
    save_vars: List[str] = field(default_factory=list, metadata={"help": "需要保存的变量名列表"})

    def __post_init__(self):
        super().__post_init__()
        if len(self.test_data_indices) > 2:
            raise ValueError("Indices should be two values at most.([start, end) or [: end))")


def main():
    # 解析命令行参数并初始化日志。
    parser = HfArgumentParser((TestingArguments, DataArguments, ModelArguments))
    testing_args, data_args, model_args = parser.parse_args_into_dataclasses()

    setup_logger(testing_args, logger)
    logger.info(f"Testing parameters {testing_args}")

    if testing_args.fixed_lead_time is not None:
        if testing_args.fixed_lead_time < 1:
            raise ValueError("fixed_lead_time must be >= 1 (1-based).")
        if data_args.predict_steps != 1:
            logger.warning(
                "For lead-time experiments, set predict_steps=1 to avoid window-length confounding. "
                f"Current predict_steps={data_args.predict_steps}."
            )

    set_seed(testing_args.seed)

    # 构建测试数据集，并校验变量顺序/索引与模型配置一致。
    test_dataset = GodasDataset(data_args)
    var_list = test_dataset.get_input_var_list_cmip6()
    var_index = [test_dataset.get_var_index(v) for v in var_list]

    model_list = []
    # 支持多 checkpoint 集成：可为所有 ckpt 复用同一个 config。
    if len(testing_args.ckpt_list) != len(testing_args.config_path_list):
        if len(testing_args.config_path_list) == 1:
            testing_args.config_path_list = [testing_args.config_path_list[0]] * len(testing_args.ckpt_list)
            logger.warning("Only one config path is provided, using it for all checkpoints.")
        elif len(testing_args.config_path_list) == 0:
            testing_args.config_path_list = [None] * len(testing_args.ckpt_list)
        else:
            raise ValueError("The config path list length should be the same as the checkpoint list length or 1.")

    for ckpt_path, cfg_path in zip(testing_args.ckpt_list, testing_args.config_path_list):
        if cfg_path is not None:
            config = ORCADLConfig.from_json_file(cfg_path)
            model = PerturbationModel.from_pretrained(ckpt_path, config=config)
        else:
            model = PerturbationModel.from_pretrained(ckpt_path)

        if model.config.var_list != var_list or model.config.var_index != var_index:
            raise ValueError("var_list/var_index in args is not the same as in pretrained model config")

        model.config.update({
            'predict_time_steps': data_args.predict_steps,
        })
        model_list.append(model)

    # 根据用户指定的下标范围截取测试集子集。
    indices_ = None
    if testing_args.num_test_samples is not None or len(testing_args.test_data_indices) > 0:
        indices = testing_args.test_data_indices
        if len(indices) > 0:
            if len(indices) == 1:
                indices_ = list(range(indices[0]))
            else:
                if indices[1] < 0:
                    indices[1] = len(test_dataset) + indices[1] + 1
                indices_ = list(range(indices[0], indices[1]))
        else:
            indices_ = list(range(testing_args.num_test_samples))

    if indices_ is not None:
        test_dataset = test_dataset.get_subset(indices_)

    init_time_list = test_dataset.times[:len(test_dataset)]

    dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=testing_args.per_device_eval_batch_size,
        shuffle=False,
        num_workers=testing_args.dataloader_num_workers,
        collate_fn=collate_fn,
        drop_last=False,
    )

    for model in model_list:
        model.cuda()
        model.eval()

    start_months = []
    preds = []

    model_config = model_list[0].config
    outdir = testing_args.output_dir
    preds_save_dir = os.path.join(testing_args.output_dir, 'preds')
    os.makedirs(outdir, exist_ok=True)
    if testing_args.save_preds:
        os.makedirs(preds_save_dir, exist_ok=True)

    num_model = len(model_list)

    for batch in tqdm(dataloader):
        start_months.append(batch.pop('start_month').numpy())
        batch.pop('labels')
        for k in batch:
            batch[k] = batch[k].cuda()

        # 固定 lead-time 实验：用同一个 lead_time 覆盖 batch 内所有样本。
        if testing_args.fixed_lead_time is not None:
            lead_value = testing_args.fixed_lead_time - 1
            bsz = batch['ocean_vars'].shape[0]
            fixed_lead = torch.full((bsz,), lead_value, device=batch['ocean_vars'].device, dtype=torch.long)
            batch['lead_time'] = fixed_lead
            if 'atmo_vars' in batch:
                batch['atmo_lead_time'] = fixed_lead

        with torch.no_grad():
            # 多模型预测取平均，实现简单集成。
            logits_sum = 0.0
            for model in model_list:
                logits = model(**batch).preds
                logits_sum += logits
            logits = logits_sum / num_model
        preds.append(logits.detach().cpu().numpy())

    start_months = np.concatenate(start_months, axis=0)

    output_vars, output_chans = _resolve_output_vars_and_chans(model_config, var_list)
    split_indices = np.array(output_chans)
    split_indices = list(accumulate(split_indices))[:-1]
    # [B, C, H, W] 在 axis=1 切分；多步输出通常是 [B, T, C, H, W]，在 axis=2 切分。
    split_axis = 1 if preds and preds[0].ndim == 4 else 2

    preds = [np.split(p, split_indices, axis=split_axis) for p in preds]
    all_preds = []
    for i in range(len(preds[0])):
        all_preds.append(np.concatenate([p[i] for p in preds], axis=0))

    for v, pred in zip(output_vars, all_preds):
        # 仅保存用户指定变量；多步输出按 step 拆分为多个 .pt 文件。
        if testing_args.save_preds and v in testing_args.save_vars:
            pred_arr = pred
            if pred_arr.ndim == 4 and pred_arr.shape[1] == 1:
                pred_arr = pred_arr[:, 0]

            if pred_arr.ndim == 3:
                torch.save(pred_arr, os.path.join(preds_save_dir, f'{v}.pt'), pickle_protocol=4)
            elif pred_arr.ndim == 4:
                for i in range(pred_arr.shape[1]):
                    torch.save(pred_arr[:, i], os.path.join(preds_save_dir, f'{v}_step{i+1}.pt'), pickle_protocol=4)
            else:
                raise ValueError(f"Unexpected prediction shape for {v}: {pred_arr.shape}")

    json.dump(init_time_list, open(os.path.join(outdir, 'init_times.json'), 'w'), indent=4)

    print('\n' + '*'*10 + ' Done ' + '*'*10)



if __name__ == "__main__":
    main()
