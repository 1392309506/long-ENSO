import os
import json
import torch
import logging
import numpy as np
from tqdm import tqdm
from typing import List, Optional
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
from model import ORCADLModel, ORCADLPerturbationModel, ORCADLConfig, ModelArguments

logger = logging.getLogger(__name__)


@dataclass
class TestingArguments(TrainingArguments):
    num_test_samples: int = field(default=None, metadata={"help": "How many samples used for testing."})
    test_data_indices: List[int] = field(default_factory=list, metadata={"help": "Samples indices in test dataset. If set, will ignore num test samples."})
    save_preds: bool = field(default=False, metadata={"help": "If True, will save model predictions and true labels."})
    test_start_year: int = field(default=1983)
    test_end_year: int = field(default=2020)
    pred_start_year: int = field(default=1980)
    
    # Stage 1 model paths
    stage1_ckpt_list: List[str] = field(default_factory=list, metadata={"help": "阶段一模型权重路径列表"})
    stage1_config_path_list: List[str] = field(default_factory=list, metadata={"help": "阶段一模型配置路径列表"})
    
    # Stage 2 model paths (optional)
    stage2_ckpt_list: List[str] = field(default_factory=list, metadata={"help": "阶段二模型权重路径列表（可选）"})
    stage2_config_path_list: List[str] = field(default_factory=list, metadata={"help": "阶段二模型配置路径列表"})
    
    use_stage2: bool = field(default=False, metadata={"help": "是否使用阶段二扰动模型"})
    save_vars: List[str] = field(default_factory=list, metadata={"help": "需要保存的变量名列表"})

    def __post_init__(self):
        super().__post_init__()
        if len(self.test_data_indices) > 2:
            raise ValueError("Indices should be two values at most.([start, end) or [: end))")


def main():
    parser = HfArgumentParser((TestingArguments, DataArguments, ModelArguments))
    testing_args, data_args, model_args = parser.parse_args_into_dataclasses()

    setup_logger(testing_args, logger)
    logger.info(f"Testing parameters {testing_args}")

    set_seed(testing_args.seed)

    test_dataset = GodasDataset(data_args)
    var_list = test_dataset.get_input_var_list_cmip6()
    var_index = [test_dataset.get_var_index(v) for v in var_list]

    # Load Stage 1 models (deep-sea base models)
    stage1_models = []
    ckpt_list = testing_args.stage1_ckpt_list
    config_list = testing_args.stage1_config_path_list
    
    if len(ckpt_list) == 0:
        raise ValueError("Must provide at least one stage1 checkpoint via --stage1_ckpt_list")
    
    if len(config_list) != len(ckpt_list):
        if len(config_list) == 1:
            config_list = config_list * len(ckpt_list)
            logger.warning("Only one config path provided, using it for all stage1 checkpoints.")
        elif len(config_list) == 0:
            config_list = [None] * len(ckpt_list)
        else:
            raise ValueError("stage1_config_path_list length should match stage1_ckpt_list or be 1.")

    logger.info(f"Loading {len(ckpt_list)} Stage 1 models...")
    for ckpt_path, cfg_path in zip(ckpt_list, config_list):
        if cfg_path is not None:
            config = ORCADLConfig.from_json_file(cfg_path)
            model = ORCADLModel.from_pretrained(ckpt_path, config=config)
        else:
            model = ORCADLModel.from_pretrained(ckpt_path)
        
        if model.config.var_list != var_list or model.config.var_index != var_index:
            raise ValueError("var_list/var_index mismatch in stage1 model config")
        
        model.config.update({'predict_time_steps': data_args.predict_steps})
        model.cuda()
        model.eval()
        stage1_models.append(model)

    # Load Stage 2 models (perturbation models) if requested
    stage2_models = []
    if testing_args.use_stage2:
        s2_ckpt_list = testing_args.stage2_ckpt_list
        s2_config_list = testing_args.stage2_config_path_list
        
        if len(s2_ckpt_list) == 0:
            raise ValueError("use_stage2=True but no stage2_ckpt_list provided")
        
        if len(s2_config_list) != len(s2_ckpt_list):
            if len(s2_config_list) == 1:
                s2_config_list = s2_config_list * len(s2_ckpt_list)
            elif len(s2_config_list) == 0:
                s2_config_list = [None] * len(s2_ckpt_list)
            else:
                raise ValueError("stage2_config_path_list length should match stage2_ckpt_list or be 1.")
        
        logger.info(f"Loading {len(s2_ckpt_list)} Stage 2 models...")
        for ckpt_path, cfg_path in zip(s2_ckpt_list, s2_config_list):
            if cfg_path is not None:
                config = ORCADLConfig.from_json_file(cfg_path)
                model = ORCADLPerturbationModel.from_pretrained(ckpt_path, config=config)
            else:
                model = ORCADLPerturbationModel.from_pretrained(ckpt_path)
            
            if model.config.var_list != var_list or model.config.var_index != var_index:
                raise ValueError("var_list/var_index mismatch in stage2 model config")
            
            model.config.update({'predict_time_steps': data_args.predict_steps})
            model.cuda()
            model.eval()
            stage2_models.append(model)

    # Prepare dataset
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

    start_months = []
    preds = []

    model_config = stage1_models[0].config
    outdir = testing_args.output_dir
    preds_save_dir = os.path.join(outdir, 'preds')
    os.makedirs(outdir, exist_ok=True)
    if testing_args.save_preds:
        os.makedirs(preds_save_dir, exist_ok=True)

    logger.info(f"Running inference (Stage2={'ON' if testing_args.use_stage2 else 'OFF'})...")
    
    for batch in tqdm(dataloader):
        start_months.append(batch.pop('start_month').numpy())
        batch.pop('labels')
        
        # Prepare inputs based on stage
        if testing_args.use_stage2:
            # Keep atmo_vars for stage2
            for k in batch:
                batch[k] = batch[k].cuda()
        else:
            # Remove atmo_vars for stage1 only
            batch.pop('atmo_vars', None)
            for k in batch:
                batch[k] = batch[k].cuda()
        
        with torch.no_grad():
            if testing_args.use_stage2:
                # Run stage 2 models (includes stage1 inside)
                logits_sum = 0.0
                for model in stage2_models:
                    logits = model(**batch).preds
                    logits_sum += logits
                logits = logits_sum / len(stage2_models)
            else:
                # Run stage 1 models only
                logits_sum = 0.0
                for model in stage1_models:
                    logits = model(**batch).preds
                    logits_sum += logits
                logits = logits_sum / len(stage1_models)
        
        preds.append(logits.detach().cpu().numpy())

    # Post-process and save
    start_months = np.concatenate(start_months, axis=0)

    split_indices = np.array(model_config.out_chans)
    split_indices = list(accumulate(split_indices))[:-1]
    split_axis = 1 if data_args.predict_steps == 1 else 2

    preds = [np.split(p, split_indices, axis=split_axis) for p in preds]
    all_preds = []
    for i in range(len(preds[0])):
        all_preds.append(np.concatenate([p[i] for p in preds], axis=0))

    for v, pred in zip(var_list, all_preds):
        pred = pred.squeeze()
        if testing_args.save_preds and v in testing_args.save_vars:
            if data_args.predict_steps == 1:
                torch.save(pred, os.path.join(preds_save_dir, f'{v}.pt'), pickle_protocol=4)
            else:
                for i in range(pred.shape[1]):
                    torch.save(pred[:, i], os.path.join(preds_save_dir, f'{v}_step{i+1}.pt'), pickle_protocol=4)

    json.dump(init_time_list, open(os.path.join(outdir, 'init_times.json'), 'w'), indent=4)

    stage_info = "stage1+stage2" if testing_args.use_stage2 else "stage1_only"
    logger.info(f"Saved predictions ({stage_info}) to {outdir}")
    print('\n' + '*'*10 + ' Done ' + '*'*10)


if __name__ == "__main__":
    main()
