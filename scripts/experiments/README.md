# ENSO 对照实验脚本说明

本目录给出 4 组可直接运行的对照实验模板，原则是单因素变化，其余训练配置尽量保持一致。

## 实验 1: 输入变量消融

脚本: `exp1_input_ablation_stage1.sh`

对照组设置:
- `full6`: `so thetao tos uo vo zos`
- `no_surface`: 去掉表层变量 `tos zos`
- `no_currents`: 去掉流场变量 `uo vo`
- `no_salinity`: 去掉盐度变量 `so`

目标:
- 定量比较不同物理变量对 ENSO 预测的贡献。

输出目录:
- `./output/train_stage1/exp_input_ablation/<setting_name>`

## 实验 2: 训练数据源对照

脚本: `exp2_data_source_control.sh`

对照组设置:
- `stage1_cmip6`: 使用 `train_stage1.py` 在 CMIP6 上训练
- `stage1_godas`: 使用 `train_stage1_GODAS.py` 在 GODAS 上训练

控制方式:
- 保持 `input_var_list`、学习率、batch size、epoch 一致，只改变训练数据来源。

目标:
- 分离并评估数据域差异（模拟数据 vs 再分析数据）对模型性能的影响。

输出目录:
- `./output/ctrl_data_source/stage1_cmip6`
- `./output/ctrl_data_source/stage1_godas`

## 实验 3: Stage2 冻结策略对照

脚本: `exp3_stage2_freeze_control.sh`

对照组设置:
- `freeze_true`: `--freeze_base_model True`
- `freeze_false`: `--freeze_base_model False`

控制方式:
- 同一 `BASE_MODEL_PATH`，同一训练数据与超参数，只改变是否冻结 stage1 主干。

目标:
- 比较参数高效微调与全参数微调在 ENSO 任务上的差异。

输出目录:
- `./output/train_stage2/exp_freeze_control/freeze_true`
- `./output/train_stage2/exp_freeze_control/freeze_false`

## 实验 4: 推理集成数对照

脚本: `exp4_ensemble_size_control.sh`

对照组设置:
- `K=1`
- `K=3`
- `K=5`

控制方式:
- 同一测试集、同一模型结构、同一推理参数，只改变 checkpoint 集成个数。

目标:
- 评估集成是否提升 Nino3.4 预测稳定性和平均误差指标。

输出目录:
- `./output/predict/exp_ensemble_size/k1`
- `./output/predict/exp_ensemble_size/k3`
- `./output/predict/exp_ensemble_size/k5`

## 运行方式

1. 根据你的环境修改每个脚本头部的路径变量（如 `DATA_DIR`、`BASE_MODEL_PATH`、`CKPT_DIRS`）。
2. 赋予脚本执行权限并运行:

```bash
cd scripts/experiments
chmod +x *.sh
bash exp1_input_ablation_stage1.sh
bash exp2_data_source_control.sh
bash exp3_stage2_freeze_control.sh
bash exp4_ensemble_size_control.sh
```

## 建议统计指标

建议对每组实验统一计算:
- Nino3.4 相关系数（ACC）
- Nino3.4 均方根误差（RMSE）
- 不同 lead month 的技能曲线（1-24）

可使用仓库已有脚本:
- `scripts/calculate_result/calculate_nino3.4_by_pt_dir.py`
- `scripts/draw_nino3.4_by_csv_dir.py`
