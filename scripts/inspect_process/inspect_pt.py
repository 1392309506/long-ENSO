import torch
import numpy as np

file_dir = '../output/predict/exp1_fix/preds/tos.pt'
# 加载文件
data = torch.load(file_dir, map_location='cpu', weights_only=False)

print("=" * 50)
print("文件内容探索")
print("=" * 50)

# 1. 首先查看数据类型
print(f"\n1. 数据类型: {type(data)}")

# 2. 如果是字典，查看所有键和值的类型/形状
if isinstance(data, dict):
    print(f"\n2. 字典包含 {len(data)} 个键:")
    for i, (key, value) in enumerate(data.items(), 1):
        print(f"\n   [{i}] 键: '{key}'")
        print(f"       类型: {type(value)}")
        
        # 如果是张量，显示详细信息
        if isinstance(value, torch.Tensor):
            print(f"       形状: {value.shape}")
            print(f"       数据类型: {value.dtype}")
            print(f"       设备: {value.device}")
            print(f"       值范围: [{value.min():.4f}, {value.max():.4f}]")
            print(f"       是否有NaN: {torch.isnan(value).any()}")
            
            # 显示前几个值作为样本
            if value.numel() > 0:
                flat_values = value.flatten()
                sample_size = min(5, len(flat_values))
                print(f"       前{sample_size}个值: {flat_values[:sample_size].tolist()}")
        
        # 如果是其他类型，也显示
        else:
            print(f"       内容: {value}")
            
# 3. 如果是张量（直接保存的张量）
elif isinstance(data, torch.Tensor):
    print(f"\n2. 这是一个张量:")
    print(f"   形状: {data.shape}")
    print(f"   数据类型: {data.dtype}")
    print(f"   设备: {data.device}")
    print(f"   维度数: {data.dim()}")
    print(f"   元素总数: {data.numel()}")
    print(f"   值范围: [{data.min():.4f}, {data.max():.4f}]")
    print(f"   是否有NaN: {torch.isnan(data).any()}")
    
    # 统计信息
    print(f"   均值: {data.mean():.4f}")
    print(f"   标准差: {data.std():.4f}")
    
    # 显示前几个值
    if data.numel() > 0:
        flat_values = data.flatten()
        sample_size = min(10, len(flat_values))
        print(f"   前{sample_size}个值: {flat_values[:sample_size].tolist()}")
    
    # 尝试解释维度（常见模式）
    print(f"\n3. 维度解释推测:")
    dim_explanations = []
    
    if len(data.shape) == 4:
        dim_explanations.append("4D张量常见解释:")
        dim_explanations.append("  - 维度0: 样本数/时间步 (batch/time)")
        dim_explanations.append("  - 维度1: 通道数 (channel)")
        dim_explanations.append("  - 维度2: 高度/纬度 (height/lat)")
        dim_explanations.append("  - 维度3: 宽度/经度 (width/lon)")
    elif len(data.shape) == 3:
        dim_explanations.append("3D张量常见解释:")
        dim_explanations.append("  - 方案A: [时间, 纬度, 经度]")
        dim_explanations.append("  - 方案B: [样本, 特征, 空间]")
        dim_explanations.append("  - 方案C: [通道, 高度, 宽度]")
    elif len(data.shape) == 2:
        dim_explanations.append("2D张量常见解释:")
        dim_explanations.append("  - [时间, 空间] 或 [样本, 特征]")
    elif len(data.shape) == 1:
        dim_explanations.append("1D张量常见解释:")
        dim_explanations.append("  - 时间序列或展平的特征")
    
    for exp in dim_explanations:
        print(exp)

# 4. 如果是列表或元组
elif isinstance(data, (list, tuple)):
    print(f"\n2. 这是一个{type(data).__name__}, 长度: {len(data)}")
    for i, item in enumerate(data[:3]):  # 只显示前3个
        print(f"   第{i}个元素: 类型={type(item)}")
        if isinstance(item, torch.Tensor):
            print(f"     形状: {item.shape}")

# 5. 尝试查看是否有属性或方法可以提供更多信息
print(f"\n4. 其他可能的信息:")
if hasattr(data, 'shape'):
    print(f"   已有shape属性: {data.shape}")
if hasattr(data, 'keys'):
    print(f"   已有keys方法，包含键: {list(data.keys()) if callable(data.keys) else '不可调用'}")
if hasattr(data, 'items'):
    print(f"   包含items方法")

# 6. 如果是自定义对象，查看其__dict__
if hasattr(data, '__dict__'):
    print(f"\n5. 对象的属性字典:")
    for key, value in data.__dict__.items():
        print(f"   {key}: {type(value)}")
        if isinstance(value, torch.Tensor):
            print(f"     形状: {value.shape}")

print("\n" + "=" * 50)
print("探索完成")
print("=" * 50)