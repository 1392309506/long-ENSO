import numpy as np
import os
import glob

def inspect_npy_file(file_path, label="Data"):
    """检查单个npy文件的详细信息"""
    print(f"\n{'='*60}")
    print(f"检查文件: {file_path}")
    print(f"{'='*60}")
    
    # 加载数据
    data = np.load(file_path)
    
    print(f"文件标签: {label}")
    print(f"数据类型: {data.dtype}")
    print(f"数据形状: {data.shape}")
    print(f"数据维度: {data.ndim}D")
    print(f"数据大小: {data.nbytes / 1024 / 1024:.2f} MB")
    print(f"数据范围: [{data.min():.4f}, {data.max():.4f}]")
    print(f"数据均值: {data.mean():.4f}")
    print(f"数据标准差: {data.std():.4f}")
    print(f"是否存在NaN: {np.any(np.isnan(data))}")
    print(f"是否存在Inf: {np.any(np.isinf(data))}")
    
    return data

def compare_data_files(your_file, example_file):
    """对比两个数据文件"""
    print(f"\n{'#'*60}")
    print("对比分析")
    print(f"{'#'*60}")
    
    your_data = np.load(your_file)
    example_data = np.load(example_file)
    
    print(f"你的数据形状: {your_data.shape}")
    print(f"示例数据形状: {example_data.shape}")
    
    if your_data.shape == example_data.shape:
        print("\n✅ 形状相同!")
        # 计算统计差异
        diff = your_data - example_data
        print(f"差异范围: [{diff.min():.4f}, {diff.max():.4f}]")
        print(f"差异均值: {diff.mean():.4f}")
        print(f"差异标准差: {diff.std():.4f}")
        print(f"完全相同: {np.allclose(your_data, example_data)}")
    else:
        print("\n❌ 形状不同!")
        # 详细对比通道维度
        print(f"\n通道维度对比:")
        print(f"  你的数据: 通道数 = {your_data.shape[0] if your_data.ndim > 2 else 'N/A'}")
        print(f"  示例数据: 通道数 = {example_data.shape[0] if example_data.ndim > 2 else 'N/A'}")
        
        # 如果是3D数据 (C, H, W)
        if your_data.ndim == 3 and example_data.ndim == 3:
            print(f"\n通道数比例: 示例数据通道数 / 你的数据通道数 = {example_data.shape[0] / your_data.shape[0]}")
            
            # 检查是否是16倍关系
            if example_data.shape[0] == your_data.shape[0] * 16:
                print("\n🔍 发现: 示例数据通道数是你的16倍!")
                print("   这可能是因为示例数据做了patch embedding或通道扩展")
                
                # 尝试重塑你的数据来匹配示例数据
                try:
                    # 假设你的数据是 [4, H, W]，示例数据是 [64, H, W]
                    # 可以尝试重复或插值
                    repeated = np.repeat(your_data, 16, axis=0)
                    if repeated.shape == example_data.shape:
                        print("\n   通过重复可以将你的数据扩展到相同形状")
                except:
                    pass

def scan_directory(data_dir, pattern="*.npy"):
    """扫描目录中的所有npy文件"""
    files = glob.glob(os.path.join(data_dir, pattern))
    files.sort()
    return files

def batch_inspect():
    """批量检查目录结构"""
    print("\n\n批量检查目录结构")
    print("=================")
    
    # 这里需要修改为实际路径
    base_dirs = {
        "你的数据": "/path/to/your/data",
        "示例数据": "/path/to/example/data"
    }
    
    for label, base_dir in base_dirs.items():
        print(f"\n{label}:")
        if not os.path.exists(base_dir):
            print(f"  ❌ 目录不存在: {base_dir}")
            continue
            
        # 检查变量子目录
        var_dirs = glob.glob(os.path.join(base_dir, "*"))
        for var_dir in var_dirs[:5]:  # 只显示前5个
            var_name = os.path.basename(var_dir)
            files = glob.glob(os.path.join(var_dir, "*.npy"))
            if files:
                sample = np.load(files[0])
                print(f"  {var_name}: {len(files)} files, sample shape {sample.shape}")


def main():
    # 配置路径 - 需要你修改为实际路径
    YOUR_DATA_DIR = "../data/BCC-CSM2-MR/tos/1850_1.npy"
    EXAMPLE_DATA_DIR = "../data/train_data/godas/tos/1980_1.npy"
    YOUR_DATA_DIR=EXAMPLE_DATA_DIR
    VARIABLE = "tos"  # 要对比的变量名
    
    print("数据检查脚本")
    print("============")
    
    # 找到对应的文件
    your_files = scan_directory(os.path.join(YOUR_DATA_DIR, VARIABLE))
    example_files = scan_directory(os.path.join(EXAMPLE_DATA_DIR, VARIABLE))
    
    if not your_files:
        print(f"❌ 在你的数据目录中没有找到文件: {os.path.join(YOUR_DATA_DIR, VARIABLE)}")
        return
    
    if not example_files:
        print(f"❌ 在示例数据目录中没有找到文件: {os.path.join(EXAMPLE_DATA_DIR, VARIABLE)}")
        return
    
    # 取第一个文件进行对比
    your_file = your_files[0]
    example_file = example_files[0]
    
    print(f"\n对比文件:")
    print(f"  你的数据: {your_file}")
    print(f"  示例数据: {example_file}")
    
    # 分别检查
    your_data = inspect_npy_file(your_file, "你的数据")
    example_data = inspect_npy_file(example_file, "示例数据")
    
    # 对比
    compare_data_files(your_file, example_file)
    
    # 检查多个文件的一致性
    print(f"\n\n{'#'*60}")
    print("检查多个文件的一致性")
    print(f"{'#'*60}")
    
    # 检查你的数据中多个文件
    print(f"\n你的数据 - 前5个文件:")
    for i, f in enumerate(your_files[:5]):
        data = np.load(f)
        print(f"  {os.path.basename(f)}: shape {data.shape}, mean={data.mean():.4f}")
    
    # 检查示例数据中多个文件
    print(f"\n示例数据 - 前5个文件:")
    for i, f in enumerate(example_files[:5]):
        data = np.load(f)
        print(f"  {os.path.basename(f)}: shape {data.shape}, mean={data.mean():.4f}")

if __name__ == "__main__":
    main()
    # 如果需要批量检查目录结构，取消下面的注释
    # batch_inspect()