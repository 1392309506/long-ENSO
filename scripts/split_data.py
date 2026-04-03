import os
import numpy as np
import shutil
import re

def split_dataset_by_time(data_root, output_root, split_year=None, train_years=None, test_years=None):
    """
    按时间拆分数据集（基于文件名中的年份信息）
    
    Parameters:
    -----------
    data_root : str
        原始数据根目录
    output_root : str
        输出根目录
    split_year : int
        指定分割年份（例如：2000，则1999及之前为训练集，2000及之后为测试集）
    train_years : list
        指定训练集年份范围，如 [1850, 1999]
    test_years : list
        指定测试集年份范围，如 [2000, 2010]
    """
    
    def extract_year(filename):
        """从文件名提取年份 (假设格式为 '1850_1.npy' 或 '1850_01.npy')"""
        match = re.match(r'(\d+)', filename)
        if match:
            return int(match.group(1))
        return None
    
    # 创建输出目录
    train_dir = os.path.join(output_root, 'train')
    test_dir = os.path.join(output_root, 'test')
    
    # 遍历所有变量目录
    for var_dir in os.listdir(data_root):
        var_path = os.path.join(data_root, var_dir)
        if not os.path.isdir(var_path):
            continue
            
        print(f"处理变量: {var_dir}")
        
        # 获取所有.npy文件
        npy_files = [f for f in os.listdir(var_path) if f.endswith('.npy')]
        npy_files.sort()
        
        if not npy_files:
            print(f"  警告: {var_dir} 中没有.npy文件")
            continue
        
        # 按年份分类文件
        train_files = []
        test_files = []
        
        for f in npy_files:
            year = extract_year(f)
            if year is None:
                print(f"  警告: 无法从文件名提取年份: {f}")
                continue
            
            if split_year is not None:
                # 按分割年份
                if year < split_year:
                    train_files.append(f)
                else:
                    test_files.append(f)
            elif train_years and test_years:
                # 按指定年份范围
                if train_years[0] <= year <= train_years[1]:
                    train_files.append(f)
                elif test_years[0] <= year <= test_years[1]:
                    test_files.append(f)
                else:
                    print(f"  跳过: 年份 {year} 不在指定范围内")
        
        print(f"  总文件数: {len(npy_files)}")
        print(f"  训练集: {len(train_files)}")
        print(f"  测试集: {len(test_files)}")
        
        # 创建并复制文件
        train_var_dir = os.path.join(train_dir, var_dir)
        test_var_dir = os.path.join(test_dir, var_dir)
        os.makedirs(train_var_dir, exist_ok=True)
        os.makedirs(test_var_dir, exist_ok=True)
        
        for f in train_files:
            src = os.path.join(var_path, f)
            dst = os.path.join(train_var_dir, f)
            shutil.copy2(src, dst)
            
        for f in test_files:
            src = os.path.join(var_path, f)
            dst = os.path.join(test_var_dir, f)
            shutil.copy2(src, dst)
    
    print(f"\n拆分完成！")
    print(f"训练集目录: {train_dir}")
    print(f"测试集目录: {test_dir}")

# 使用示例
if __name__ == "__main__":
    data_root = "../data/godas"
    output_root = "./godas_split_by_year"
    
    # 方法1：按分割年份
    split_dataset_by_time(data_root, output_root, split_year=2022)
    
    # 方法2：按指定年份范围
    # split_dataset_by_time(data_root, output_root, 
    #                      train_years=[1850, 1999], 
    #                      test_years=[2000, 2010])