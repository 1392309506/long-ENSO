import numpy as np
import pandas as pd
import os
import glob
from nino34_utils import get_nino34_indices_and_weights, nino34_weighted_mean

# --- 配置参数 ---
input_dir = '../data/test_data/godas_split/tos'  # npy文件所在目录
output_file = './nino3.4/godas_2010-2023.csv'
lat_file = '../grid/lat.npy'
lon_file = '../grid/lon.npy'


def _build_grid_from_shape(shape_hw):
    h, w = shape_hw
    lat = np.linspace(-63.5, 63.5, h)
    lon = np.linspace(0.5, 359.5, w)
    return lat, lon

# --- 如果输出目录不存在，则创建目录 ---
output_dir = os.path.dirname(output_file)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"创建输出目录: {output_dir}")

sample_files = glob.glob(os.path.join(input_dir, "*.npy"))
if len(sample_files) == 0:
    raise FileNotFoundError(f"No npy files found in {input_dir}")

sample_data = np.load(sample_files[0])
if sample_data.ndim == 3 and sample_data.shape[0] == 1:
    sample_data = sample_data.squeeze(0)
if sample_data.ndim != 2:
    raise ValueError(f"Expected 2D tos data, got shape={sample_data.shape}")

if os.path.exists(lat_file) and os.path.exists(lon_file):
    lat_grid = np.load(lat_file).squeeze()
    lon_grid = np.load(lon_file).squeeze()
else:
    lat_grid, lon_grid = _build_grid_from_shape(sample_data.shape)
    print("未找到 grid/lat.npy 或 grid/lon.npy，已按数据形状自动构建网格。")

lat_idx, lon_idx, weights = get_nino34_indices_and_weights(lat_grid, lon_grid)

# --- 读取并计算区域平均 ---
results = []

# 获取所有文件并按时间排序 (1980_1, 1980_2...)
file_list = glob.glob(os.path.join(input_dir, "*.npy"))
# 排序逻辑：先按年份排，再按月份排
file_list.sort(key=lambda x: [int(i) for i in os.path.basename(x).replace('.npy','').split('_')])

for file_path in file_list:
    # 从文件名解析时间
    file_name = os.path.basename(file_path).replace('.npy', '')
    year, month = map(int, file_name.split('_'))
    
    # 加载数据 [Lat, Lon]
    data = np.load(file_path)
    
    weighted_mean = nino34_weighted_mean(data, lat_idx, lon_idx, weights)
        
    results.append({
        'Year': year,
        'Month': month,
        'SST_Raw': weighted_mean
    })

# --- 计算距平 (Anomaly) ---
df = pd.DataFrame(results)

# 计算每个月的多年平均 (Climatology)
monthly_climatology = df.groupby('Month')['SST_Raw'].transform('mean')

# 计算指数：当前值 - 气候态
df['Nino34_Index'] = df['SST_Raw'] - monthly_climatology

# --- 导出结果 ---
df.to_csv(output_file, index=False)

print(f"处理完成，结果已保存至 {output_file}")
print(df.head(12)) # 打印第一年的数据预览