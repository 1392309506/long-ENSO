import torch
import pandas as pd
import numpy as np
import json
import os
from nino34_utils import get_nino34_indices_and_weights, nino34_weighted_mean

# --- 配置路径 ---
CONST = 'lead_time_17'
data_dir = f'../output/predict/{CONST}/preds'
pt_file = os.path.join(data_dir, 'tos.pt')
json_file = f'../output/predict/{CONST}/init_times.json'
output_file = f'./lead_time_exp/{CONST}.csv'
os.makedirs(os.path.dirname(output_file), exist_ok=True)
lat_file = '../grid/lat.npy'
lon_file = '../grid/lon.npy'


def _build_grid_from_shape(shape_hw):
    h, w = shape_hw
    lat = np.linspace(-63.5, 63.5, h)
    lon = np.linspace(0.5, 359.5, w)
    return lat, lon

# 2. 加载时间信息
with open(json_file, 'r') as f:
    init_times = json.load(f)
    time_index = pd.to_datetime(init_times, format='%Y_%m')

# 3. 加载数据
print(f"正在加载数据: {pt_file}")
loaded_data = torch.load(pt_file, map_location='cpu', weights_only=False)

# --- 处理加载后的数据类型 ---
if isinstance(loaded_data, np.ndarray):
    data = torch.from_numpy(loaded_data).float()
else:
    data = loaded_data.float()

# 4. 处理维度
if data.dim() == 4: # (Time, C, Lat, Lon)
    data = data.squeeze(1)
elif data.dim() == 2: # (Lat, Lon)
    data = data.unsqueeze(0)

print(f"数据形状: {data.shape}")

# 1. 坐标与权重设置
if os.path.exists(lat_file) and os.path.exists(lon_file):
    lats = np.load(lat_file).squeeze()
    lons = np.load(lon_file).squeeze()
else:
    lats, lons = _build_grid_from_shape((data.shape[-2], data.shape[-1]))
    print("未找到 grid/lat.npy 或 grid/lon.npy，已按数据形状自动构建网格。")

lat_idx, lon_idx, weights = get_nino34_indices_and_weights(lats, lons)

# 5. 计算 Nino 3.4
nino_raw_values = np.array([
    nino34_weighted_mean(data[i].numpy(), lat_idx, lon_idx, weights)
    for i in range(data.shape[0])
])

# 6. 构建结果表格
df = pd.DataFrame({
    'Date': time_index,
    'SST_Raw': nino_raw_values
})

df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month

# 7. 计算距平 (Anomaly)
# 核心：减去该月份在当前时间序列里的气候平均态
df['Climatology'] = df.groupby('Month')['SST_Raw'].transform('mean')
df['Nino34_Index'] = df['SST_Raw'] - df['Climatology']

# 8. 导出 (增加 SST_Raw 列)
df[['Year', 'Month', 'SST_Raw', 'Nino34_Index']].to_csv(output_file, index=False)

print("-" * 30)
print(f"成功！计算完成。")
print(f"文件保存至: {os.path.abspath(output_file)}")
print(f"前 5 行预览：\n{df[['Year', 'Month', 'SST_Raw', 'Nino34_Index']].head()}")