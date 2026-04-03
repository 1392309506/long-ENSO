import xarray as xr
import pandas as pd
import numpy as np

# 1. 读取数据
# 请替换为你的文件名
ds = xr.open_dataset('sst_data.nc')

# 自动识别维度名称（处理大小写不统一的情况）
lon_name = 'lon' if 'lon' in ds.coords else 'longitude'
lat_name = 'lat' if 'lat' in ds.coords else 'latitude'

# 2. 定义 Nino 3.4 区域 (5°N-5°S, 170°W-120°W)
# 注意：如果你的经度是 0-360 格式，170°W-120°W 对应 190-240
if ds[lon_name].max() > 180:
    lon_range = [190, 240]
else:
    lon_range = [-170, -120]

nino34_area = ds.sel(
    **{lat_name: slice(5, -5), # 纬度从北到南
       lon_name: slice(lon_range[0], lon_range[1])}
)

# 3. 计算区域加权平均
# 考虑纬度造成的网格面积差异，进行余弦加权
weights = np.cos(np.deg2rad(nino34_area[lat_name]))
weights.name = "weights"
nino34_weighted = nino34_area['sst'].weighted(weights)
nino34_index_raw = nino34_weighted.mean(dim=[lat_name, lon_name])

# 4. 计算距平 (Anomalies)
# 计算每个月的多年平均值（气候态）
climatology = nino34_index_raw.groupby("time.month").mean("time")
# 减去对应月份的气候平均值
nino34_anomaly = nino34_index_raw.groupby("time.month") - climatology

# 5. 导出为表格
df = nino34_anomaly.to_dataframe(name='Nino34_Index').reset_index()

# 保存为 CSV
output_file = 'nino34_results.csv'
df.to_csv(output_file, index=False)

print(f"计算完成！结果已保存至: {output_file}")
print(df.head())