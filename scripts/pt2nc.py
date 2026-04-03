import torch
import numpy as np
import xarray as xr
from netCDF4 import Dataset
import os

# 文件路径
pt_file = '../output/predict/exp1_fix/preds/tos.pt'
output_nc = '../output/predict/exp1_fix/preds/tos.nc'

# 确保输出目录存在
os.makedirs(os.path.dirname(output_nc), exist_ok=True)

# 加载.pt文件
print("正在加载.pt文件...")
data = torch.load(pt_file, map_location='cpu', weights_only=False)

# 转换为numpy数组（如果是张量）
if isinstance(data, torch.Tensor):
    data_np = data.numpy()
    print(f"数据形状: {data_np.shape}")
    
    # 方法1: 使用xarray创建带坐标的NetCDF
    print("\n方法1: 使用xarray创建NetCDF...")
    
    # 根据数据维度创建坐标
    if len(data_np.shape) == 4:  # [time, channel, lat, lon] 或 [batch, time, lat, lon]
        # 假设常见的维度顺序
        dims = ['time', 'lat', 'lon']
        if data_np.shape[1] in [1, 3]:  # 可能是通道维度
            dims = ['time', 'channel', 'lat', 'lon']
            coords = {
                'time': np.arange(data_np.shape[0]),
                'channel': np.arange(data_np.shape[1]),
                'lat': np.linspace(90, -90, data_np.shape[2]),
                'lon': np.linspace(0, 360, data_np.shape[3])
            }
        else:
            coords = {
                'time': np.arange(data_np.shape[0]),
                'lat': np.linspace(90, -90, data_np.shape[1]),
                'lon': np.linspace(0, 360, data_np.shape[2])
            }
    
    elif len(data_np.shape) == 3:  # [time, lat, lon]
        dims = ['time', 'lat', 'lon']
        coords = {
            'time': np.arange(data_np.shape[0]),
            'lat': np.linspace(90, -90, data_np.shape[1]),
            'lon': np.linspace(0, 360, data_np.shape[2])
        }
    
    elif len(data_np.shape) == 2:  # [lat, lon]
        dims = ['lat', 'lon']
        coords = {
            'lat': np.linspace(90, -90, data_np.shape[0]),
            'lon': np.linspace(0, 360, data_np.shape[1])
        }
    
    else:
        # 如果是其他维度，直接保存为不带坐标的数据
        da = xr.DataArray(data_np, name='tos')
        da.to_netcdf(output_nc)
        print(f"已保存到: {output_nc}")
        
    # 创建DataArray并保存
    if len(data_np.shape) in [2, 3, 4]:
        da = xr.DataArray(
            data_np,
            dims=dims,
            coords=coords,
            name='tos',
            attrs={
                'long_name': 'Sea Surface Temperature',
                'units': '°C',
                'description': 'Converted from PyTorch .pt file',
                'source_file': os.path.basename(pt_file)
            }
        )
        da.to_netcdf(output_nc)
        print(f"已保存到: {output_nc}")
        
        # 显示转换后的文件信息
        print("\n转换后的文件信息:")
        print(da)

# 如果是字典，尝试提取关键数据
elif isinstance(data, dict):
    print(f"字典包含的键: {data.keys()}")
    
    # 查找可能包含SST数据的键
    possible_keys = ['preds', 'predictions', 'output', 'target', 'tos', 'sst', 'data']
    sst_data = None
    sst_key = None
    
    for key in possible_keys:
        if key in data:
            value = data[key]
            if isinstance(value, torch.Tensor):
                sst_data = value.numpy()
                sst_key = key
                break
    
    if sst_data is not None:
        print(f"找到SST数据，键: '{sst_key}', 形状: {sst_data.shape}")
        
        # 创建xarray Dataset
        ds = xr.Dataset()
        
        # 添加主要数据
        ds['tos'] = xr.DataArray(
            sst_data,
            dims=[f'dim_{i}' for i in range(len(sst_data.shape))],
            attrs={'long_name': 'Sea Surface Temperature', 'units': '°C'}
        )
        
        # 添加其他元数据
        for key, value in data.items():
            if key != sst_key:
                if isinstance(value, (str, int, float, list, np.ndarray)):
                    ds.attrs[key] = str(value)
                elif isinstance(value, torch.Tensor):
                    ds[f'{key}_meta'] = xr.DataArray(
                        value.numpy(),
                        attrs={'description': f'Metadata from key: {key}'}
                    )
        
        # 保存
        ds.to_netcdf(output_nc)
        print(f"已保存到: {output_nc}")
        print(f"\n数据集信息:\n{ds}")
    
    else:
        print("未找到明显的SST数据，保存所有字典内容")
        # 保存所有张量到不同的变量
        ds = xr.Dataset()
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                ds[key] = xr.DataArray(
                    value.numpy(),
                    dims=[f'{key}_dim_{i}' for i in range(len(value.shape))]
                )
            else:
                ds.attrs[key] = str(value)
        ds.to_netcdf(output_nc)
        print(f"已保存所有数据到: {output_nc}")

# 方法2: 使用netCDF4直接创建（更精细的控制）
print("\n方法2: 使用netCDF4创建（提供更多控制）...")
nc_file2 = '../output/predict/exp1_fix/preds/tos_advanced.nc'

if isinstance(data, torch.Tensor):
    data_np = data.numpy()
    
    # 创建NetCDF文件
    with Dataset(nc_file2, 'w', format='NETCDF4') as nc:
        # 创建维度
        if len(data_np.shape) == 4:
            nc.createDimension('time', data_np.shape[0])
            nc.createDimension('channel', data_np.shape[1])
            nc.createDimension('lat', data_np.shape[2])
            nc.createDimension('lon', data_np.shape[3])
            
            # 创建变量
            times = nc.createVariable('time', 'f4', ('time',))
            channels = nc.createVariable('channel', 'i4', ('channel',))
            lats = nc.createVariable('lat', 'f4', ('lat',))
            lons = nc.createVariable('lon', 'f4', ('lon',))
            tos = nc.createVariable('tos', 'f4', ('time', 'channel', 'lat', 'lon'), 
                                   fill_value=np.nan)
            
            # 添加坐标数据
            times[:] = np.arange(data_np.shape[0])
            channels[:] = np.arange(data_np.shape[1])
            lats[:] = np.linspace(90, -90, data_np.shape[2])
            lons[:] = np.linspace(0, 360, data_np.shape[3])
            
        elif len(data_np.shape) == 3:
            nc.createDimension('time', data_np.shape[0])
            nc.createDimension('lat', data_np.shape[1])
            nc.createDimension('lon', data_np.shape[2])
            
            times = nc.createVariable('time', 'f4', ('time',))
            lats = nc.createVariable('lat', 'f4', ('lat',))
            lons = nc.createVariable('lon', 'f4', ('lon',))
            tos = nc.createVariable('tos', 'f4', ('time', 'lat', 'lon'), 
                                   fill_value=np.nan)
            
            times[:] = np.arange(data_np.shape[0])
            lats[:] = np.linspace(90, -90, data_np.shape[1])
            lons[:] = np.linspace(0, 360, data_np.shape[2])
            
        else:
            # 简化处理
            dim_names = [f'dim{i}' for i in range(len(data_np.shape))]
            for i, size in enumerate(data_np.shape):
                nc.createDimension(dim_names[i], size)
            tos = nc.createVariable('tos', 'f4', tuple(dim_names), fill_value=np.nan)
        
        # 写入数据
        tos[:] = data_np
        
        # 添加属性
        tos.units = '°C'
        tos.long_name = 'Sea Surface Temperature'
        tos.source = os.path.basename(pt_file)
        
        print(f"高级版NetCDF已保存到: {nc_file2}")

print(f"\n转换完成！文件已保存。")