import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import matplotlib.dates as mdates

# --- 配置 ---
input_dir = './predict/exp2_fix'
output_image = './predict/exp2_fix/Asst_raw_comparison.png'

# 1. 获取并排序所有 CSV 文件
file_list = glob.glob(os.path.join(input_dir, "*.csv"))
file_list.sort()

if not file_list:
    print(f"错误：在 {os.path.abspath(input_dir)} 下未找到 CSV 文件！")
    exit()

# 设置绘图风格
plt.style.use('seaborn-v0_8-whitegrid') # 或者使用 'ggplot'
fig, ax = plt.subplots(figsize=(13, 7))

# 2. 循环读取并绘图
for file_path in file_list:
    file_name = os.path.basename(file_path)
    
    try:
        df = pd.read_csv(file_path)
        
        # 检查列名 (兼容大小写)
        cols = {c.upper(): c for c in df.columns}
        if not {'YEAR', 'MONTH', 'SST_RAW'}.issubset(cols.keys()):
            print(f"跳过 {file_name}: 缺少必要列")
            continue
            
        # 构建时间轴
        # 使用 assign 临时创建 Day 列以便 pd.to_datetime 识别
        df['Date'] = pd.to_datetime(pd.DataFrame({
            'year': df[cols['YEAR']],
            'month': df[cols['MONTH']],
            'day': 1
        }))
        
        # 排序，防止连线混乱
        df = df.sort_values('Date')
        
        # 绘图
        ax.plot(df['Date'], df[cols['SST_RAW']], 
                label=file_name.replace('.csv', ''), 
                linewidth=1.5, marker='o', markersize=3, alpha=0.8)
                
    except Exception as e:
        print(f"处理文件 {file_name} 时出错: {e}")

# 3. 格式化图表
ax.set_title('Regional Average SST (Raw) Comparison', fontsize=15, pad=20)
ax.set_xlabel('Time (Year-Month)', fontsize=12)
ax.set_ylabel('SST_RAW (Temperature)', fontsize=12)

# 优化横坐标显示（每 12 个月显示一个刻度）
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=45)

ax.grid(True, linestyle='--', alpha=0.6)
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)

# 自动调整布局防止标签切断
plt.tight_layout()

# 4. 保存
plt.savefig(output_image, dpi=300)
print(f"绘 bird 完成！图片已保存至: {os.path.abspath(output_image)}")
plt.show()