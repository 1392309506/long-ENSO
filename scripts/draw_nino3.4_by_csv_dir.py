import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

# --- 配置 ---
input_dir = './predict/exp2_fix'
output_image = './predict/exp2_fix/Anino34_comparison.png'

# 1. 获取所有 CSV 文件
file_list = glob.glob(os.path.join(input_dir, "*.csv"))
file_list.sort() # 按名称排序

if not file_list:
    print(f"在 {input_dir} 下没找到 CSV 文件！")
    exit()

plt.figure(figsize=(12, 6))

# 2. 遍历文件并绘图
for file_path in file_list:
    file_name = os.path.basename(file_path)
    
    # 读取数据
    df = pd.read_csv(file_path)
    
    # 检查必要的列是否存在
    required_cols = {'Year', 'Month', 'Nino34_Index'}
    if not required_cols.issubset(df.columns):
        print(f"跳过文件 {file_name}: 缺少必要列 {required_cols - set(df.columns)}")
        continue
    
    # 构建时间轴 (YYYY-MM)
    # 将 Year 和 Month 合并为 datetime 对象
    df['Date'] = pd.to_datetime(df[['Year', 'Month']].assign(Day=1))
    
    # 按照时间排序确保连线正确
    df = df.sort_values('Date')
    
    # 绘图
    plt.plot(df['Date'], df['Nino34_Index'], label=file_name.replace('.csv', ''), marker='.', markersize=4, alpha=0.8)

# 3. 图表修饰
# 绘制 0 刻度线
plt.axhline(0, color='black', linewidth=1, linestyle='--')
# 绘制 厄尔尼诺/拉尼娜 阈值线 (通常为 ±0.5)
plt.axhline(0.5, color='red', linewidth=0.8, linestyle=':', alpha=0.5, label='El Nino Threshold')
plt.axhline(-0.5, color='blue', linewidth=0.8, linestyle=':', alpha=0.5, label='La Nina Threshold')

plt.title('Nino 3.4 Index Comparison', fontsize=14)
plt.xlabel('Time (Year-Month)', fontsize=12)
plt.ylabel('Nino 3.4 Index (°C)', fontsize=12)
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left') # 图例放在外面防止遮挡
plt.tight_layout()

# 4. 保存与显示
plt.savefig(output_image, dpi=300)
print(f"绘图完成！图片已保存至: {output_image}")
plt.show()