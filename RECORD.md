# 模型架构

ENSO 与**深海活动密切相关**，而与**海表、风、大气等变量**关联较弱。
坐标系：
LAT:[-63.5,63.5]
LON:[0.5, 359.5】

## 模型结构

![Snipaste_2026-01-09_14-41-52](D:\lty\硕士\组会\pictures\Snipaste_2026-01-09_14-41-52.png)

- **深海层输入**：温度、盐度、东西向/南北向海水流速；
- **海表+大气输入**：海面风速（东西向、南北向）；
- 使用**多变量专用编码解码器**处理深海数据；
- 利用**扩散模型**模拟外部扰动（海表、风、大气）对深海的影响；
- 最终融合输出预测结果。

## 深海子模型：多变量 Encoder-Decoder

- 输入：历史深海状态序列（滑动窗口）
- 输出：未来深海状态（自回归或直接预测）
- 特点：通道专用卷积 / 注意力机制，保留变量间物理关系

## 扰动子模型：条件扩散

- 输入：当前深海状态 + 海表风场

- 生成：扰动后的深海状态增量
- 优势：能建模复杂概率分布，符合物理扰动的随机性

## 其他

**数据处理细节**

- 数据集为多时空文件，格式为 `x = {time, 经度, 纬度}`；
- 每个空间点对应一组观测值：
  - 深海：`y₁ = {温度, 盐度, 东西向流速, 南北向流速}`
  - 风、大气：`y₂ = {海面风速（东西向, 南北向）}`
  - 海表：`y3 = {温度, 盐度}`
- 注意事项：
  - 海洋与大气数据分辨率不同，需插值至统一网格；
  - 所有数据需归一化到相同单位；
  - 深海数据包含不同深度层（如 100m, 200m, ...）。

# 训练方式

**两阶段训练法 (Two-Stage Training)**，这是最稳妥且易于收敛的方案。

**第一阶段：深海子模型预训练**

- **目标**：训练 Encoder-Fusion-Decoder，让它能够精准地捕捉深海状态的长周期趋势。
- **输入**：历史深海数据。
- **输出**：未来深海状态的预测结果 $\hat{y}_{deep}$。
- **Loss**：标准的回归损失（如 RMSE 或 L1）。
- **关键点**：此时不考虑大气强迫，目标是让模型学出一个“在理想物理状态下，深海应该如何自然演化”。

**第二阶段：扰动子模型（扩散模型）联合训练**

在这一步，我们将深海子模型的输出作为“条件（Condition）”。

1. **冻结阶段**：你可以选择先冻结深海子模型的参数，只训练扰动子模型。

2. **扩散逻辑**：

   - **Conditioning**：将“第一阶段预测的 $\hat{y}_{deep}$” + “海表/大气强迫数据”拼接或通过注意力机制喂给扩散模型。
   - **Target**：扩散模型要学习的是“真实观测值 $y_{true}$”与“物理底稿 $\hat{y}_{deep}$”之间的**差值（残差）**或直接通过条件生成最终的 $SST$。

3. **损失函数**：使用扩散模型的常用损失 
   $$
   L = ||\epsilon - \epsilon_\theta(x_t, t, condition)||^2
   $$

- **输入**：Stage-1预测的深海数据 + 海表/大气强迫。
- **输出**：深海-海表残差。

# 预测

$$
N_{sample}=N_{time} −input_{steps}−predict_{steps}+1
$$

各参数含义：

- N<sub>time</sub>：评估数据时间轴总长度（总月数）。
  例：2010_01 到 2023_12 共 168 个月。
- `input_steps`：每个样本作为输入用到的历史步数。
  你现在是 `1`，表示每次只拿 1 个月作为输入。
- predict_steps：每个样本要预测/对齐标签的未来步数窗口长度。
  你现在是 `24`，表示为每个起报时刻需要预留 24 个月未来窗口。
- `+1`：滑窗计数的边界修正。
  因为起点是“包含首个位置”的计数，不是纯差值。

# 数据集
## CMIP6

```
Data var: thetao
Lon name: lon size: 720 min: 0.0 max: 359.5
Lat name: lat size: 464 min: -81.5 max: 90.0
Lon res: 0.5 Lat res: None
Lev name: lev size: 29 min: 5.0 max: 879.286376953125
Time name: time size: 60
Var dims: ('member_id', 'dcpp_init_year', 'time', 'lev', 'lat', 'lon') shape: (1, 1, 60, 29, 464, 720)
```

## Glo12V1

```
NC file: glo12/glo12_1993-02.nc
Data var: thetao
Lon name: longitude size: 4320 min: -180.0 max: 179.9166717529297
Lat name: latitude size: 2041 min: -80.0 max: 90.0
Lon res: None Lat res: None
Lev name: depth size: 35 min: 0.49402499198913574 max: 902.3392944335938
Time name: time size: 1
Var dims: ('time', 'depth', 'latitude', 'longitude') shape: (1, 35, 2041, 4320)
```



## ERA5

```
NC file: era5/era5_1993-02.nc
Data var: thetao
Lon name: longitude size: 1440 min: 0.0 max: 359.75
Lat name: latitude size: 721 min: -90.0 max: 90.0
Lon res: 0.25 Lat res: -0.25
Lev coord not found
Time coord not found
Var not found in dataset
```