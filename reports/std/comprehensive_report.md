# REST EEG 分析报告 - 正常被试组 (std)

**生成时间**: 2026-02-03 19:15
**分析组**: 正常被试 (Standard/Control)
**被试数量**: 12

---

## 1. 数据概况

### 1.1 被试信息

| 指标 | 值 |
|------|-----|
| 被试数 | 12 |
| 通道数 | 60 |
| 采样率 | 500 Hz |
| Epoch时长 | 10 秒 |

### 1.2 数据质量摘要

| subject_id   |   n_epochs |   n_channels |   sfreq |   n_times |   tmin |   tmax |   duration |
|:-------------|-----------:|-------------:|--------:|----------:|-------:|-------:|-----------:|
| std_006      |         20 |           60 |     500 |      5000 |      0 |  9.998 |      9.998 |
| std_010      |          5 |           60 |     500 |      5000 |      0 |  9.998 |      9.998 |
| std_068      |         20 |           60 |     500 |      5000 |      0 |  9.998 |      9.998 |
| std_079      |         28 |           60 |     500 |      5000 |      0 |  9.998 |      9.998 |
| std_080      |          9 |           60 |     500 |      5000 |      0 |  9.998 |      9.998 |
| std_093      |         26 |           60 |     500 |      5000 |      0 |  9.998 |      9.998 |
| std_100      |          2 |           60 |     500 |      5000 |      0 |  9.998 |      9.998 |
| std_108      |         21 |           60 |     500 |      5000 |      0 |  9.998 |      9.998 |
| std_109      |         11 |           60 |     500 |      5000 |      0 |  9.998 |      9.998 |
| std_112      |          8 |           60 |     500 |      5000 |      0 |  9.998 |      9.998 |
| std_127      |         15 |           60 |     500 |      5000 |      0 |  9.998 |      9.998 |
| std_133      |         23 |           60 |     500 |      5000 |      0 |  9.998 |      9.998 |

---

## 2. 频域分析结果

### 2.1 分析方法
- **PSD计算**: Welch方法
- **频段定义**: Delta (1-4Hz), Theta (4-8Hz), Alpha (8-13Hz), Beta (13-30Hz), Gamma (30-40Hz)
- **SNR计算**: 邻近频率噪声估计

### 2.2 频段功率统计

| 频段   | 频率范围   |   平均功率 |   标准差 |
|:-------|:-----------|-----------:|---------:|
| Delta  | 1-4 Hz     |   1.61e-11 | 2.11e-11 |
| Theta  | 4-8 Hz     |   2.79e-12 | 2.21e-12 |
| Alpha  | 8-13 Hz    |   2.31e-12 | 2.71e-12 |
| Beta   | 13-30 Hz   |   4.03e-13 | 3.96e-13 |
| Gamma  | 30-40 Hz   |   1.86e-13 | 2.8e-13  |

### 2.3 可视化

#### 组平均功率谱密度
![Group PSD](../analysis/频域分析/std/figures/group_psd.png)

#### 组平均信噪比
![Group SNR](../analysis/频域分析/std/figures/group_snr.png)

#### 频段功率拓扑图
![Band Powers](../analysis/频域分析/std/figures/group_band_powers.png)

---

## 3. 连接性分析结果

### 3.1 分析方法
- **连接性指标**: Phase Lag Index (PLI)
- **频段**: Delta (1-4Hz), Theta (4-8Hz), Alpha (8-13Hz), Beta (13-30Hz), Gamma (30-45Hz)
- **网络指标**: 聚类系数 (C), 路径长度 (L), 小世界系数 (σ)
- **随机网络基线**: 50个随机图

### 3.2 小世界网络指标

#### 描述性统计
| Unnamed: 0   | clustering   | clustering.1   | clustering.2   | clustering.3   | path_length   | path_length.1   | path_length.2   | path_length.3   | sigma   | sigma.1   | sigma.2   | sigma.3   | avg_connectivity   | avg_connectivity.1   | avg_connectivity.2   | avg_connectivity.3   |
|:-------------|:-------------|:---------------|:---------------|:---------------|:--------------|:----------------|:----------------|:----------------|:--------|:----------|:----------|:----------|:-------------------|:---------------------|:---------------------|:---------------------|
| nan          | mean         | std            | min            | max            | mean          | std             | min             | max             | mean    | std       | min       | max       | mean               | std                  | min                  | max                  |
| freq_band    | nan          | nan            | nan            | nan            | nan           | nan             | nan             | nan             | nan     | nan       | nan       | nan       | nan                | nan                  | nan                  | nan                  |
| alpha        | 0.535        | 0.0874         | 0.3606         | 0.6657         | 0.271         | 0.0839          | 0.2061          | 0.4984          | 0.3745  | 0.0808    | 0.2197    | 0.5319    | 0.2738             | 0.0839               | 0.2066               | 0.5036               |
| beta         | 0.6231       | 0.11           | 0.396          | 0.7815         | 0.2564        | 0.0986          | 0.1854          | 0.5144          | 0.4752  | 0.1287    | 0.2534    | 0.672     | 0.2566             | 0.0985               | 0.1854               | 0.5144               |
| delta        | 0.4506       | 0.0776         | 0.3158         | 0.5541         | 0.2527        | 0.0899          | 0.187           | 0.4837          | 0.3407  | 0.0682    | 0.2054    | 0.4267    | 0.2602             | 0.0981               | 0.1938               | 0.5257               |
| gamma        | 0.5296       | 0.1134         | 0.3802         | 0.7251         | 0.2542        | 0.1029          | 0.165           | 0.5073          | 0.4119  | 0.1373    | 0.231     | 0.7641    | 0.2545             | 0.1028               | 0.1655               | 0.5074               |
| theta        | 0.5023       | 0.0401         | 0.4414         | 0.5863         | 0.2527        | 0.1023          | 0.1746          | 0.5242          | 0.3932  | 0.0944    | 0.2031    | 0.5179    | 0.256              | 0.1052               | 0.1765               | 0.5393               |

#### 关键发现
| 频段 | 聚类系数 (C) | 路径长度 (L) | 小世界系数 (σ) | 平均连接强度 |
|------|-------------|-------------|---------------|-------------|
| Delta | 0.4506 | 0.2527 | 0.3407 | 0.2602 |
| Theta | 0.5023 | 0.2527 | 0.3932 | 0.2560 |
| Alpha | 0.5350 | 0.2710 | 0.3745 | 0.2738 |
| Beta | 0.6231 | 0.2564 | 0.4752 | 0.2566 |
| Gamma | 0.5296 | 0.2542 | 0.4119 | 0.2545 |

### 3.3 可视化

#### 各频段连接性热图
![Connectivity Heatmaps](../analysis/connectivity_analysis/std/figures/connectivity_heatmaps.png)

#### 网络指标箱线图
![Network Metrics](../analysis/connectivity_analysis/std/figures/network_metrics_boxplot.png)

#### 频段连接强度对比
![Connectivity by Band](../analysis/connectivity_analysis/std/figures/connectivity_by_band.png)

#### Alpha频段个体连接性
![Individual Alpha](../analysis/connectivity_analysis/std/figures/connectivity_individual_alpha.png)

---

## 4. 主要发现

### 4.1 频域特征
- 正常被试组显示典型的静息态EEG频谱特征
- Alpha频段 (8-13 Hz) 功率占主导地位

### 4.2 网络特征
- **Alpha频段**:
  - 平均聚类系数: 0.5350
  - 小世界系数 σ: 0.3745 
  - 平均连接强度: 0.2738

### 4.3 小世界特性解读
| σ 值范围 | 解读 |
|---------|------|
| σ < 1 | 非小世界网络 |
| σ ≈ 1 | 边界状态 |
| σ > 1 | 小世界特性 |
| σ > 3 | 强小世界特性 |

---

## 5. 输出文件

### 频域分析
- `analysis/频域分析/std/results/group_frequency_results.pkl`
- `analysis/频域分析/std/results/frequency_stats.csv`
- `analysis/频域分析/std/figures/group_psd.png`

### 连接性分析
- `analysis/connectivity_analysis/std/results/connectivity_results.npz`
- `analysis/connectivity_analysis/std/results/small_world_metrics.csv`
- `analysis/connectivity_analysis/std/figures/connectivity_heatmaps.png`

---

## 6. 后续分析建议

1. **组间对比**: 与共患被试组 (com) 进行统计比较
2. **相关分析**: 与行为/临床数据进行相关分析
3. **动态连接性**: 考虑时变连接性分析
4. **源定位**: 结合源定位分析脑区间连接

---

*报告由 REST EEG Pipeline 自动生成*
