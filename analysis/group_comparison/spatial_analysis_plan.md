# 空间维度分析方案

**生成时间**: 2026-03-14
**目的**: 探索全脑平均可能掩盖的局部脑区组间差异
**背景**: 当前全脑59通道平均分析未发现显著组间差异，但ADHD神经异常可能是区域特异性的

---

## 当前分析问题

### 全脑平均的局限性

```python
# 当前代码（Section 8）
power = psds[trial_idx, :, freq_mask].mean()  # 所有59通道平均
```

**问题**:
- ADHD的神经异常可能是**局部的**（如前额叶、顶叶）
- 全脑平均会**稀释**局部差异
- 如果ADHD组在前额叶Delta降低，但在枕叶Delta升高，平均后可能无差异

### 文献支持

ADHD的EEG研究常报告**区域特异性**差异：

1. **前额叶Theta增加** - 执行功能缺陷
2. **中央区Beta降低** - 运动控制问题
3. **后部Alpha异常** - 注意网络

---

## 方案A: ROI（感兴趣区）分析 ⭐ 推荐优先

### 理论依据

- **理论驱动**: 基于ADHD神经机制假设
- **统计功效**: 减少多重比较负担（5个ROI vs 59个通道）
- **可解释性**: 结果容易与文献对比
- **实施简单**: 只需修改Section 8的代码

### ROI定义

```python
ROI_CHANNELS = {
    'frontal': ['Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8', 'Fz', 'FC1', 'FC2', 'FC5', 'FC6'],
    'central': ['C3', 'C4', 'Cz', 'CP1', 'CP2', 'CP5', 'CP6'],
    'parietal': ['P3', 'P4', 'Pz', 'P7', 'P8', 'PO3', 'PO4'],
    'occipital': ['O1', 'O2', 'Oz', 'PO7', 'PO8'],
    'temporal': ['T7', 'T8', 'TP9', 'TP10', 'FT9', 'FT10']
}
```

### 实施步骤

1. **修改Section 8**: 添加ROI计算逻辑
   - 对每个ROI分别计算功率
   - 数据结构: `[subject_id, group, trial_idx, roi, band, power, log_power]`

2. **修改Section 9**: 对每个ROI分别做LME分析
   - 模型: `log_power ~ C(group, Treatment('td')) + (1|subject_id)`
   - 嵌套循环: ROI × Band

3. **修改Section 10**: ROI-specific方差异质性检验

4. **修改Section 12**: 添加地形图可视化
   - 使用`mne.viz.plot_topomap()`
   - 显示显著ROI的空间分布

5. **FDR校正**: 5个ROI × 5个频段 = 25次检验

### 预期结果

- **如果前额叶Theta有差异** → 支持执行功能假设
- **如果中央区Beta有差异** → 支持运动控制假设
- **如果仍无差异** → 说明样本在频域功率上真的无组间差异

### 统计考虑

- **多重比较**: 25次检验，使用FDR校正（α=0.05）
- **效应量**: 报告Cohen's d或β系数
- **功效分析**: 如果仍无显著，计算检测中等效应量所需样本量

---

## 方案B: 探索性通道分析

### 理论依据

- **数据驱动**: 不预设ROI，让数据说话
- **全面性**: 不遗漏任何可能的空间模式
- **发现性**: 可能发现文献未报告的区域

### 实施步骤

1. **对每个通道单独检验**

```python
channel_results = []
for ch_idx, ch_name in enumerate(epochs.ch_names):
    for band in BANDS.keys():
        # 提取该通道该频段的所有trial数据
        channel_band_df = trial_df[(trial_df['channel']==ch_name) & (trial_df['band']==band)]

        # LME分析
        model = smf.mixedlm("log_power ~ C(group, Treatment('td'))",
                           data=channel_band_df,
                           groups=channel_band_df['subject_id'])
        result = model.fit()

        channel_results.append({
            'channel': ch_name,
            'band': band,
            'p_adhd': result.pvalues['C(group, Treatment(\'td\'))[T.adhd]'],
            'p_com': result.pvalues['C(group, Treatment(\'td\'))[T.com]']
        })
```

2. **FDR校正**: 59通道 × 5频段 = 295次检验

3. **空间聚类**: 将显著通道聚类为ROI
   - 使用邻接矩阵定义通道邻近性
   - 显著通道形成连续区域

4. **可视化**: 地形图显示p值分布

### 预期结果

- **发现显著通道簇** → 定义为新的ROI，在独立样本验证
- **无显著通道** → 确认无空间特异性差异

### 统计考虑

- **多重比较**: 295次检验，FDR校正非常严格
- **假阳性控制**: 可能需要cluster-based permutation test
- **验证**: 显著结果需要在独立样本或交叉验证中确认

---

## 方案C: 数据驱动的空间聚类

### 理论依据

- **降维**: 从59维通道空间降到少数主成分
- **无偏**: 不预设ROI或通道
- **模式发现**: 可能发现非传统的空间模式

### 实施步骤

1. **PCA降维**

```python
from sklearn.decomposition import PCA

# 构建被试×通道×频段矩阵
subject_channel_band = []
for subject_id in trial_df['subject_id'].unique():
    subject_data = trial_df[trial_df['subject_id']==subject_id]
    for band in BANDS.keys():
        band_data = subject_data[subject_data['band']==band]
        # 每个通道的平均功率
        channel_powers = band_data.groupby('channel')['log_power'].mean().values
        subject_channel_band.append(channel_powers)

# PCA
pca = PCA(n_components=5)
spatial_components = pca.fit_transform(subject_channel_band)  # (n_subjects*n_bands, 5)
```

2. **对每个主成分做组间比较**

```python
# 将主成分分配回被试和频段
component_df = pd.DataFrame({
    'subject_id': ...,
    'band': ...,
    'group': ...,
    'PC1': spatial_components[:, 0],
    'PC2': spatial_components[:, 1],
    ...
})

# LME分析
for pc in ['PC1', 'PC2', 'PC3', 'PC4', 'PC5']:
    model = smf.mixedlm(f"{pc} ~ C(group, Treatment('td'))",
                       data=component_df,
                       groups=component_df['subject_id'])
    result = model.fit()
```

3. **解释主成分**: 可视化loading map

```python
# PC1的loading（每个通道的权重）
loading_map = pca.components_[0, :]  # 59个通道的权重
mne.viz.plot_topomap(loading_map, epochs.info, show=True)
```

### 预期结果

- **某个PC显著** → 该PC代表的空间模式有组间差异
- **所有PC不显著** → 确认无空间模式差异

### 统计考虑

- **多重比较**: 5个PC × 5个频段 = 25次检验
- **解释性**: PC的神经解剖学意义可能不清晰
- **验证**: 需要在独立样本验证PC的稳定性

---

## 实施顺序建议

### 第一阶段: ROI分析（方案A）

**理由**:
- 理论驱动，结果可解释
- 统计功效最高
- 实施最简单

**如果发现显著差异**:
- 报告具体ROI和频段
- 绘制地形图
- 与文献对比

**如果仍无显著差异**:
- 进入第二阶段

### 第二阶段: 探索性通道分析（方案B）

**理由**:
- 全面搜索，不遗漏
- 可能发现ROI分析错过的模式

**如果发现显著通道簇**:
- 定义为新ROI
- 在独立样本验证

**如果仍无显著差异**:
- 进入第三阶段或结束

### 第三阶段: PCA分析（方案C，可选）

**理由**:
- 探索非传统空间模式
- 学术价值（方法学创新）

**如果发现显著PC**:
- 解释PC的神经意义
- 报告为探索性发现

---

## 数据结构变化

### 当前数据结构

```python
trial_df.columns = ['subject_id', 'group', 'trial_idx', 'band', 'power', 'log_power']
```

### 方案A数据结构

```python
trial_df_roi.columns = ['subject_id', 'group', 'trial_idx', 'roi', 'band', 'power', 'log_power']
```

### 方案B数据结构

```python
trial_df_channel.columns = ['subject_id', 'group', 'trial_idx', 'channel', 'band', 'power', 'log_power']
```

### 方案C数据结构

```python
component_df.columns = ['subject_id', 'group', 'band', 'PC1', 'PC2', 'PC3', 'PC4', 'PC5']
```

---

## 文献参考方向

### ADHD EEG频域研究

1. **Frontal Theta**:
   - Loo & Makeig (2012) - Clinical Neurophysiology
   - 执行功能缺陷的神经标记

2. **Central Beta**:
   - Barry et al. (2003) - Clinical Neurophysiology
   - 运动准备和抑制控制

3. **Posterior Alpha**:
   - Clarke et al. (2001) - Clinical Neurophysiology
   - 注意网络成熟度

### 方法学参考

1. **ROI分析**:
   - Delorme & Makeig (2004) - EEGLAB
   - 标准ROI定义

2. **Cluster-based permutation**:
   - Maris & Oostenveld (2007) - Journal of Neuroscience Methods
   - 控制空间多重比较

---

## 下一步行动

### 立即执行

1. ✅ 创建本文档
2. ⏳ 实施方案A（ROI分析）
3. ⏳ 根据结果决定是否进入方案B/C

### 长期规划

- 如果所有方案均无显著差异 → 转向其他指标（连接性、复杂度、时频分析）
- 如果发现显著差异 → 在独立样本验证，撰写论文

---

## 记录与追踪

- **方案A状态**: 待执行
- **方案B状态**: 待定
- **方案C状态**: 待定
- **最后更新**: 2026-03-14
