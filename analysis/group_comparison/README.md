# 组间统计分析

对 ADHD 单纯组 (adhd)、ADHD 共患阅读困难组 (com)、正常发展组 (td) 的 EEG 指标进行三组统计比较。

## 统计思路

采用**被试水平**的分析策略：

1. 每位被试的多个 epoch 数据先在被试内部平均，得到该被试的单一指标值
2. 以被试为观测单位进行组间比较，避免 epoch 间的非独立性问题
3. 每个被试贡献一个数据点，样本量 = 被试数

```
epoch_1 ─┐
epoch_2 ─┤→ 被试内平均 → 被试级指标值 → 三组 Omnibus 检验 → 事后两两比较
epoch_3 ─┘
```

## 分析内容

| 指标类型 | 具体指标 | 数据来源 |
|----------|----------|----------|
| 连接性 | 聚类系数、路径长度、小世界系数 σ、平均连接强度 (PLI) × 5频段 | `connectivity_analysis/{group}/results/small_world_metrics.csv` |
| 频域 | Delta、Theta、Alpha、Beta、Gamma 频段功率 | `频域分析/{group}/results/subject_band_powers.csv` |

## 统计方法

### Omnibus 检验 (三组整体差异)
- 正态性检验: Shapiro-Wilk (每组)
- 方差齐性: Levene 检验
- 三组均正态 + 方差齐 → **单因素 ANOVA** (η²)
- 三组均正态 + 方差不齐 → **Welch's ANOVA** (η²)
- 任一组非正态 → **Kruskal-Wallis** (ε²)

### 事后检验 (仅 Omnibus 显著时)
- 参数 + 方差齐 → **Tukey HSD**
- 参数 + 方差不齐 → **Games-Howell**
- 非参数 → **Dunn 检验** (Bonferroni 校正)
- 每对计算 **Cohen's d**

### 多重比较校正
- FDR (Benjamini-Hochberg) 应用于所有 Omnibus p 值

## 输出

结果保存至 `reports/comparison/`：

- `connectivity_omnibus_stats.csv` — 连接性 Omnibus 检验结果
- `frequency_omnibus_stats.csv` — 频域 Omnibus 检验结果
- `posthoc_stats.csv` — 事后两两比较结果 (仅 omnibus 显著时)
- `group_comparison_report.md` — 完整分析报告
- `figures/` — 箱线图 (每指标×频段)、显著性热图、效应量森林图

## 运行

```bash
python group_statistical_analysis.py
# 或在 VS Code / Jupyter 中以交互模式逐 cell 运行
```

## 前置依赖

需先完成以下分析，生成对应结果文件：

1. **频域分析** (`analysis/频域分析/frequency_analysis_group.ipynb`) — 需对 adhd/com/td 三组分别运行，生成 `subject_band_powers.csv`
2. **连接性分析** (`analysis/connectivity_analysis/rest_connectivity.ipynb`) — 需对三组分别运行
