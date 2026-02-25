# 组间统计分析

对 ADHD 共患组 (com)、ADHD 单纯组 (adhd)、正常发展组 (td) 的 EEG 指标进行组间统计比较。

## 统计思路

采用**被试水平**的分析策略：

1. 每位被试的多个 epoch 数据先在被试内部进行平均，得到该被试的单一指标值
2. 以被试为观测单位进行组间比较，避免 epoch 间的非独立性问题
3. 每个被试贡献一个数据点，样本量 = 被试数

```
epoch_1 ─┐
epoch_2 ─┤→ 被试内平均 → 被试级指标值 → 组间比较
epoch_3 ─┘
```

## 分析内容

| 指标类型 | 具体指标 | 数据来源 |
|----------|----------|----------|
| 连接性 | 聚类系数、路径长度、小世界系数 σ、平均连接强度 (PLI) | `connectivity_analysis/{group}/results/` |
| 频域 | Delta、Theta、Alpha、Beta、Gamma 频段功率 | `频域分析/{group}/results/` |

## 统计方法

- **正态性检验**: Shapiro-Wilk
- **组间比较**: 正态 → 独立样本 t 检验；非正态 → Mann-Whitney U 检验
- **多重比较校正**: FDR (Benjamini-Hochberg)
- **效应量**: Cohen's d

## 输出

结果保存至 `d:\LYW\REST_COM\reports\comparison\`：

- `connectivity_stats.csv` — 连接性指标统计检验结果
- `group_comparison_report.md` — 完整分析报告
- `figures/` — 箱线图、效应量森林图、显著性热图

## 运行

```bash
# 作为 Python 脚本运行（.py 格式的 Jupyter 百分号脚本）
python group_statistical_analysis.py

# 或在 VS Code / Jupyter 中以交互模式逐 cell 运行
```

## 前置依赖

需先完成以下分析，生成对应结果文件：

1. **频域分析** (`analysis/频域分析/frequency_analysis_group.ipynb`)
2. **连接性分析** (`analysis/connectivity_analysis/rest_connectivity.ipynb`)
