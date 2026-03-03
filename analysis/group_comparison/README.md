# 组间统计分析

对 ADHD 单纯组 (adhd)、ADHD 共患阅读困难组 (com)、正常发展组 (td) 的 EEG 指标进行三组统计比较。

## 文件结构

| 文件 | 用途 |
|------|------|
| `stats_utils.py` | 共享统计模块 (配置、检验、可视化、报告生成) |
| `nbs_utils.py` | NBS 核心函数模块 (独立样本置换检验) |
| `graph_roi_stats.ipynb` | 图论指标 + ROI 连接性: 全脑 (20检验) + ROI (45检验), 合计 65 检验统一 FDR |
| `nbs_stats.ipynb` | NBS 连接性分析: 3 组对 × 5 频段 = 15 次置换检验 |
| `frequency_stats.ipynb` | 频域功率分析: 5 频段, 5 检验独立 FDR |

## 统计思路

采用**被试水平**的分析策略：

1. 每位被试的多个 epoch 数据先在被试内部平均，得到该被试的单一指标值
2. 以被试为观测单位进行组间比较，避免 epoch 间的非独立性问题
3. 每个被试贡献一个数据点，样本量 = 被试数

```text
epoch_1 ─┐
epoch_2 ─┤→ 被试内平均 → 被试级指标值 → 三组 Omnibus 检验 → 事后两两比较
epoch_3 ─┘
```

## 分析内容

| Notebook | 指标类型 | 具体指标 | 检验数 |
|----------|----------|----------|--------|
| `graph_roi_stats.ipynb` | 全脑图论指标 | 聚类系数、路径长度、σ、平均连接强度 × 5频段 | 20 |
| `graph_roi_stats.ipynb` | ROI 连接性 (wPLI) | Paper ROIs (prefrontal, left_pot, between) + Exploratory ROIs (6区域) × 5频段 | 45 |
| `nbs_stats.ipynb` | NBS 边级别连接性 | wPLI 矩阵边级别置换检验, 3 组对 × 5 频段 | 15 |
| `frequency_stats.ipynb` | 频域功率 | Delta, Theta, Alpha, Beta, Gamma | 5 |

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

- **连接性**: 全脑 + ROI 合并 65 检验统一 FDR (Benjamini-Hochberg)
- **频域**: 5 检验独立 FDR

### NBS (Network-Based Statistic)

图论指标是全局摘要，对局部网络差异不敏感。NBS 在边级别检测组间差异，利用连接的空间聚类特性提升统计效力。

- 每条边做独立样本 t 检验 (Welch's t-test)
- 超阈值边 (|t| > threshold) 通过 BFS 找连通分量
- 置换检验 (shuffle group labels) 控制 FWER
- 参考: Zalesky, A., Fornito, A., & Bullmore, E. T. (2010). *NeuroImage*

## 输出

| Notebook | 输出目录 |
|----------|----------|
| `graph_roi_stats.ipynb` | `reports/comparison/connectivity/` |
| `nbs_stats.ipynb` | `reports/comparison/nbs/` |
| `frequency_stats.ipynb` | `reports/comparison/frequency/` |

每个目录包含:

- `*_omnibus_stats.csv` — Omnibus 检验结果
- `posthoc_stats.csv` — 事后两两比较
- `*_comparison_report.md` — 完整分析报告
- `figures/` — 箱线图、显著性热图、效应量森林图

## 运行

在 Jupyter 中分别运行两个 notebook (工作目录需为 `analysis/group_comparison/`):

1. `graph_roi_stats.ipynb` — 图论指标 + ROI 连接性
2. `nbs_stats.ipynb` — NBS 连接性分析 (边级别置换检验)
3. `frequency_stats.ipynb` — 频域分析

## 前置依赖

需先完成以下分析，生成对应结果文件：

1. **频域分析** (`analysis/频域分析/frequency_analysis_group.ipynb`) — 三组分别运行，生成 `subject_band_powers.csv`
2. **连接性分析** (`analysis/connectivity_analysis/rest_connectivity.ipynb`) — 三组分别运行，生成 `small_world_metrics.csv` 和 `connectivity_results.npz`

## 下一阶段计划

### 混合效应模型分析 (年龄协变量控制)

当前分析采用被试水平的方差分析/非参数检验，未考虑年龄等协变量的影响。下一阶段将引入**线性混合效应模型 (Linear Mixed-Effects Models, LMM)** 进行更精细的统计建模：

#### 分析目标

1. **控制年龄协变量**: 在组间比较中控制年龄的线性/非线性效应
2. **提升统计效力**: 利用 LMM 对个体差异的建模能力，减少残差方差
3. **灵活建模**: 支持固定效应 (组别、年龄) 和随机效应 (被试个体差异)

#### 模型设计

**基础模型**:

```r
Y ~ Group + Age + (1|Subject)
```

- `Y`: 因变量 (图论指标、ROI 连接性、频域功率)
- `Group`: 固定效应 (adhd/com/td)
- `Age`: 协变量 (连续变量)
- `(1|Subject)`: 随机截距 (被试个体差异)

**扩展模型** (如需要):

```r
Y ~ Group * Age + (1|Subject)  # 组别与年龄交互作用
Y ~ Group + Age + Age² + (1|Subject)  # 年龄非线性效应
```

#### 实施步骤

1. **数据准备**: 合并三组数据，添加年龄变量 (需从被试信息表获取)
2. **模型拟合**: 使用 `statsmodels.MixedLM` 或 `R lme4::lmer`
3. **模型比较**: AIC/BIC 选择最优模型 (是否需要交互项/非线性项)
4. **假设检验**:
   - 组间差异: Wald χ² 检验 (控制年龄后)
   - 事后比较: 估计边际均值 (Estimated Marginal Means, EMM) + 对比
5. **效应量**: 报告标准化回归系数 (β) 和偏 η²
6. **诊断**: 残差正态性、同方差性、多重共线性检查

#### 预期输出

- `lmm_results.csv` — 模型参数估计、显著性检验
- `lmm_posthoc.csv` — 控制年龄后的组间两两比较
- `lmm_diagnostics/` — 残差诊断图、模型拟合图
- `lmm_report.md` — 完整分析报告

#### 工具选择

- **Python**: `statsmodels.MixedLM` (推荐，与现有流程集成)
- **R**: `lme4::lmer` + `emmeans` (更成熟的 LMM 生态)

#### 注意事项

- 年龄数据需从被试信息表提取并合并到分析数据中
- 检查年龄分布是否在三组间平衡 (如不平衡，LMM 的协变量控制尤为重要)
- 对于 NBS 边级别分析，LMM 可能计算量较大，需评估可行性
