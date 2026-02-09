# %% [markdown]
# # 组间统计分析 - com vs std
#
# 对共患组 (com) 和正常组 (std) 进行 EEG 指标的组间统计比较分析。
#
# **分析内容:**
# - 连接性指标: 聚类系数、路径长度、小世界系数、平均连接强度
# - 频域指标: 各频段功率 (Delta, Theta, Alpha, Beta, Gamma)
# - 统计检验: 自动选择 (正态性检验后决定 t检验/Mann-Whitney U)
# - 多重比较校正: FDR (Benjamini-Hochberg)

# %% [markdown]
# ## 1. 导入库和配置

# %%
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

import numpy as np
import pandas as pd
import pickle
from scipy import stats
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

print("库导入完成")

# %%
# =============================================================================
# 路径配置
# =============================================================================
BASE_PATH = Path(r'd:\LYW\REST_COM')

# 连接性分析结果路径
CONN_COM_PATH = BASE_PATH / 'analysis' / 'connectivity_analysis' / 'com' / 'results'
CONN_STD_PATH = BASE_PATH / 'analysis' / 'connectivity_analysis' / 'std' / 'results'

# 频域分析结果路径
FREQ_COM_PATH = BASE_PATH / 'analysis' / '频域分析' / 'com' / 'results' / 'group'
FREQ_STD_PATH = BASE_PATH / 'analysis' / '频域分析' / 'std' / 'results'

# 输出路径
OUTPUT_PATH = BASE_PATH / 'reports' / 'comparison'
FIGURES_PATH = OUTPUT_PATH / 'figures'

# 创建输出目录
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
FIGURES_PATH.mkdir(parents=True, exist_ok=True)

print(f"输出目录: {OUTPUT_PATH}")

# %% [markdown]
# ## 2. 数据加载与整合

# %%
def load_connectivity_data() -> pd.DataFrame:
    """加载并整合两组连接性分析数据"""

    # 加载 com 组
    com_df = pd.read_csv(CONN_COM_PATH / 'small_world_metrics.csv')
    com_df['group'] = 'com'
    # 添加前缀到 subject_id (先转为字符串)
    com_df['subject_id'] = com_df['subject_id'].astype(str)
    if not com_df['subject_id'].str.startswith('com_').any():
        com_df['subject_id'] = 'com_' + com_df['subject_id']

    # 加载 std 组
    std_df = pd.read_csv(CONN_STD_PATH / 'small_world_metrics.csv')
    if 'group' not in std_df.columns:
        std_df['group'] = 'std'

    # 合并
    combined_df = pd.concat([com_df, std_df], ignore_index=True)

    print(f"连接性数据加载完成:")
    print(f"  com 组: {com_df['subject_id'].nunique()} 被试, {len(com_df)} 条记录")
    print(f"  std 组: {std_df['subject_id'].nunique()} 被试, {len(std_df)} 条记录")

    return combined_df

# 加载连接性数据
conn_df = load_connectivity_data()
print(f"\n数据预览:")
print(conn_df.head())

# %%
def load_frequency_data() -> pd.DataFrame:
    """加载并整合两组频域分析数据"""

    freq_data = []

    # 加载 com 组
    com_pkl_path = FREQ_COM_PATH / 'group_frequency_results.pkl'
    if com_pkl_path.exists():
        with open(com_pkl_path, 'rb') as f:
            com_results = pickle.load(f)

        # 提取频段功率数据
        if 'band_powers_group' in com_results:
            band_powers = com_results['band_powers_group']
            if isinstance(band_powers, dict) and 'mean' in band_powers:
                band_powers = band_powers['mean']

            # 获取被试信息
            n_subjects = com_results.get('n_subjects', 12)

            for band_name, powers in band_powers.items():
                # powers 是通道平均功率，取全脑平均
                mean_power = np.mean(powers) if isinstance(powers, np.ndarray) else powers
                freq_data.append({
                    'group': 'com',
                    'freq_band': band_name,
                    'mean_power': mean_power
                })
        print(f"  com 组频域数据加载成功")
    else:
        print(f"  com 组频域数据文件不存在: {com_pkl_path}")

    # 加载 std 组
    std_pkl_path = FREQ_STD_PATH / 'group_frequency_results.pkl'
    if std_pkl_path.exists():
        with open(std_pkl_path, 'rb') as f:
            std_results = pickle.load(f)

        if 'band_powers_group' in std_results:
            band_powers = std_results['band_powers_group']
            if isinstance(band_powers, dict) and 'mean' in band_powers:
                band_powers = band_powers['mean']

            for band_name, powers in band_powers.items():
                mean_power = np.mean(powers) if isinstance(powers, np.ndarray) else powers
                freq_data.append({
                    'group': 'std',
                    'freq_band': band_name,
                    'mean_power': mean_power
                })
        print(f"  std 组频域数据加载成功")
    else:
        print(f"  std 组频域数据文件不存在: {std_pkl_path}")

    if freq_data:
        return pd.DataFrame(freq_data)
    else:
        return pd.DataFrame()

# 加载频域数据
print("\n频域数据加载:")
freq_df = load_frequency_data()
if not freq_df.empty:
    print(f"\n频域数据预览:")
    print(freq_df)

# %% [markdown]
# ## 3. 描述性统计

# %%
def compute_descriptive_stats(df: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    """计算描述性统计"""

    stats_list = []

    for metric in metrics:
        for band in df['freq_band'].unique():
            band_data = df[df['freq_band'] == band]

            for group in ['com', 'std']:
                group_data = band_data[band_data['group'] == group][metric]

                stats_list.append({
                    'metric': metric,
                    'freq_band': band,
                    'group': group,
                    'n': len(group_data),
                    'mean': group_data.mean(),
                    'std': group_data.std(),
                    'median': group_data.median(),
                    'min': group_data.min(),
                    'max': group_data.max()
                })

    return pd.DataFrame(stats_list)

# 连接性指标描述性统计
conn_metrics = ['clustering', 'path_length', 'sigma', 'avg_connectivity']
conn_desc_stats = compute_descriptive_stats(conn_df, conn_metrics)

print("连接性指标描述性统计:")
print(conn_desc_stats.pivot_table(
    index=['metric', 'freq_band'],
    columns='group',
    values=['mean', 'std']
).round(4))

# %% [markdown]
# ## 4. 统计检验

# %%
def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """计算 Cohen's d 效应量"""
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(), group2.var()

    # 池化标准差
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    return (group1.mean() - group2.mean()) / pooled_std


def run_group_comparison(
    df: pd.DataFrame,
    metric: str,
    freq_band: str
) -> Dict:
    """对单个指标进行组间比较"""

    band_data = df[df['freq_band'] == freq_band]
    com_data = band_data[band_data['group'] == 'com'][metric].values
    std_data = band_data[band_data['group'] == 'std'][metric].values

    # 正态性检验
    _, p_norm_com = stats.shapiro(com_data) if len(com_data) >= 3 else (0, 0)
    _, p_norm_std = stats.shapiro(std_data) if len(std_data) >= 3 else (0, 0)

    is_normal = (p_norm_com > 0.05) and (p_norm_std > 0.05)

    # 选择检验方法
    if is_normal:
        test_name = 't-test'
        statistic, p_value = stats.ttest_ind(com_data, std_data)
    else:
        test_name = 'Mann-Whitney U'
        statistic, p_value = stats.mannwhitneyu(com_data, std_data, alternative='two-sided')

    # 效应量
    d = cohens_d(com_data, std_data)

    return {
        'metric': metric,
        'freq_band': freq_band,
        'com_mean': com_data.mean(),
        'com_std': com_data.std(),
        'std_mean': std_data.mean(),
        'std_std': std_data.std(),
        'test': test_name,
        'statistic': statistic,
        'p_value': p_value,
        'cohens_d': d,
        'is_normal': is_normal
    }


def run_all_comparisons(df: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    """运行所有组间比较"""

    results = []
    freq_bands = df['freq_band'].unique()

    for metric in metrics:
        for band in freq_bands:
            result = run_group_comparison(df, metric, band)
            results.append(result)

    results_df = pd.DataFrame(results)

    # FDR 校正
    _, p_corrected, _, _ = multipletests(results_df['p_value'], method='fdr_bh')
    results_df['p_fdr'] = p_corrected
    results_df['significant'] = results_df['p_fdr'] < 0.05

    return results_df

# 运行连接性指标组间比较
print("运行连接性指标组间比较...")
conn_stats = run_all_comparisons(conn_df, conn_metrics)

print("\n连接性指标统计检验结果:")
print(conn_stats[['metric', 'freq_band', 'com_mean', 'std_mean', 'test', 'p_value', 'p_fdr', 'cohens_d', 'significant']].round(4))

# 保存结果
conn_stats.to_csv(OUTPUT_PATH / 'connectivity_stats.csv', index=False)
print(f"\n✓ 保存: {OUTPUT_PATH / 'connectivity_stats.csv'}")

# %% [markdown]
# ## 5. 可视化

# %%
def plot_connectivity_boxplot(df: pd.DataFrame, stats_df: pd.DataFrame):
    """绘制连接性指标组间箱线图"""

    metrics = ['clustering', 'path_length', 'sigma', 'avg_connectivity']
    metric_labels = {
        'clustering': 'Clustering Coefficient',
        'path_length': 'Path Length',
        'sigma': 'Small-world σ',
        'avg_connectivity': 'Avg Connectivity (PLI)'
    }

    # 计算每组被试数
    n_com = df[df['group'] == 'com']['subject_id'].nunique()
    n_std = df[df['group'] == 'std']['subject_id'].nunique()

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    freq_order = ['delta', 'theta', 'alpha', 'beta', 'gamma']

    for ax, metric in zip(axes, metrics):
        # 绘制箱线图
        sns.boxplot(
            data=df, x='freq_band', y=metric, hue='group',
            order=freq_order, palette={'com': '#E74C3C', 'std': '#3498DB'},
            ax=ax
        )

        ax.set_title(metric_labels[metric], fontsize=12, fontweight='bold')
        ax.set_xlabel('Frequency Band')
        ax.set_ylabel(metric_labels[metric])
        ax.legend(title='Group', loc='upper right',
                  labels=[f'com (n={n_com})', f'std (n={n_std})'])

        # 添加显著性标记
        for i, band in enumerate(freq_order):
            row = stats_df[(stats_df['metric'] == metric) & (stats_df['freq_band'] == band)]
            if not row.empty and row['significant'].values[0]:
                # 获取 y 位置
                y_max = df[df['freq_band'] == band][metric].max()
                ax.annotate('*', xy=(i, y_max * 1.05), ha='center', fontsize=14, color='red')

    # 添加总标题，包含样本信息
    fig.suptitle(f'Group Comparison: com (n={n_com}) vs std (n={n_std})\n* = FDR < 0.05',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    fig.savefig(FIGURES_PATH / 'connectivity_boxplot.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✓ 保存: {FIGURES_PATH / 'connectivity_boxplot.png'}")

# 绘制连接性箱线图
plot_connectivity_boxplot(conn_df, conn_stats)

# %%
def plot_effect_size(stats_df: pd.DataFrame, n_com: int, n_std: int):
    """绘制效应量森林图"""

    fig, ax = plt.subplots(figsize=(12, 10))

    # 准备数据
    stats_df = stats_df.copy()
    stats_df['label'] = stats_df['freq_band'] + ' - ' + stats_df['metric']
    stats_df = stats_df.sort_values(['metric', 'freq_band'])

    y_pos = range(len(stats_df))
    colors = ['#E74C3C' if sig else '#95A5A6' for sig in stats_df['significant']]

    # 绘制效应量
    ax.barh(y_pos, stats_df['cohens_d'], color=colors, alpha=0.7)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax.axvline(x=0.2, color='gray', linestyle='--', linewidth=0.5, label='Small (0.2)')
    ax.axvline(x=-0.2, color='gray', linestyle='--', linewidth=0.5)
    ax.axvline(x=0.5, color='gray', linestyle=':', linewidth=0.5, label='Medium (0.5)')
    ax.axvline(x=-0.5, color='gray', linestyle=':', linewidth=0.5)
    ax.axvline(x=0.8, color='gray', linestyle='-.', linewidth=0.5, label='Large (0.8)')
    ax.axvline(x=-0.8, color='gray', linestyle='-.', linewidth=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(stats_df['label'])
    ax.set_xlabel("Cohen's d (com - std)")
    ax.set_title(f"Effect Size Forest Plot\ncom (n={n_com}) vs std (n={n_std})\n(Red = FDR < 0.05, Gray = Not significant)",
                 fontsize=12, fontweight='bold')
    ax.legend(loc='lower right')

    plt.tight_layout()
    fig.savefig(FIGURES_PATH / 'effect_size_forest.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✓ 保存: {FIGURES_PATH / 'effect_size_forest.png'}")

# 计算被试数用于图表
n_com = conn_df[conn_df['group'] == 'com']['subject_id'].nunique()
n_std = conn_df[conn_df['group'] == 'std']['subject_id'].nunique()

# 绘制效应量图
plot_effect_size(conn_stats, n_com, n_std)

# %%
def plot_significance_heatmap(stats_df: pd.DataFrame, n_com: int, n_std: int):
    """绘制显著性热图"""

    # 创建 p 值矩阵
    metrics = ['clustering', 'path_length', 'sigma', 'avg_connectivity']
    freq_bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']

    p_matrix = np.zeros((len(metrics), len(freq_bands)))
    d_matrix = np.zeros((len(metrics), len(freq_bands)))

    for i, metric in enumerate(metrics):
        for j, band in enumerate(freq_bands):
            row = stats_df[(stats_df['metric'] == metric) & (stats_df['freq_band'] == band)]
            if not row.empty:
                p_matrix[i, j] = -np.log10(row['p_fdr'].values[0] + 1e-10)
                d_matrix[i, j] = row['cohens_d'].values[0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # P 值热图
    sns.heatmap(
        p_matrix, ax=axes[0],
        xticklabels=freq_bands, yticklabels=metrics,
        cmap='Reds', annot=True, fmt='.2f',
        cbar_kws={'label': '-log10(p_FDR)'}
    )
    axes[0].set_title(f'Significance Heatmap\ncom (n={n_com}) vs std (n={n_std})\n(-log10 p-value, higher = more significant)')
    axes[0].axhline(y=0, color='black', linewidth=0.5)

    # 效应量热图
    sns.heatmap(
        d_matrix, ax=axes[1],
        xticklabels=freq_bands, yticklabels=metrics,
        cmap='RdBu_r', center=0, annot=True, fmt='.2f',
        cbar_kws={'label': "Cohen's d"}
    )
    axes[1].set_title(f"Effect Size Heatmap\ncom (n={n_com}) vs std (n={n_std})\n(Cohen's d: com - std)")

    plt.tight_layout()
    fig.savefig(FIGURES_PATH / 'significance_heatmap.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✓ 保存: {FIGURES_PATH / 'significance_heatmap.png'}")

# 绘制显著性热图
plot_significance_heatmap(conn_stats, n_com, n_std)

# %% [markdown]
# ## 6. 生成报告

# %%
def generate_report(conn_stats: pd.DataFrame, conn_desc: pd.DataFrame, n_com: int, n_std: int) -> str:
    """生成 Markdown 报告"""

    date_str = datetime.now().strftime('%Y-%m-%d %H:%M')

    # 统计显著结果
    sig_results = conn_stats[conn_stats['significant']]
    n_sig = len(sig_results)
    n_total = len(conn_stats)

    report = f"""# 组间统计分析报告 - com vs std

**生成时间**: {date_str}

---

## 1. 分析概况

| 项目 | 值 |
|------|-----|
| 共患组 (com) 被试数 | {n_com} |
| 正常组 (std) 被试数 | {n_std} |
| 总样本量 | {n_com + n_std} |
| 分析指标数 | 4 (连接性) |
| 频段数 | 5 |
| 总比较数 | {n_total} |
| 显著结果数 (FDR < 0.05) | {n_sig} |

---

## 2. 统计方法

- **正态性检验**: Shapiro-Wilk 检验
- **组间比较**:
  - 正态分布 → 独立样本 t 检验
  - 非正态分布 → Mann-Whitney U 检验
- **多重比较校正**: FDR (Benjamini-Hochberg)
- **效应量**: Cohen's d

---

## 3. 连接性指标结果

### 3.1 描述性统计

| 指标 | 频段 | com (n={n_com}, M±SD) | std (n={n_std}, M±SD) |
|------|------|------------|------------|
"""

    for _, row in conn_stats.iterrows():
        com_str = f"{row['com_mean']:.4f}±{row['com_std']:.4f}"
        std_str = f"{row['std_mean']:.4f}±{row['std_std']:.4f}"
        report += f"| {row['metric']} | {row['freq_band']} | {com_str} | {std_str} |\n"

    report += f"""
### 3.2 统计检验结果

| 指标 | 频段 | 检验方法 | p值 | p_FDR | Cohen's d | 显著 |
|------|------|----------|-----|-------|-----------|------|
"""

    for _, row in conn_stats.iterrows():
        sig_mark = "✓" if row['significant'] else ""
        report += f"| {row['metric']} | {row['freq_band']} | {row['test']} | {row['p_value']:.4f} | {row['p_fdr']:.4f} | {row['cohens_d']:.3f} | {sig_mark} |\n"

    report += f"""
### 3.3 显著性结果汇总

"""

    if n_sig > 0:
        report += "以下指标在组间存在显著差异 (FDR < 0.05):\n\n"
        for _, row in sig_results.iterrows():
            direction = "com > std" if row['cohens_d'] > 0 else "com < std"
            report += f"- **{row['metric']} ({row['freq_band']})**: {direction}, d = {row['cohens_d']:.3f}\n"
    else:
        report += "未发现显著的组间差异 (FDR < 0.05)\n"

    report += f"""
---

## 4. 可视化

### 4.1 连接性指标箱线图
![Connectivity Boxplot](figures/connectivity_boxplot.png)

### 4.2 效应量森林图
![Effect Size](figures/effect_size_forest.png)

### 4.3 显著性热图
![Significance Heatmap](figures/significance_heatmap.png)

---

## 5. 效应量解读

| Cohen's d | 解读 |
|-----------|------|
| |d| < 0.2 | 微小效应 |
| 0.2 ≤ |d| < 0.5 | 小效应 |
| 0.5 ≤ |d| < 0.8 | 中等效应 |
| |d| ≥ 0.8 | 大效应 |

---

## 6. 结论

{f"本分析发现 {n_sig} 个指标在共患组和正常组之间存在显著差异。" if n_sig > 0 else "本分析未发现共患组和正常组之间存在显著差异。"}

---

*报告由 REST EEG Pipeline 自动生成*
"""

    return report

# 生成报告
report = generate_report(conn_stats, conn_desc_stats, n_com, n_std)

# 保存报告
report_file = OUTPUT_PATH / 'group_comparison_report.md'
with open(report_file, 'w', encoding='utf-8') as f:
    f.write(report)

print(f"✓ 报告已保存: {report_file}")

# %% [markdown]
# ## 7. 分析完成

# %%
print("\n" + "=" * 60)
print("组间统计分析完成!")
print("=" * 60)

print(f"\n输出文件:")
for f in OUTPUT_PATH.glob('*'):
    if f.is_file():
        print(f"  - {f.name}")

print(f"\n可视化文件:")
for f in FIGURES_PATH.glob('*.png'):
    print(f"  - {f.name}")

# 显示显著结果摘要
sig_results = conn_stats[conn_stats['significant']]
print(f"\n显著结果 (FDR < 0.05): {len(sig_results)}/{len(conn_stats)}")
if len(sig_results) > 0:
    print(sig_results[['metric', 'freq_band', 'p_fdr', 'cohens_d']].to_string(index=False))
