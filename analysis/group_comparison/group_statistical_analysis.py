# %% [markdown]
# # 组间统计分析 - ADHD vs COM vs TD (三组比较)
#
# 对 ADHD 单纯组 (adhd)、ADHD 共患阅读困难组 (com)、正常发展组 (td) 进行 EEG 指标的组间统计比较。
#
# **统计思路:** 被试水平分析 (每位被试贡献一个数据点)
#
# **分析内容:**
# - 连接性指标: 聚类系数、路径长度、小世界系数、平均连接强度 (按频段)
# - 频域指标: 各频段功率 (Delta, Theta, Alpha, Beta, Gamma)
# - Omnibus 检验: ANOVA / Welch's ANOVA / Kruskal-Wallis
# - 事后检验: Tukey HSD / Games-Howell / Dunn
# - 多重比较校正: FDR (Benjamini-Hochberg)

# %% [markdown]
# ## 1. 导入库和配置

# %%
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from itertools import combinations
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
import scikit_posthocs as sp
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

warnings.filterwarnings('ignore')

# 中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

print("库导入完成")

# %%
# =============================================================================
# 配置
# =============================================================================
BASE_PATH = Path(r'd:\LYW\REST_COM')

GROUPS = ['adhd', 'com', 'td']
GROUP_LABELS = {
    'adhd': 'ADHD单纯组',
    'com': 'ADHD共患阅读困难组',
    'td': '正常发展组',
}
GROUP_COLORS = {
    'adhd': '#E74C3C',
    'com': '#F39C12',
    'td': '#3498DB',
}
GROUP_PAIRS = [('adhd', 'com'), ('adhd', 'td'), ('com', 'td')]

FREQ_BANDS = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']

# 数据路径
CONN_PATHS = {g: BASE_PATH / 'analysis' / 'connectivity_analysis' / g / 'results' for g in GROUPS}
FREQ_PATHS = {g: BASE_PATH / 'analysis' / '频域分析' / g / 'results' for g in GROUPS}

# 输出路径
OUTPUT_PATH = BASE_PATH / 'reports' / 'comparison'
FIGURES_PATH = OUTPUT_PATH / 'figures'
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
FIGURES_PATH.mkdir(parents=True, exist_ok=True)

print(f"输出目录: {OUTPUT_PATH}")

# %% [markdown]
# ## 2. 数据加载

# %%
def load_connectivity_data() -> pd.DataFrame:
    """加载三组连接性数据"""
    dfs = []
    for g in GROUPS:
        path = CONN_PATHS[g] / 'small_world_metrics.csv'
        if not path.exists():
            print(f"⚠ 缺少 {g} 组连接性数据: {path}")
            continue
        df = pd.read_csv(path)
        df['group'] = g
        df['subject_id'] = df['subject_id'].astype(str)
        dfs.append(df)
        print(f"  {g}: {df['subject_id'].nunique()} 被试, {len(df)} 行")

    conn_df = pd.concat(dfs, ignore_index=True)
    print(f"\n连接性数据合并: {conn_df['subject_id'].nunique()} 被试, {len(conn_df)} 行")
    return conn_df


def load_frequency_data() -> pd.DataFrame:
    """加载三组被试级频段功率数据"""
    dfs = []
    for g in GROUPS:
        path = FREQ_PATHS[g] / 'subject_band_powers.csv'
        if not path.exists():
            print(f"⚠ 缺少 {g} 组频域数据: {path}")
            print(f"  请先运行 frequency_analysis_group.ipynb (GROUP_NAME='{g}')")
            continue
        df = pd.read_csv(path)
        df['group'] = g
        dfs.append(df)
        print(f"  {g}: {len(df)} 被试")

    if not dfs:
        print("❌ 无频域数据可加载")
        return pd.DataFrame()

    freq_df = pd.concat(dfs, ignore_index=True)
    print(f"\n频域数据合并: {len(freq_df)} 被试")
    return freq_df


print("加载连接性数据:")
conn_df = load_connectivity_data()

print("\n加载频域数据:")
freq_df = load_frequency_data()

# %% [markdown]
# ## 3. 描述性统计

# %%
def compute_descriptive_stats(df: pd.DataFrame, metrics: List[str], group_col: str = 'group') -> pd.DataFrame:
    """按组计算描述性统计"""
    rows = []
    for metric in metrics:
        for g in GROUPS:
            data = df[df[group_col] == g][metric].dropna()
            rows.append({
                'metric': metric,
                'group': g,
                'n': len(data),
                'mean': data.mean(),
                'std': data.std(),
                'median': data.median(),
                'q25': data.quantile(0.25),
                'q75': data.quantile(0.75),
            })
    return pd.DataFrame(rows)


# 连接性指标描述性统计 (按频段)
conn_metrics = ['clustering', 'path_length', 'sigma', 'avg_connectivity']
conn_desc_list = []
for band in conn_df['freq_band'].unique():
    band_df = conn_df[conn_df['freq_band'] == band]
    desc = compute_descriptive_stats(band_df, conn_metrics)
    desc['freq_band'] = band
    conn_desc_list.append(desc)
conn_desc_stats = pd.concat(conn_desc_list, ignore_index=True)

print("连接性指标描述性统计:")
display(conn_desc_stats.head(15))

# 频域指标描述性统计
if not freq_df.empty:
    freq_desc_stats = compute_descriptive_stats(freq_df, FREQ_BANDS)
    print("\n频域指标描述性统计:")
    display(freq_desc_stats)

# %% [markdown]
# ## 4. 统计检验

# %%
def check_normality(groups_data: List[np.ndarray], alpha: float = 0.05) -> Tuple[bool, List[float]]:
    """Shapiro-Wilk 正态性检验 (每组)"""
    p_values = []
    for data in groups_data:
        if len(data) < 3:
            p_values.append(0.0)  # 样本太小，视为非正态
        else:
            _, p = stats.shapiro(data)
            p_values.append(p)
    all_normal = all(p > alpha for p in p_values)
    return all_normal, p_values


def check_homogeneity(groups_data: List[np.ndarray], alpha: float = 0.05) -> Tuple[bool, float]:
    """Levene 方差齐性检验"""
    if any(len(d) < 2 for d in groups_data):
        return False, 0.0
    _, p = stats.levene(*groups_data)
    return p > alpha, p


def compute_eta_squared(groups_data: List[np.ndarray]) -> float:
    """计算 η² (ANOVA 效应量)"""
    all_data = np.concatenate(groups_data)
    grand_mean = all_data.mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups_data)
    ss_total = np.sum((all_data - grand_mean) ** 2)
    return ss_between / ss_total if ss_total > 0 else 0.0


def compute_epsilon_squared(H: float, n: int, k: int) -> float:
    """计算 ε² (Kruskal-Wallis 效应量)"""
    return (H - k + 1) / (n - k) if (n - k) > 0 else 0.0


def compute_cohens_d(g1: np.ndarray, g2: np.ndarray) -> float:
    """计算 Cohen's d"""
    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        return 0.0
    pooled_std = np.sqrt(((n1 - 1) * g1.std(ddof=1)**2 + (n2 - 1) * g2.std(ddof=1)**2) / (n1 + n2 - 2))
    return (g1.mean() - g2.mean()) / pooled_std if pooled_std > 0 else 0.0


def run_omnibus_test(groups_data: List[np.ndarray]) -> Dict:
    """运行 omnibus 检验 (自动选择参数/非参数)"""
    is_normal, norm_ps = check_normality(groups_data)
    is_homogeneous, levene_p = check_homogeneity(groups_data)
    n_total = sum(len(g) for g in groups_data)
    k = len(groups_data)

    result = {
        'is_normal': is_normal,
        'normality_ps': norm_ps,
        'is_homogeneous': is_homogeneous,
        'levene_p': levene_p,
    }

    if is_normal:
        if is_homogeneous:
            # 标准 ANOVA
            stat, p = stats.f_oneway(*groups_data)
            result.update({
                'test': 'ANOVA',
                'statistic': stat,
                'p_value': p,
                'effect_size': compute_eta_squared(groups_data),
                'effect_size_name': 'η²',
            })
        else:
            # Welch's ANOVA (Alexander-Govern)
            try:
                res = stats.alexandergovern(*groups_data)
                stat, p = res.statistic, res.pvalue
            except Exception:
                stat, p = stats.f_oneway(*groups_data)
            result.update({
                'test': "Welch's ANOVA",
                'statistic': stat,
                'p_value': p,
                'effect_size': compute_eta_squared(groups_data),
                'effect_size_name': 'η²',
            })
    else:
        # Kruskal-Wallis
        stat, p = stats.kruskal(*groups_data)
        result.update({
            'test': 'Kruskal-Wallis',
            'statistic': stat,
            'p_value': p,
            'effect_size': compute_epsilon_squared(stat, n_total, k),
            'effect_size_name': 'ε²',
        })

    return result


def run_posthoc_tests(groups_data: List[np.ndarray], group_names: List[str],
                      is_normal: bool, is_homogeneous: bool) -> pd.DataFrame:
    """运行事后两两比较"""
    rows = []
    pairs = list(combinations(range(len(group_names)), 2))

    if is_normal:
        # 构建 DataFrame 用于 posthoc
        all_vals = np.concatenate(groups_data)
        all_groups = np.concatenate([[group_names[i]] * len(groups_data[i]) for i in range(len(groups_data))])
        posthoc_df = pd.DataFrame({'value': all_vals, 'group': all_groups})

        if is_homogeneous:
            # Tukey HSD
            try:
                from statsmodels.stats.multicomp import pairwise_tukeyhsd
                tukey = pairwise_tukeyhsd(posthoc_df['value'], posthoc_df['group'], alpha=0.05)
                for i, (g1_idx, g2_idx) in enumerate(pairs):
                    g1, g2 = group_names[g1_idx], group_names[g2_idx]
                    # 从 tukey 结果中找到对应的行
                    for row_idx in range(len(tukey.summary().data) - 1):
                        row_data = tukey.summary().data[row_idx + 1]
                        if (str(row_data[0]) == g1 and str(row_data[1]) == g2) or \
                           (str(row_data[0]) == g2 and str(row_data[1]) == g1):
                            rows.append({
                                'group1': g1, 'group2': g2,
                                'test': 'Tukey HSD',
                                'p_value': float(row_data[3]),
                                'cohens_d': compute_cohens_d(groups_data[g1_idx], groups_data[g2_idx]),
                            })
                            break
                    else:
                        # fallback
                        _, p = stats.ttest_ind(groups_data[g1_idx], groups_data[g2_idx])
                        rows.append({
                            'group1': g1, 'group2': g2,
                            'test': 't-test (Bonferroni)',
                            'p_value': min(p * len(pairs), 1.0),
                            'cohens_d': compute_cohens_d(groups_data[g1_idx], groups_data[g2_idx]),
                        })
            except Exception:
                # fallback: Bonferroni-corrected t-tests
                for g1_idx, g2_idx in pairs:
                    _, p = stats.ttest_ind(groups_data[g1_idx], groups_data[g2_idx])
                    rows.append({
                        'group1': group_names[g1_idx], 'group2': group_names[g2_idx],
                        'test': 't-test (Bonferroni)',
                        'p_value': min(p * len(pairs), 1.0),
                        'cohens_d': compute_cohens_d(groups_data[g1_idx], groups_data[g2_idx]),
                    })
        else:
            # Games-Howell (via scikit_posthocs)
            try:
                ph = sp.posthoc_gameshowell(posthoc_df, val_col='value', group_col='group')
                for g1_idx, g2_idx in pairs:
                    g1, g2 = group_names[g1_idx], group_names[g2_idx]
                    rows.append({
                        'group1': g1, 'group2': g2,
                        'test': 'Games-Howell',
                        'p_value': ph.loc[g1, g2],
                        'cohens_d': compute_cohens_d(groups_data[g1_idx], groups_data[g2_idx]),
                    })
            except Exception:
                for g1_idx, g2_idx in pairs:
                    _, p = stats.ttest_ind(groups_data[g1_idx], groups_data[g2_idx], equal_var=False)
                    rows.append({
                        'group1': group_names[g1_idx], 'group2': group_names[g2_idx],
                        'test': "Welch t-test (Bonferroni)",
                        'p_value': min(p * len(pairs), 1.0),
                        'cohens_d': compute_cohens_d(groups_data[g1_idx], groups_data[g2_idx]),
                    })
    else:
        # Dunn's test
        all_vals = np.concatenate(groups_data)
        all_groups = np.concatenate([[group_names[i]] * len(groups_data[i]) for i in range(len(groups_data))])
        posthoc_df = pd.DataFrame({'value': all_vals, 'group': all_groups})

        try:
            ph = sp.posthoc_dunn(posthoc_df, val_col='value', group_col='group', p_adjust='bonferroni')
            for g1_idx, g2_idx in pairs:
                g1, g2 = group_names[g1_idx], group_names[g2_idx]
                rows.append({
                    'group1': g1, 'group2': g2,
                    'test': 'Dunn (Bonferroni)',
                    'p_value': ph.loc[g1, g2],
                    'cohens_d': compute_cohens_d(groups_data[g1_idx], groups_data[g2_idx]),
                })
        except Exception:
            for g1_idx, g2_idx in pairs:
                _, p = stats.mannwhitneyu(groups_data[g1_idx], groups_data[g2_idx], alternative='two-sided')
                rows.append({
                    'group1': group_names[g1_idx], 'group2': group_names[g2_idx],
                    'test': 'Mann-Whitney U (Bonferroni)',
                    'p_value': min(p * len(pairs), 1.0),
                    'cohens_d': compute_cohens_d(groups_data[g1_idx], groups_data[g2_idx]),
                })

    return pd.DataFrame(rows)


print("统计函数定义完成")

# %% [markdown]
# ## 4.2 执行统计检验

# %%
# --- 连接性指标检验 ---
print("=" * 60)
print("连接性指标 Omnibus 检验")
print("=" * 60)

conn_omnibus_results = []
conn_posthoc_results = []

for band in conn_df['freq_band'].unique():
    band_data = conn_df[conn_df['freq_band'] == band]
    for metric in conn_metrics:
        result = run_omnibus_test(band_data, metric)
        result['freq_band'] = band
        conn_omnibus_results.append(result)

conn_omnibus_df = pd.DataFrame(conn_omnibus_results)

# FDR 校正
if len(conn_omnibus_df) > 0:
    _, conn_omnibus_df['p_fdr'], _, _ = multipletests(
        conn_omnibus_df['p_value'], method='fdr_bh'
    )
    conn_omnibus_df['significant'] = conn_omnibus_df['p_fdr'] < 0.05

print(f"\n连接性 Omnibus 结果: {conn_omnibus_df['significant'].sum()}/{len(conn_omnibus_df)} 显著")
display(conn_omnibus_df[['metric', 'freq_band', 'test', 'statistic', 'p_value', 'p_fdr', 'effect_size', 'significant']])

# 事后检验 (仅对 omnibus 显著的指标)
sig_conn = conn_omnibus_df[conn_omnibus_df['significant']]
for _, row in sig_conn.iterrows():
    band_data = conn_df[conn_df['freq_band'] == row['freq_band']]
    posthoc = run_posthoc_tests(band_data, row['metric'], row['is_parametric'], row['equal_var'])
    posthoc['metric'] = row['metric']
    posthoc['freq_band'] = row['freq_band']
    conn_posthoc_results.append(posthoc)

if conn_posthoc_results:
    conn_posthoc_df = pd.concat(conn_posthoc_results, ignore_index=True)
    print(f"\n连接性事后检验 ({len(conn_posthoc_df)} 对比较):")
    display(conn_posthoc_df)
else:
    conn_posthoc_df = pd.DataFrame()
    print("\n无显著 omnibus 结果，跳过事后检验")

# %%
# --- 频域指标检验 ---
print("=" * 60)
print("频域指标 Omnibus 检验")
print("=" * 60)

freq_omnibus_results = []
freq_posthoc_results = []

if not freq_df.empty:
    for band in FREQ_BANDS:
        result = run_omnibus_test(freq_df, band)
        result['freq_band'] = band
        freq_omnibus_results.append(result)

    freq_omnibus_df = pd.DataFrame(freq_omnibus_results)

    if len(freq_omnibus_df) > 0:
        _, freq_omnibus_df['p_fdr'], _, _ = multipletests(
            freq_omnibus_df['p_value'], method='fdr_bh'
        )
        freq_omnibus_df['significant'] = freq_omnibus_df['p_fdr'] < 0.05

    print(f"\n频域 Omnibus 结果: {freq_omnibus_df['significant'].sum()}/{len(freq_omnibus_df)} 显著")
    display(freq_omnibus_df[['metric', 'freq_band', 'test', 'statistic', 'p_value', 'p_fdr', 'effect_size', 'significant']])

    # 事后检验
    sig_freq = freq_omnibus_df[freq_omnibus_df['significant']]
    for _, row in sig_freq.iterrows():
        posthoc = run_posthoc_tests(freq_df, row['metric'], row['is_parametric'], row['equal_var'])
        posthoc['metric'] = row['metric']
        posthoc['freq_band'] = row['freq_band']
        freq_posthoc_results.append(posthoc)

    if freq_posthoc_results:
        freq_posthoc_df = pd.concat(freq_posthoc_results, ignore_index=True)
        print(f"\n频域事后检验 ({len(freq_posthoc_df)} 对比较):")
        display(freq_posthoc_df)
    else:
        freq_posthoc_df = pd.DataFrame()
        print("\n无显著 omnibus 结果，跳过事后检验")
else:
    freq_omnibus_df = pd.DataFrame()
    freq_posthoc_df = pd.DataFrame()
    print("⚠ 无频域数据，跳过频域检验")

# %%
# 保存统计结果
conn_omnibus_df.to_csv(OUTPUT_PATH / 'connectivity_omnibus_stats.csv', index=False)
print(f"✓ 已保存: connectivity_omnibus_stats.csv")

if not conn_posthoc_df.empty:
    conn_posthoc_df.to_csv(OUTPUT_PATH / 'connectivity_posthoc_stats.csv', index=False)
    print(f"✓ 已保存: connectivity_posthoc_stats.csv")

if not freq_omnibus_df.empty:
    freq_omnibus_df.to_csv(OUTPUT_PATH / 'frequency_omnibus_stats.csv', index=False)
    print(f"✓ 已保存: frequency_omnibus_stats.csv")

if not freq_posthoc_df.empty:
    freq_posthoc_df.to_csv(OUTPUT_PATH / 'frequency_posthoc_stats.csv', index=False)
    print(f"✓ 已保存: frequency_posthoc_stats.csv")

# %% [markdown]
# ## 5. 可视化

# %%
def plot_group_boxplots(df: pd.DataFrame, metrics: List[str], title_prefix: str,
                        filename: str, freq_band: str = None):
    """三组箱线图"""
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        data_to_plot = []
        labels = []
        for g in GROUPS:
            vals = df[df['group'] == g][metric].dropna()
            data_to_plot.append(vals)
            labels.append(f"{GROUP_LABELS[g]}\n(n={len(vals)})")

        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True, widths=0.6)
        for patch, g in zip(bp['boxes'], GROUPS):
            patch.set_facecolor(GROUP_COLORS[g])
            patch.set_alpha(0.7)

        ax.set_title(metric, fontsize=12)
        ax.tick_params(axis='x', labelsize=9)

    band_str = f" ({freq_band})" if freq_band else ""
    fig.suptitle(f"{title_prefix}{band_str}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_PATH / filename, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✓ 已保存: {filename}")


# 连接性指标箱线图 (按频段)
for band in conn_df['freq_band'].unique():
    band_data = conn_df[conn_df['freq_band'] == band]
    plot_group_boxplots(
        band_data, conn_metrics,
        title_prefix="连接性指标组间比较",
        filename=f"conn_boxplot_{band}.png",
        freq_band=band
    )

# 频域指标箱线图
if not freq_df.empty:
    plot_group_boxplots(
        freq_df, FREQ_BANDS,
        title_prefix="频段功率组间比较",
        filename="freq_boxplot_band_powers.png"
    )

# %%
def plot_omnibus_heatmap(omnibus_df: pd.DataFrame, title: str, filename: str):
    """Omnibus p 值热图"""
    if omnibus_df.empty:
        return

    pivot = omnibus_df.pivot_table(index='metric', columns='freq_band', values='p_fdr')
    fig, ax = plt.subplots(figsize=(8, max(3, len(pivot) * 0.8)))

    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn_r',
                vmin=0, vmax=0.1, ax=ax, linewidths=0.5,
                cbar_kws={'label': 'FDR-corrected p'})

    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('频段')
    ax.set_ylabel('指标')
    plt.tight_layout()
    plt.savefig(FIGURES_PATH / filename, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✓ 已保存: {filename}")


plot_omnibus_heatmap(conn_omnibus_df, "连接性指标 Omnibus p 值 (FDR)", "conn_omnibus_heatmap.png")
plot_omnibus_heatmap(freq_omnibus_df, "频域指标 Omnibus p 值 (FDR)", "freq_omnibus_heatmap.png")

# %%
def plot_posthoc_effect_sizes(posthoc_df: pd.DataFrame, title: str, filename: str):
    """事后检验效应量 (Cohen's d) 森林图"""
    if posthoc_df.empty:
        return

    posthoc_df = posthoc_df.copy()
    posthoc_df['comparison'] = posthoc_df['group1'] + ' vs ' + posthoc_df['group2']
    posthoc_df['label'] = posthoc_df['metric'] + ' (' + posthoc_df.get('freq_band', '') + ')'

    fig, ax = plt.subplots(figsize=(10, max(3, len(posthoc_df) * 0.4)))

    y_pos = range(len(posthoc_df))
    colors = ['#E74C3C' if p < 0.05 else '#95A5A6' for p in posthoc_df['p_value']]

    ax.barh(y_pos, posthoc_df['cohens_d'], color=colors, alpha=0.7, height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{r['label']}\n{r['comparison']}" for _, r in posthoc_df.iterrows()], fontsize=8)
    ax.set_xlabel("Cohen's d")
    ax.set_title(title, fontsize=13, fontweight='bold')

    # 效应量参考线
    for val, label in [(0.2, '小'), (0.5, '中'), (0.8, '大')]:
        ax.axvline(val, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(-val, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(FIGURES_PATH / filename, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✓ 已保存: {filename}")


plot_posthoc_effect_sizes(conn_posthoc_df, "连接性事后检验效应量", "conn_posthoc_effects.png")
plot_posthoc_effect_sizes(freq_posthoc_df, "频域事后检验效应量", "freq_posthoc_effects.png")

# %% [markdown]
# ## 6. 生成报告

# %%
def format_omnibus_table(omnibus_df: pd.DataFrame) -> str:
    """格式化 omnibus 结果为 markdown 表格"""
    if omnibus_df.empty:
        return "无数据\n"
    lines = ["| 指标 | 频段 | 检验方法 | 统计量 | p值 | p(FDR) | 效应量 | 显著 |",
             "|------|------|----------|--------|-----|--------|--------|------|"]
    for _, r in omnibus_df.iterrows():
        sig = "✓" if r.get('significant', False) else ""
        lines.append(f"| {r['metric']} | {r['freq_band']} | {r['test']} | "
                     f"{r['statistic']:.3f} | {r['p_value']:.4f} | {r['p_fdr']:.4f} | "
                     f"{r['effect_size']:.3f} | {sig} |")
    return "\n".join(lines) + "\n"


def format_posthoc_table(posthoc_df: pd.DataFrame) -> str:
    """格式化事后检验结果为 markdown 表格"""
    if posthoc_df.empty:
        return "无显著 omnibus 结果，未执行事后检验\n"
    lines = ["| 指标 | 频段 | 比较 | 检验方法 | p值 | Cohen's d |",
             "|------|------|------|----------|-----|-----------|"]
    for _, r in posthoc_df.iterrows():
        d_str = f"{r['cohens_d']:.3f}" if pd.notna(r['cohens_d']) else "-"
        lines.append(f"| {r['metric']} | {r.get('freq_band', '-')} | "
                     f"{r['group1']} vs {r['group2']} | {r['test']} | "
                     f"{r['p_value']:.4f} | {d_str} |")
    return "\n".join(lines) + "\n"


# 样本量
n_per_group = {g: conn_df[conn_df['group'] == g]['subject_id'].nunique() for g in GROUPS}

report = f"""# 组间统计分析报告 - ADHD vs COM vs TD

**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## 研究设计

- **分析水平**: 被试水平 (每位被试贡献一个数据点)
- **组别**: {', '.join(f'{GROUP_LABELS[g]} ({g}, n={n_per_group.get(g, "?")})' for g in GROUPS)}
- **总被试数**: {sum(n_per_group.values())}

## 统计方法

- **Omnibus 检验**: 正态+方差齐 → ANOVA; 正态+方差不齐 → Welch's ANOVA; 非正态 → Kruskal-Wallis
- **事后检验** (仅 omnibus 显著时): 参数 → Tukey HSD / Games-Howell; 非参数 → Dunn (Bonferroni)
- **多重比较校正**: FDR (Benjamini-Hochberg)
- **效应量**: Omnibus η²/ε²; 事后 Cohen's d

## 连接性指标结果

### Omnibus 检验

{format_omnibus_table(conn_omnibus_df)}

### 事后检验

{format_posthoc_table(conn_posthoc_df)}

## 频域指标结果

### Omnibus 检验

{format_omnibus_table(freq_omnibus_df)}

### 事后检验

{format_posthoc_table(freq_posthoc_df)}

---

*报告由 REST EEG Pipeline 自动生成*
"""

report_file = OUTPUT_PATH / 'group_comparison_report.md'
with open(report_file, 'w', encoding='utf-8') as f:
    f.write(report)
print(f"✓ 报告已保存: {report_file}")

# %% [markdown]
# ## 7. 分析完成

# %%
print("\n" + "=" * 60)
print("三组统计分析完成!")
print("=" * 60)

print(f"\n输出文件:")
for f in sorted(OUTPUT_PATH.glob('*')):
    if f.is_file():
        print(f"  - {f.name}")

print(f"\n可视化文件:")
for f in sorted(FIGURES_PATH.glob('*.png')):
    print(f"  - {f.name}")

# 显著结果摘要
if not conn_omnibus_df.empty:
    sig = conn_omnibus_df[conn_omnibus_df['significant']]
    print(f"\n连接性显著结果 (FDR < 0.05): {len(sig)}/{len(conn_omnibus_df)}")
    if len(sig) > 0:
        print(sig[['metric', 'freq_band', 'p_fdr', 'effect_size']].to_string(index=False))

if not freq_omnibus_df.empty:
    sig = freq_omnibus_df[freq_omnibus_df['significant']]
    print(f"\n频域显著结果 (FDR < 0.05): {len(sig)}/{len(freq_omnibus_df)}")
    if len(sig) > 0:
        print(sig[['metric', 'freq_band', 'p_fdr', 'effect_size']].to_string(index=False))
