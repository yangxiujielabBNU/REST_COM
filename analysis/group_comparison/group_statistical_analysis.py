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
CONN_METRICS = ['clustering', 'path_length', 'sigma', 'avg_connectivity']

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
    """加载三组连接性数据 (small_world_metrics.csv)"""
    dfs = []
    for g in GROUPS:
        csv_path = CONN_PATHS[g] / 'small_world_metrics.csv'
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df['group'] = g
            dfs.append(df)
            print(f"  {GROUP_LABELS[g]}: {df['subject_id'].nunique()} 被试, {len(df)} 行")
        else:
            print(f"  ⚠ 未找到: {csv_path}")
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def load_frequency_data() -> pd.DataFrame:
    """加载三组频域数据 (subject_band_powers.csv)"""
    dfs = []
    for g in GROUPS:
        csv_path = FREQ_PATHS[g] / 'subject_band_powers.csv'
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df['group'] = g
            dfs.append(df)
            print(f"  {GROUP_LABELS[g]}: {len(df)} 被试")
        else:
            print(f"  ⚠ 未找到: {csv_path}")
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


print("--- 加载连接性数据 ---")
conn_df = load_connectivity_data()
print(f"\n连接性数据: {conn_df.shape}")

print("\n--- 加载频域数据 ---")
freq_df = load_frequency_data()
print(f"\n频域数据: {freq_df.shape}")

# %% [markdown]
# ## 3. 描述性统计

# %%
def compute_descriptive_stats(df: pd.DataFrame, value_col: str, group_col: str = 'group') -> pd.DataFrame:
    """按组计算描述性统计"""
    rows = []
    for g in GROUPS:
        vals = df[df[group_col] == g][value_col].dropna()
        rows.append({
            'group': g,
            'group_label': GROUP_LABELS[g],
            'n': len(vals),
            'mean': vals.mean(),
            'std': vals.std(),
            'median': vals.median(),
            'q25': vals.quantile(0.25),
            'q75': vals.quantile(0.75),
            'min': vals.min(),
            'max': vals.max(),
        })
    return pd.DataFrame(rows)

# %% [markdown]
# ## 4. 统计检验函数

# %%
def extract_groups_data(df: pd.DataFrame, value_col: str, group_col: str = 'group') -> dict:
    """从 DataFrame 提取每组数据数组"""
    return {g: df[df[group_col] == g][value_col].dropna().values for g in GROUPS}


def test_normality(groups_data: dict) -> dict:
    """Shapiro-Wilk 正态性检验 (每组)"""
    results = {}
    for g, vals in groups_data.items():
        if len(vals) >= 3:
            stat, p = stats.shapiro(vals)
            results[g] = {'statistic': stat, 'p': p, 'is_normal': p > 0.05}
        else:
            results[g] = {'statistic': np.nan, 'p': np.nan, 'is_normal': False}
    return results


def test_homogeneity(groups_data: dict) -> dict:
    """Levene 方差齐性检验"""
    arrays = [groups_data[g] for g in GROUPS]
    stat, p = stats.levene(*arrays)
    return {'statistic': stat, 'p': p, 'is_homogeneous': p > 0.05}


def run_omnibus_test(groups_data: dict) -> dict:
    """
    根据正态性和方差齐性选择 omnibus 检验:
    - 全正态 + 方差齐 → ANOVA (η²)
    - 全正态 + 方差不齐 → Welch's ANOVA (η²)
    - 任一非正态 → Kruskal-Wallis (ε²)
    """
    normality = test_normality(groups_data)
    homogeneity = test_homogeneity(groups_data)
    all_normal = all(normality[g]['is_normal'] for g in GROUPS)
    is_homogeneous = homogeneity['is_homogeneous']

    arrays = [groups_data[g] for g in GROUPS]
    n_total = sum(len(a) for a in arrays)
    k = len(arrays)

    if all_normal and is_homogeneous:
        stat, p = stats.f_oneway(*arrays)
        # η² = SS_between / SS_total
        grand_mean = np.concatenate(arrays).mean()
        ss_between = sum(len(a) * (a.mean() - grand_mean) ** 2 for a in arrays)
        ss_total = sum(((a - grand_mean) ** 2).sum() for a in arrays)
        effect_size = ss_between / ss_total if ss_total > 0 else 0
        test_name = 'ANOVA'
        es_name = 'η²'
    elif all_normal and not is_homogeneous:
        # Welch's ANOVA (scipy >= 1.7)
        result = stats.alexandergovern(*arrays)
        stat, p = result.statistic, result.pvalue
        grand_mean = np.concatenate(arrays).mean()
        ss_between = sum(len(a) * (a.mean() - grand_mean) ** 2 for a in arrays)
        ss_total = sum(((a - grand_mean) ** 2).sum() for a in arrays)
        effect_size = ss_between / ss_total if ss_total > 0 else 0
        test_name = "Welch's ANOVA"
        es_name = 'η²'
    else:
        stat, p = stats.kruskal(*arrays)
        # ε² = (H - k + 1) / (n - k)
        effect_size = (stat - k + 1) / (n_total - k) if (n_total - k) > 0 else 0
        test_name = 'Kruskal-Wallis'
        es_name = 'ε²'

    return {
        'test': test_name,
        'statistic': stat,
        'p': p,
        'effect_size': effect_size,
        'es_name': es_name,
        'all_normal': all_normal,
        'is_homogeneous': is_homogeneous,
        'normality': normality,
        'homogeneity': homogeneity,
    }


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """计算 Cohen's d (pooled SD)"""
    n1, n2 = len(a), len(b)
    var1, var2 = a.var(ddof=1), b.var(ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (a.mean() - b.mean()) / pooled_std if pooled_std > 0 else 0


def run_posthoc_tests(groups_data: dict, all_normal: bool, is_homogeneous: bool) -> list:
    """
    事后两两比较:
    - 参数 + 方差齐 → Tukey HSD
    - 参数 + 方差不齐 → Games-Howell
    - 非参数 → Dunn (Bonferroni)
    每对计算 Cohen's d
    """
    results = []

    if all_normal and is_homogeneous:
        # Tukey HSD via scikit_posthocs
        all_vals = np.concatenate([groups_data[g] for g in GROUPS])
        all_groups = np.concatenate([[g] * len(groups_data[g]) for g in GROUPS])
        posthoc_df = sp.posthoc_tukey(all_vals, all_groups)
        posthoc_name = 'Tukey HSD'
    elif all_normal and not is_homogeneous:
        all_vals = np.concatenate([groups_data[g] for g in GROUPS])
        all_groups = np.concatenate([[g] * len(groups_data[g]) for g in GROUPS])
        posthoc_df = sp.posthoc_games_howell(all_vals, all_groups)
        posthoc_name = 'Games-Howell'
    else:
        all_vals = np.concatenate([groups_data[g] for g in GROUPS])
        all_groups = np.concatenate([[g] * len(groups_data[g]) for g in GROUPS])
        posthoc_df = sp.posthoc_dunn(all_vals, all_groups, p_adjust='bonferroni')
        posthoc_name = 'Dunn (Bonferroni)'

    for g1, g2 in GROUP_PAIRS:
        p_val = posthoc_df.loc[g1, g2]
        d = cohens_d(groups_data[g1], groups_data[g2])
        results.append({
            'pair': f"{g1} vs {g2}",
            'pair_label': f"{GROUP_LABELS[g1]} vs {GROUP_LABELS[g2]}",
            'test': posthoc_name,
            'p': p_val,
            'cohens_d': d,
            'significant': p_val < 0.05,
        })

    return results


# %% [markdown]
# ## 5. 执行统计分析

# %%
def run_analysis_for_metric(df: pd.DataFrame, value_col: str, metric_label: str,
                            freq_band: str = '') -> dict:
    """对单个指标执行完整统计流程: omnibus + 事后检验"""
    groups_data = extract_groups_data(df, value_col)

    # 检查每组样本量
    for g in GROUPS:
        if len(groups_data[g]) < 3:
            print(f"  ⚠ {GROUP_LABELS[g]} 样本量不足 ({len(groups_data[g])}), 跳过")
            return None

    omnibus = run_omnibus_test(groups_data)

    result = {
        'metric': metric_label,
        'freq_band': freq_band,
        'test': omnibus['test'],
        'statistic': omnibus['statistic'],
        'p': omnibus['p'],
        'effect_size': omnibus['effect_size'],
        'es_name': omnibus['es_name'],
        'all_normal': omnibus['all_normal'],
        'is_homogeneous': omnibus['is_homogeneous'],
    }

    # 事后检验 (仅 omnibus p < 0.05)
    posthoc = []
    if omnibus['p'] < 0.05:
        posthoc = run_posthoc_tests(groups_data, omnibus['all_normal'], omnibus['is_homogeneous'])

    result['posthoc'] = posthoc
    return result


# --- 连接性指标分析 ---
print("\n" + "=" * 60)
print("连接性指标 Omnibus 检验")
print("=" * 60)

conn_results = []
if not conn_df.empty:
    bands_in_data = conn_df['freq_band'].unique()
    for band in sorted(bands_in_data):
        band_data = conn_df[conn_df['freq_band'] == band]
        for metric in CONN_METRICS:
            if metric not in band_data.columns:
                continue
            res = run_analysis_for_metric(band_data, metric, metric, freq_band=band)
            if res:
                conn_results.append(res)
                sig_mark = '***' if res['p'] < 0.001 else '**' if res['p'] < 0.01 else '*' if res['p'] < 0.05 else ''
                print(f"  {band:>8s} | {metric:<18s} | {res['test']:<16s} | "
                      f"p={res['p']:.4f} {sig_mark:>3s} | {res['es_name']}={res['effect_size']:.3f}")

# --- 频域指标分析 ---
print("\n" + "=" * 60)
print("频域指标 Omnibus 检验")
print("=" * 60)

freq_results = []
if not freq_df.empty:
    for band in FREQ_BANDS:
        if band not in freq_df.columns:
            continue
        res = run_analysis_for_metric(freq_df, band, 'band_power', freq_band=band)
        if res:
            freq_results.append(res)
            sig_mark = '***' if res['p'] < 0.001 else '**' if res['p'] < 0.01 else '*' if res['p'] < 0.05 else ''
            print(f"  {band:<8s} | {res['test']:<16s} | "
                  f"p={res['p']:.4f} {sig_mark:>3s} | {res['es_name']}={res['effect_size']:.3f}")

# %% [markdown]
# ## 5.1 FDR 校正

# %%
def apply_fdr(results: list) -> pd.DataFrame:
    """对 omnibus p 值进行 FDR 校正, 返回汇总 DataFrame"""
    if not results:
        return pd.DataFrame()

    rows = []
    for r in results:
        rows.append({
            'metric': r['metric'],
            'freq_band': r['freq_band'],
            'test': r['test'],
            'statistic': r['statistic'],
            'p': r['p'],
            'effect_size': r['effect_size'],
            'es_name': r['es_name'],
            'all_normal': r['all_normal'],
            'is_homogeneous': r['is_homogeneous'],
        })
    df = pd.DataFrame(rows)

    # FDR 校正
    reject, p_fdr, _, _ = multipletests(df['p'].values, method='fdr_bh')
    df['p_fdr'] = p_fdr
    df['significant'] = reject

    return df


conn_omnibus_df = apply_fdr(conn_results)
freq_omnibus_df = apply_fdr(freq_results)

if not conn_omnibus_df.empty:
    print("\n连接性 Omnibus 结果 (FDR 校正后):")
    print(conn_omnibus_df[['metric', 'freq_band', 'test', 'p', 'p_fdr', 'significant', 'effect_size']].to_string(index=False))

if not freq_omnibus_df.empty:
    print("\n频域 Omnibus 结果 (FDR 校正后):")
    print(freq_omnibus_df[['metric', 'freq_band', 'test', 'p', 'p_fdr', 'significant', 'effect_size']].to_string(index=False))

# --- 汇总事后检验结果 ---
all_posthoc = []
for r in conn_results + freq_results:
    for ph in r.get('posthoc', []):
        all_posthoc.append({
            'metric': r['metric'],
            'freq_band': r['freq_band'],
            **ph,
        })
posthoc_df = pd.DataFrame(all_posthoc) if all_posthoc else pd.DataFrame()

if not posthoc_df.empty:
    print("\n事后检验结果 (仅 omnibus 显著指标):")
    print(posthoc_df[['metric', 'freq_band', 'pair', 'test', 'p', 'cohens_d', 'significant']].to_string(index=False))

# --- 保存 CSV ---
conn_omnibus_df.to_csv(OUTPUT_PATH / 'connectivity_omnibus_stats.csv', index=False, encoding='utf-8-sig')
freq_omnibus_df.to_csv(OUTPUT_PATH / 'frequency_omnibus_stats.csv', index=False, encoding='utf-8-sig')
if not posthoc_df.empty:
    posthoc_df.to_csv(OUTPUT_PATH / 'posthoc_stats.csv', index=False, encoding='utf-8-sig')
print("\n✓ 统计结果已保存")

# %% [markdown]
# ## 6. 可视化

# %%
def plot_boxplots(df: pd.DataFrame, value_col: str, title: str, ylabel: str,
                  filename: str, freq_band: str = None):
    """三组箱线图 + 散点"""
    if freq_band:
        plot_df = df[df['freq_band'] == freq_band] if 'freq_band' in df.columns else df
    else:
        plot_df = df

    fig, ax = plt.subplots(figsize=(6, 5))
    positions = range(len(GROUPS))
    bp_data = [plot_df[plot_df['group'] == g][value_col].dropna().values for g in GROUPS]

    bp = ax.boxplot(bp_data, positions=positions, widths=0.5, patch_artist=True,
                    showmeans=True, meanprops=dict(marker='D', markerfacecolor='white', markersize=6))
    for patch, g in zip(bp['boxes'], GROUPS):
        patch.set_facecolor(GROUP_COLORS[g])
        patch.set_alpha(0.6)

    # 散点
    for i, g in enumerate(GROUPS):
        vals = plot_df[plot_df['group'] == g][value_col].dropna().values
        jitter = np.random.normal(0, 0.05, len(vals))
        ax.scatter([i] * len(vals) + jitter, vals, color=GROUP_COLORS[g],
                   alpha=0.7, s=30, zorder=3, edgecolors='white', linewidth=0.5)

    ax.set_xticks(positions)
    ax.set_xticklabels([GROUP_LABELS[g] for g in GROUPS], fontsize=10)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    sns.despine()
    fig.tight_layout()
    fig.savefig(FIGURES_PATH / filename, dpi=150, bbox_inches='tight')
    plt.close(fig)


# 连接性箱线图
if not conn_df.empty:
    bands_in_data = sorted(conn_df['freq_band'].unique())
    for metric in CONN_METRICS:
        for band in bands_in_data:
            plot_boxplots(conn_df, metric,
                          f'{metric} ({band})', metric,
                          f'boxplot_conn_{metric}_{band}.png',
                          freq_band=band)
    print("✓ 连接性箱线图已保存")

# 频域箱线图
if not freq_df.empty:
    for band in FREQ_BANDS:
        if band in freq_df.columns:
            plot_boxplots(freq_df, band,
                          f'{band} 频段功率', f'{band} Power (μV²/Hz)',
                          f'boxplot_freq_{band}.png')
    print("✓ 频域箱线图已保存")

# %%
# --- 效应量森林图 ---
def plot_effect_sizes(posthoc_df: pd.DataFrame, filename: str):
    """事后检验 Cohen's d 森林图"""
    if posthoc_df.empty:
        print("  无显著事后检验结果, 跳过森林图")
        return

    fig, ax = plt.subplots(figsize=(10, max(4, len(posthoc_df) * 0.35)))
    y_labels = [f"{r['freq_band']} {r['metric']}\n{r['pair']}" for _, r in posthoc_df.iterrows()]
    y_pos = range(len(y_labels))
    colors = ['#E74C3C' if r['significant'] else '#95A5A6' for _, r in posthoc_df.iterrows()]

    ax.barh(y_pos, posthoc_df['cohens_d'].values, color=colors, alpha=0.7, height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_xlabel("Cohen's d", fontsize=11)
    ax.set_title("事后检验效应量 (Cohen's d)", fontsize=13, fontweight='bold')
    ax.axvline(x=0, color='black', linewidth=0.8)
    # 效应量参考线
    for val, label in [(0.2, '小'), (0.5, '中'), (0.8, '大')]:
        ax.axvline(x=val, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
        ax.axvline(x=-val, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
    ax.grid(axis='x', alpha=0.3)
    sns.despine()
    fig.tight_layout()
    fig.savefig(FIGURES_PATH / filename, dpi=150, bbox_inches='tight')
    plt.close(fig)


plot_effect_sizes(posthoc_df, 'effect_sizes_forest.png')

# %%
# --- 显著性热图 ---
def plot_significance_heatmap(omnibus_df: pd.DataFrame, title: str, filename: str):
    """Omnibus p 值热图"""
    if omnibus_df.empty:
        return

    pivot = omnibus_df.pivot_table(index='metric', columns='freq_band', values='p_fdr')
    fig, ax = plt.subplots(figsize=(8, max(3, len(pivot) * 0.8)))
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn_r', vmin=0, vmax=0.1,
                ax=ax, linewidths=0.5, cbar_kws={'label': 'FDR-corrected p'})
    ax.set_title(title, fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(FIGURES_PATH / filename, dpi=150, bbox_inches='tight')
    plt.close(fig)


plot_significance_heatmap(conn_omnibus_df, '连接性指标 Omnibus p 值 (FDR)', 'heatmap_conn_significance.png')
plot_significance_heatmap(freq_omnibus_df, '频域指标 Omnibus p 值 (FDR)', 'heatmap_freq_significance.png')
print("✓ 可视化完成")

# %% [markdown]
# ## 7. 生成报告

# %%
def generate_report() -> str:
    """生成 Markdown 分析报告"""
    now = datetime.now().strftime('%Y-%m-%d %H:%M')

    # 描述性统计表格
    desc_sections = []

    if not conn_df.empty:
        desc_sections.append("### 连接性指标\n")
        for band in sorted(conn_df['freq_band'].unique()):
            band_data = conn_df[conn_df['freq_band'] == band]
            desc_sections.append(f"**{band} 频段:**\n")
            for metric in CONN_METRICS:
                if metric not in band_data.columns:
                    continue
                desc = compute_descriptive_stats(band_data, metric)
                desc_sections.append(f"*{metric}*\n")
                desc_sections.append(f"| 组 | n | M ± SD | Mdn (IQR) |")
                desc_sections.append(f"|---|---|--------|-----------|")
                for _, row in desc.iterrows():
                    desc_sections.append(
                        f"| {row['group_label']} | {row['n']:.0f} | "
                        f"{row['mean']:.4f} ± {row['std']:.4f} | "
                        f"{row['median']:.4f} ({row['q25']:.4f}-{row['q75']:.4f}) |"
                    )
                desc_sections.append("")

    if not freq_df.empty:
        desc_sections.append("### 频域指标\n")
        for band in FREQ_BANDS:
            if band not in freq_df.columns:
                continue
            desc = compute_descriptive_stats(freq_df, band)
            desc_sections.append(f"**{band}:**\n")
            desc_sections.append(f"| 组 | n | M ± SD | Mdn (IQR) |")
            desc_sections.append(f"|---|---|--------|-----------|")
            for _, row in desc.iterrows():
                desc_sections.append(
                    f"| {row['group_label']} | {row['n']:.0f} | "
                    f"{row['mean']:.4f} ± {row['std']:.4f} | "
                    f"{row['median']:.4f} ({row['q25']:.4f}-{row['q75']:.4f}) |"
                )
            desc_sections.append("")

    desc_text = '\n'.join(desc_sections)

    # Omnibus 结果表格
    omnibus_sections = []
    for label, odf in [('连接性', conn_omnibus_df), ('频域', freq_omnibus_df)]:
        if odf.empty:
            continue
        omnibus_sections.append(f"### {label}指标\n")
        omnibus_sections.append("| 指标 | 频段 | 检验 | 统计量 | p | p(FDR) | 效应量 | 显著 |")
        omnibus_sections.append("|------|------|------|--------|---|--------|--------|------|")
        for _, row in odf.iterrows():
            sig = '✓' if row['significant'] else ''
            omnibus_sections.append(
                f"| {row['metric']} | {row['freq_band']} | {row['test']} | "
                f"{row['statistic']:.3f} | {row['p']:.4f} | {row['p_fdr']:.4f} | "
                f"{row['es_name']}={row['effect_size']:.3f} | {sig} |"
            )
        omnibus_sections.append("")
    omnibus_text = '\n'.join(omnibus_sections)

    # 事后检验表格
    posthoc_text = ""
    if not posthoc_df.empty:
        ph_lines = ["| 指标 | 频段 | 比较 | 检验 | p | Cohen's d | 显著 |",
                     "|------|------|------|------|---|-----------|------|"]
        for _, row in posthoc_df.iterrows():
            sig = '✓' if row['significant'] else ''
            ph_lines.append(
                f"| {row['metric']} | {row['freq_band']} | {row['pair_label']} | "
                f"{row['test']} | {row['p']:.4f} | {row['cohens_d']:.3f} | {sig} |"
            )
        posthoc_text = '\n'.join(ph_lines)
    else:
        posthoc_text = "无 omnibus 显著结果，未执行事后检验。"

    report = f"""# 三组 EEG 统计比较报告

**生成时间:** {now}

## 1. 研究设计

| 组别 | 标签 | 样本量 |
|------|------|--------|
| adhd | {GROUP_LABELS['adhd']} | {len(conn_df[conn_df['group']=='adhd']['subject_id'].unique()) if not conn_df.empty else '?'} |
| com | {GROUP_LABELS['com']} | {len(conn_df[conn_df['group']=='com']['subject_id'].unique()) if not conn_df.empty else '?'} |
| td | {GROUP_LABELS['td']} | {len(conn_df[conn_df['group']=='td']['subject_id'].unique()) if not conn_df.empty else '?'} |

**统计策略:** 被试水平分析 (epoch 先在被试内平均)

**Omnibus 检验:**
- 三组均正态 + 方差齐 → 单因素 ANOVA (η²)
- 三组均正态 + 方差不齐 → Welch's ANOVA (η²)
- 任一组非正态 → Kruskal-Wallis (ε²)

**事后检验 (仅 omnibus p < 0.05):**
- 参数 + 方差齐 → Tukey HSD
- 参数 + 方差不齐 → Games-Howell
- 非参数 → Dunn (Bonferroni)

**多重比较校正:** FDR (Benjamini-Hochberg)

## 2. 描述性统计

{desc_text}

## 3. Omnibus 检验结果

{omnibus_text}

## 4. 事后检验结果

{posthoc_text}

---

*报告由 REST EEG Pipeline 自动生成*
"""
    return report


report = generate_report()
report_file = OUTPUT_PATH / 'group_comparison_report.md'
with open(report_file, 'w', encoding='utf-8') as f:
    f.write(report)
print(f"✓ 报告已保存: {report_file}")

# %% [markdown]
# ## 8. 分析完成

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
