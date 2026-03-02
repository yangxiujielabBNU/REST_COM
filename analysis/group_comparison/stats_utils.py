"""
stats_utils.py — 三组 EEG 统计比较共享工具模块

共享配置、统计检验、可视化、报告生成函数。
供 connectivity_stats.ipynb 和 frequency_stats.ipynb 调用。
"""

import os
from pathlib import Path
from itertools import combinations
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
import scikit_posthocs as sp
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# =============================================================================
# 共享配置
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
FREQ_BAND_KEYS = ['delta', 'theta', 'alpha', 'beta', 'gamma']
CONN_METRICS = ['clustering', 'path_length', 'sigma', 'avg_connectivity']

# 数据路径
CONN_PATHS = {g: BASE_PATH / 'analysis' / 'connectivity_analysis' / g / 'results' for g in GROUPS}
FREQ_PATHS = {g: BASE_PATH / 'analysis' / '频域分析' / g / 'results' for g in GROUPS}


def setup_plotting():
    """配置 matplotlib 中文字体"""
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False


def ensure_output_dirs(output_path: Path):
    """创建输出目录"""
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / 'figures').mkdir(parents=True, exist_ok=True)
    return output_path, output_path / 'figures'


# =============================================================================
# 描述性统计
# =============================================================================
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


# =============================================================================
# 统计检验函数
# =============================================================================
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
        grand_mean = np.concatenate(arrays).mean()
        ss_between = sum(len(a) * (a.mean() - grand_mean) ** 2 for a in arrays)
        ss_total = sum(((a - grand_mean) ** 2).sum() for a in arrays)
        effect_size = ss_between / ss_total if ss_total > 0 else 0
        test_name = 'ANOVA'
        es_name = 'η²'
    elif all_normal and not is_homogeneous:
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
    """
    results = []

    if all_normal and is_homogeneous:
        _ph_df = pd.DataFrame({'val': np.concatenate([groups_data[g] for g in GROUPS]),
                               'grp': np.concatenate([[g] * len(groups_data[g]) for g in GROUPS])})
        posthoc_df = sp.posthoc_tukey(_ph_df, val_col='val', group_col='grp')
        posthoc_name = 'Tukey HSD'
    elif all_normal and not is_homogeneous:
        _ph_df = pd.DataFrame({'val': np.concatenate([groups_data[g] for g in GROUPS]),
                               'grp': np.concatenate([[g] * len(groups_data[g]) for g in GROUPS])})
        posthoc_df = sp.posthoc_games_howell(_ph_df, val_col='val', group_col='grp')
        posthoc_name = 'Games-Howell'
    else:
        _ph_df = pd.DataFrame({'val': np.concatenate([groups_data[g] for g in GROUPS]),
                               'grp': np.concatenate([[g] * len(groups_data[g]) for g in GROUPS])})
        posthoc_df = sp.posthoc_dunn(_ph_df, val_col='val', group_col='grp', p_adjust='bonferroni')
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


# =============================================================================
# 分析执行
# =============================================================================
def run_analysis_for_metric(df: pd.DataFrame, value_col: str, metric_label: str,
                            freq_band: str = '') -> dict:
    """对单个指标执行完整统计流程: omnibus + 事后检验"""
    groups_data = extract_groups_data(df, value_col)

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

    posthoc = []
    if omnibus['p'] < 0.05:
        posthoc = run_posthoc_tests(groups_data, omnibus['all_normal'], omnibus['is_homogeneous'])

    result['posthoc'] = posthoc
    return result


# =============================================================================
# FDR 校正
# =============================================================================
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

    reject, p_fdr, _, _ = multipletests(df['p'].values, method='fdr_bh')
    df['p_fdr'] = p_fdr
    df['significant'] = reject

    return df


def collect_posthoc(results_list: list) -> pd.DataFrame:
    """汇总多个 results 列表的事后检验结果"""
    all_posthoc = []
    for r in results_list:
        for ph in r.get('posthoc', []):
            all_posthoc.append({
                'metric': r['metric'],
                'freq_band': r['freq_band'],
                **ph,
            })
    return pd.DataFrame(all_posthoc) if all_posthoc else pd.DataFrame()


# =============================================================================
# 可视化
# =============================================================================
def plot_boxplots(df: pd.DataFrame, value_col: str, title: str, ylabel: str,
                  filepath: Path, freq_band: str = None):
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
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_effect_sizes(posthoc_df: pd.DataFrame, filepath: Path):
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
    for val, label in [(0.2, '小'), (0.5, '中'), (0.8, '大')]:
        ax.axvline(x=val, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
        ax.axvline(x=-val, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
    ax.grid(axis='x', alpha=0.3)
    sns.despine()
    fig.tight_layout()
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_significance_heatmap(omnibus_df: pd.DataFrame, title: str, filepath: Path):
    """Omnibus p 值热图"""
    if omnibus_df.empty:
        return

    pivot = omnibus_df.pivot_table(index='metric', columns='freq_band', values='p_fdr')
    fig, ax = plt.subplots(figsize=(8, max(3, len(pivot) * 0.8)))
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn_r', vmin=0, vmax=0.1,
                ax=ax, linewidths=0.5, cbar_kws={'label': 'FDR-corrected p'})
    ax.set_title(title, fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)


# =============================================================================
# 报告生成
# =============================================================================
def generate_report(title: str, analysis_desc: str, sample_sizes: dict,
                    desc_text: str, omnibus_text: str, posthoc_text: str) -> str:
    """
    生成 Markdown 分析报告

    Parameters
    ----------
    title : str - 报告标题
    analysis_desc : str - 分析内容描述
    sample_sizes : dict - {group: n}
    desc_text : str - 描述性统计 markdown
    omnibus_text : str - omnibus 结果 markdown
    posthoc_text : str - 事后检验 markdown
    """
    now = datetime.now().strftime('%Y-%m-%d %H:%M')

    report = f"""# {title}

**生成时间:** {now}

## 1. 研究设计

| 组别 | 标签 | 样本量 |
|------|------|--------|
| adhd | {GROUP_LABELS['adhd']} | {sample_sizes.get('adhd', '?')} |
| com | {GROUP_LABELS['com']} | {sample_sizes.get('com', '?')} |
| td | {GROUP_LABELS['td']} | {sample_sizes.get('td', '?')} |

**统计策略:** 被试水平分析 (epoch 先在被试内平均)

**分析指标:** {analysis_desc}

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


def format_desc_table(df: pd.DataFrame, value_col: str, label: str = '') -> str:
    """生成单个指标的描述性统计 markdown 表格"""
    desc = compute_descriptive_stats(df, value_col)
    lines = []
    if label:
        lines.append(f"*{label}*\n")
    lines.append("| 组 | n | M ± SD | Mdn (IQR) |")
    lines.append("|---|---|--------|-----------|")
    for _, row in desc.iterrows():
        lines.append(
            f"| {row['group_label']} | {row['n']:.0f} | "
            f"{row['mean']:.4f} ± {row['std']:.4f} | "
            f"{row['median']:.4f} ({row['q25']:.4f}-{row['q75']:.4f}) |"
        )
    lines.append("")
    return '\n'.join(lines)


def format_omnibus_table(omnibus_df: pd.DataFrame, label: str) -> str:
    """生成 omnibus 结果 markdown 表格"""
    if omnibus_df.empty:
        return ""
    lines = [f"### {label}\n"]
    lines.append("| 指标 | 频段 | 检验 | 统计量 | p | p(FDR) | 效应量 | 显著 |")
    lines.append("|------|------|------|--------|---|--------|--------|------|")
    for _, row in omnibus_df.iterrows():
        sig = '✓' if row['significant'] else ''
        lines.append(
            f"| {row['metric']} | {row['freq_band']} | {row['test']} | "
            f"{row['statistic']:.3f} | {row['p']:.4f} | {row['p_fdr']:.4f} | "
            f"{row['es_name']}={row['effect_size']:.3f} | {sig} |"
        )
    lines.append("")
    return '\n'.join(lines)


def format_posthoc_table(posthoc_df: pd.DataFrame) -> str:
    """生成事后检验 markdown 表格"""
    if posthoc_df.empty:
        return "无 omnibus 显著结果，未执行事后检验。"
    lines = ["| 指标 | 频段 | 比较 | 检验 | p | Cohen's d | 显著 |",
             "|------|------|------|------|---|-----------|------|"]
    for _, row in posthoc_df.iterrows():
        sig = '✓' if row['significant'] else ''
        lines.append(
            f"| {row['metric']} | {row['freq_band']} | {row['pair_label']} | "
            f"{row['test']} | {row['p']:.4f} | {row['cohens_d']:.3f} | {sig} |"
        )
    return '\n'.join(lines)
