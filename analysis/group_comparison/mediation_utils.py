"""
mediation_utils.py — Bootstrap 中介分析工具模块

中介模型: Group (X) → Brain Connectivity (M) → Behavioral Outcome (Y)

基于 Preacher & Hayes (2008) 的 bootstrap 方法估计间接效应置信区间。
支持单变量中介、批量中介（多 M × 多 Y）、NBS 分量级中介。

依赖: numpy, pandas, statsmodels, joblib, matplotlib
"""

import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from joblib import Parallel, delayed
from statsmodels.stats.multitest import multipletests


# =============================================================================
# 数据加载与清洗
# =============================================================================
def load_behavioral_data(csv_path):
    """
    加载并清洗行为数据 CSV。

    处理逻辑:
    - -999 → NaN
    - "75（71.5）" 类格式 → 取第一个数值
    - 提取数字 ID: BZK018 → "018"

    Parameters
    ----------
    csv_path : str
        CSV 文件路径

    Returns
    -------
    pd.DataFrame
        清洗后的行为数据，含 'subject_id' 列（纯数字字符串，如 "018"）
    """
    df = pd.read_csv(csv_path, encoding='utf-8')

    # 删除全空行
    df = df.dropna(how='all').reset_index(drop=True)
    # 删除被试编号为空的行
    df = df.dropna(subset=['被试编号']).reset_index(drop=True)

    # 提取纯数字 ID (BZK018 → "018")
    df['subject_id'] = df['被试编号'].str.extract(r'(\d+)')[0]

    # 数值列清洗
    numeric_cols = ['年龄', '150字', '1分钟阅读平均', '数字RAN均值', '物体RAN均值',
                    '阅读流畅性', '音位删除', '部首意识']

    for col in numeric_cols:
        if col not in df.columns:
            continue
        # 处理含中文括号的格式: "75（71.5）" → 取第一个数值
        df[col] = df[col].astype(str).apply(_extract_first_number)
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # -999 → NaN
    df = df.replace(-999, np.nan)
    df = df.replace(-999.0, np.nan)

    print(f"加载行为数据: {len(df)} 名被试")
    print(f"  分组: {df['分组'].value_counts().to_dict()}")
    n_missing = df[numeric_cols].isna().sum()
    if n_missing.any():
        print(f"  缺失值: {n_missing[n_missing > 0].to_dict()}")

    return df


def _extract_first_number(s):
    """从字符串中提取第一个数值。'75（71.5）' → 75, 'nan' → NaN"""
    if pd.isna(s) or s == 'nan' or s == '':
        return np.nan
    match = re.match(r'([0-9]+\.?[0-9]*)', str(s))
    return float(match.group(1)) if match else np.nan


def merge_behavior_connectivity(behavior_df, conn_matrices, subject_ids,
                                 edges=None, component_edges=None,
                                 channel_names=None):
    """
    将行为数据与连接性指标按 subject_id 对齐。

    Parameters
    ----------
    behavior_df : pd.DataFrame
        load_behavioral_data 返回的行为数据
    conn_matrices : ndarray (n_subjects, n_channels, n_channels)
        单个频段的连接性矩阵
    subject_ids : array-like
        连接性数据中每个被试的 ID（字符串, 如 ["018", "029", ...]）
    edges : list of (i, j), optional
        需要提取的边索引列表。提取后作为单独列 "edge_i_j"
    component_edges : list of (i, j), optional
        NBS 分量的边列表。计算分量内平均连接强度 "component_mean"
    channel_names : list, optional
        通道名列表（用于命名边列）

    Returns
    -------
    pd.DataFrame
        合并后的数据，只保留两边都有的被试
    """
    # 构建连接性 DataFrame
    conn_rows = []
    subject_ids_str = [str(sid).zfill(3) if str(sid).isdigit() else str(sid)
                       for sid in subject_ids]

    for idx, sid in enumerate(subject_ids_str):
        row = {'subject_id': sid}
        mat = conn_matrices[idx]

        # 全局平均连接
        upper = mat[np.triu_indices_from(mat, k=1)]
        row['avg_connectivity'] = np.mean(upper)

        # 指定边
        if edges is not None:
            for i, j in edges:
                label = f"edge_{i}_{j}"
                if channel_names:
                    label = f"edge_{channel_names[i]}_{channel_names[j]}"
                row[label] = mat[i, j]

        # NBS 分量平均
        if component_edges is not None:
            vals = [mat[i, j] for i, j in component_edges]
            row['component_mean'] = np.mean(vals)

        conn_rows.append(row)

    conn_df = pd.DataFrame(conn_rows)

    # 合并
    merged = pd.merge(behavior_df, conn_df, on='subject_id', how='inner')

    n_behavior = len(behavior_df)
    n_conn = len(conn_df)
    n_merged = len(merged)
    print(f"数据合并: 行为={n_behavior}, 连接性={n_conn}, 匹配={n_merged}")
    if n_merged < n_conn:
        missing_in_behavior = set(subject_ids_str) - set(behavior_df['subject_id'])
        if missing_in_behavior:
            print(f"  连接性数据中无行为数据的被试: {missing_in_behavior}")

    return merged


# =============================================================================
# 核心中介检验 (Preacher & Hayes Bootstrap)
# =============================================================================
def mediation_bootstrap(X, M, Y, covariates=None, n_bootstrap=5000, ci=95,
                         seed=42):
    """
    Bootstrap 中介分析 (Preacher & Hayes, 2008)。

    模型:
        Path a:  M ~ X (+ covariates)
        Path b:  Y ~ X + M (+ covariates)  → b = M 的系数, c' = X 的系数
        Path c:  Y ~ X (+ covariates)       → 总效应
        间接效应 = a × b

    Parameters
    ----------
    X : array-like (n,)
        自变量 (如组别编码: 0/1)
    M : array-like (n,)
        中介变量 (如连接强度)
    Y : array-like (n,)
        因变量 (如行为分数)
    covariates : ndarray (n, p), optional
        协变量矩阵 (如年龄)
    n_bootstrap : int
        Bootstrap 抽样次数
    ci : float
        置信区间百分比 (默认 95)
    seed : int
        随机种子

    Returns
    -------
    dict
        a, a_se, a_p : Path a 系数、标准误、p 值
        b, b_se, b_p : Path b 系数
        c, c_se, c_p : 总效应
        c_prime, c_prime_se, c_prime_p : 直接效应
        indirect : 间接效应点估计 (a × b)
        indirect_ci : (lower, upper) 置信区间
        indirect_p : 间接效应 p 值 (bootstrap 分布中 0 的比例 × 2)
        proportion_mediated : 中介比例 |indirect / c|
        n : 样本量
        bootstrap_dist : 间接效应 bootstrap 分布
    """
    X = np.asarray(X, dtype=float).ravel()
    M = np.asarray(M, dtype=float).ravel()
    Y = np.asarray(Y, dtype=float).ravel()

    # 移除任何含 NaN 的行
    mask = ~(np.isnan(X) | np.isnan(M) | np.isnan(Y))
    if covariates is not None:
        covariates = np.asarray(covariates, dtype=float)
        if covariates.ndim == 1:
            covariates = covariates.reshape(-1, 1)
        mask &= ~np.isnan(covariates).any(axis=1)
        covariates = covariates[mask]
    X, M, Y = X[mask], M[mask], Y[mask]
    n = len(X)

    if n < 10:
        print(f"  警告: 有效样本量过小 (n={n})，中介分析结果不可靠")

    # --- 原始数据估计 ---
    a_result = _ols_fit(M, X, covariates)
    b_result = _ols_fit(Y, np.column_stack([X, M]),
                        covariates)
    c_result = _ols_fit(Y, X, covariates)

    a = a_result['coefs'][0]       # X → M
    b = b_result['coefs'][1]       # M → Y (控制 X)
    c_prime = b_result['coefs'][0] # X → Y (控制 M)
    c = c_result['coefs'][0]       # X → Y (总效应)
    indirect = a * b

    # --- Bootstrap ---
    rng = np.random.default_rng(seed)
    boot_indirect = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        X_b, M_b, Y_b = X[idx], M[idx], Y[idx]
        cov_b = covariates[idx] if covariates is not None else None

        a_b = _ols_fit(M_b, X_b, cov_b)['coefs'][0]
        b_b = _ols_fit(Y_b, np.column_stack([X_b, M_b]), cov_b)['coefs'][1]
        boot_indirect[i] = a_b * b_b

    # CI (percentile method)
    alpha_ci = (100 - ci) / 2
    ci_lower = np.percentile(boot_indirect, alpha_ci)
    ci_upper = np.percentile(boot_indirect, 100 - alpha_ci)

    # p-value: 2 × proportion of bootstrap samples on the other side of 0
    if indirect >= 0:
        p_indirect = 2 * np.mean(boot_indirect <= 0)
    else:
        p_indirect = 2 * np.mean(boot_indirect >= 0)
    p_indirect = min(p_indirect, 1.0)

    # Proportion mediated
    prop_med = np.abs(indirect / c) if np.abs(c) > 1e-10 else np.nan

    return {
        'a': a, 'a_se': a_result['ses'][0], 'a_p': a_result['pvals'][0],
        'b': b, 'b_se': b_result['ses'][1], 'b_p': b_result['pvals'][1],
        'c': c, 'c_se': c_result['ses'][0], 'c_p': c_result['pvals'][0],
        'c_prime': c_prime, 'c_prime_se': b_result['ses'][0],
        'c_prime_p': b_result['pvals'][0],
        'indirect': indirect,
        'indirect_ci': (ci_lower, ci_upper),
        'indirect_p': p_indirect,
        'proportion_mediated': prop_med,
        'n': n,
        'bootstrap_dist': boot_indirect,
    }


def _ols_fit(y, X_vars, covariates=None):
    """
    OLS 回归辅助函数。

    Parameters
    ----------
    y : array (n,)
    X_vars : array (n,) 或 (n, k) — 主要预测变量
    covariates : array (n, p) 或 None

    Returns
    -------
    dict : coefs, ses, pvals (仅对应 X_vars 的列，不含截距和协变量)
    """
    if X_vars.ndim == 1:
        X_vars = X_vars.reshape(-1, 1)
    n_main = X_vars.shape[1]

    if covariates is not None:
        design = np.column_stack([X_vars, covariates])
    else:
        design = X_vars

    design = sm.add_constant(design)

    try:
        model = sm.OLS(y, design).fit()
        # 系数: [const, X_var1, X_var2, ..., cov1, cov2, ...]
        # 返回 X_vars 对应的系数 (索引 1 到 n_main)
        coefs = model.params[1:1 + n_main]
        ses = model.bse[1:1 + n_main]
        pvals = model.pvalues[1:1 + n_main]
    except Exception:
        coefs = np.full(n_main, np.nan)
        ses = np.full(n_main, np.nan)
        pvals = np.full(n_main, np.nan)

    return {'coefs': coefs, 'ses': ses, 'pvals': pvals}


# =============================================================================
# 批量中介分析
# =============================================================================
def batch_mediation(X, mediators_dict, behaviors_dict, covariates=None,
                     n_bootstrap=5000, seed=42, fdr_method='fdr_bh'):
    """
    对多个中介变量 × 行为变量组合批量运行中介分析。

    Parameters
    ----------
    X : array-like (n,)
        自变量 (组别编码)
    mediators_dict : dict
        {mediator_name: array (n,)} 中介变量字典
    behaviors_dict : dict
        {behavior_name: array (n,)} 因变量字典
    covariates : ndarray (n, p), optional
    n_bootstrap : int
    seed : int
    fdr_method : str
        FDR 校正方法 (默认 Benjamini-Hochberg)

    Returns
    -------
    pd.DataFrame
        汇总结果，含 FDR 校正后的 p 值
    """
    rows = []
    for m_name, M in mediators_dict.items():
        for y_name, Y in behaviors_dict.items():
            print(f"  中介: {m_name} → {y_name}")
            res = mediation_bootstrap(X, M, Y, covariates=covariates,
                                       n_bootstrap=n_bootstrap, seed=seed)
            rows.append({
                'mediator': m_name,
                'behavior': y_name,
                'n': res['n'],
                'a': res['a'], 'a_p': res['a_p'],
                'b': res['b'], 'b_p': res['b_p'],
                'c': res['c'], 'c_p': res['c_p'],
                'c_prime': res['c_prime'], 'c_prime_p': res['c_prime_p'],
                'indirect': res['indirect'],
                'indirect_ci_lo': res['indirect_ci'][0],
                'indirect_ci_hi': res['indirect_ci'][1],
                'indirect_p': res['indirect_p'],
                'proportion_mediated': res['proportion_mediated'],
            })

    df = pd.DataFrame(rows)

    if len(df) > 1:
        reject, p_fdr, _, _ = multipletests(df['indirect_p'].values,
                                             method=fdr_method)
        df['indirect_p_fdr'] = p_fdr
        df['significant_fdr'] = reject
    else:
        df['indirect_p_fdr'] = df['indirect_p']
        df['significant_fdr'] = df['indirect_p'] < 0.05

    return df


# =============================================================================
# NBS 分量级中介
# =============================================================================
def nbs_component_mediation(conn_matrices_g1, conn_matrices_g2,
                             subject_ids_g1, subject_ids_g2,
                             behavior_df, nbs_result,
                             behavior_cols=None, covariates_col=None,
                             alpha=0.05, n_bootstrap=5000, seed=42):
    """
    对 NBS 显著分量运行中介分析。

    每个显著分量 → 计算每个被试的分量内平均连接强度作为 M。

    Parameters
    ----------
    conn_matrices_g1 : ndarray (n1, n_ch, n_ch)
    conn_matrices_g2 : ndarray (n2, n_ch, n_ch)
    subject_ids_g1, subject_ids_g2 : array-like
        两组的被试 ID
    behavior_df : pd.DataFrame
        行为数据（含 'subject_id' 列）
    nbs_result : dict
        nbs_permutation_test 返回的结果
    behavior_cols : list of str, optional
        行为变量列名。默认所有数值列
    covariates_col : str or list of str, optional
        协变量列名（如 '年龄'）
    alpha : float
        NBS 显著性阈值
    n_bootstrap : int
    seed : int

    Returns
    -------
    list of dict
        每个显著分量的中介分析结果
    pd.DataFrame
        汇总表
    """
    # 找显著分量
    sig_indices = [i for i, p in enumerate(nbs_result['component_pvals'])
                   if p < alpha]
    if not sig_indices:
        print("无显著 NBS 分量，跳过中介分析")
        return [], pd.DataFrame()

    print(f"发现 {len(sig_indices)} 个显著分量，开始中介分析...")

    # 合并两组数据
    all_matrices = np.concatenate([conn_matrices_g1, conn_matrices_g2], axis=0)
    all_ids = list(subject_ids_g1) + list(subject_ids_g2)
    all_ids_str = [str(sid).zfill(3) if str(sid).isdigit() else str(sid)
                   for sid in all_ids]
    # 组别编码: g1=1, g2=0
    X_all = np.array([1] * len(subject_ids_g1) + [0] * len(subject_ids_g2))

    # 对齐行为数据
    behavior_indexed = behavior_df.set_index('subject_id')

    # 确定行为变量
    if behavior_cols is None:
        numeric_cols = ['150字', '1分钟阅读平均', '数字RAN均值', '物体RAN均值',
                        '阅读流畅性', '音位删除', '部首意识']
        behavior_cols = [c for c in numeric_cols if c in behavior_df.columns]

    all_results = []
    summary_rows = []

    for comp_idx in sig_indices:
        comp_edges = nbs_result['components'][comp_idx]
        comp_p = nbs_result['component_pvals'][comp_idx]
        comp_dir = (nbs_result.get('directions', [None] * len(nbs_result['components']))
                    [comp_idx])

        print(f"\n--- 分量 {comp_idx + 1}: {len(comp_edges)} 条边, "
              f"p={comp_p:.4f}, 方向={comp_dir} ---")

        # 计算每个被试的分量内平均连接强度
        M_values = []
        valid_idx = []
        valid_X = []

        for subj_i, (sid, mat) in enumerate(zip(all_ids_str, all_matrices)):
            if sid not in behavior_indexed.index:
                continue
            comp_mean = np.mean([mat[i, j] for i, j in comp_edges])
            M_values.append(comp_mean)
            valid_idx.append(sid)
            valid_X.append(X_all[subj_i])

        M_arr = np.array(M_values)
        X_arr = np.array(valid_X)

        # 协变量
        cov_arr = None
        if covariates_col is not None:
            cov_cols = [covariates_col] if isinstance(covariates_col, str) else covariates_col
            cov_arr = behavior_indexed.loc[valid_idx, cov_cols].values.astype(float)

        # 对每个行为变量运行中介
        comp_results = {'component_idx': comp_idx, 'n_edges': len(comp_edges),
                        'component_p': comp_p, 'direction': comp_dir,
                        'mediation_results': {}}

        for y_col in behavior_cols:
            Y_arr = behavior_indexed.loc[valid_idx, y_col].values.astype(float)

            res = mediation_bootstrap(X_arr, M_arr, Y_arr,
                                       covariates=cov_arr,
                                       n_bootstrap=n_bootstrap, seed=seed)
            comp_results['mediation_results'][y_col] = res

            sig_marker = ""
            ci_lo, ci_hi = res['indirect_ci']
            if ci_lo > 0 or ci_hi < 0:
                sig_marker = " *"

            print(f"  {y_col}: indirect={res['indirect']:.4f} "
                  f"CI=[{ci_lo:.4f}, {ci_hi:.4f}]{sig_marker}, "
                  f"a={res['a']:.3f}(p={res['a_p']:.3f}), "
                  f"b={res['b']:.3f}(p={res['b_p']:.3f})")

            summary_rows.append({
                'component': comp_idx + 1,
                'n_edges': len(comp_edges),
                'component_p': comp_p,
                'direction': comp_dir,
                'behavior': y_col,
                'n': res['n'],
                'a': res['a'], 'a_p': res['a_p'],
                'b': res['b'], 'b_p': res['b_p'],
                'c': res['c'], 'c_p': res['c_p'],
                'c_prime': res['c_prime'], 'c_prime_p': res['c_prime_p'],
                'indirect': res['indirect'],
                'indirect_ci_lo': ci_lo,
                'indirect_ci_hi': ci_hi,
                'indirect_p': res['indirect_p'],
                'proportion_mediated': res['proportion_mediated'],
            })

        all_results.append(comp_results)

    summary_df = pd.DataFrame(summary_rows)

    # FDR 校正
    if len(summary_df) > 1:
        reject, p_fdr, _, _ = multipletests(summary_df['indirect_p'].values,
                                             method='fdr_bh')
        summary_df['indirect_p_fdr'] = p_fdr
        summary_df['significant_fdr'] = reject
    elif len(summary_df) == 1:
        summary_df['indirect_p_fdr'] = summary_df['indirect_p']
        summary_df['significant_fdr'] = summary_df['indirect_p'] < 0.05

    return all_results, summary_df


# =============================================================================
# 可视化
# =============================================================================
def plot_mediation_diagram(result, labels=None, title=None, ax=None):
    """
    绘制经典中介路径图。

        M
       / \\
      a   b
     /     \\
    X --c'-→ Y
    (X --c--→ Y)

    Parameters
    ----------
    result : dict
        mediation_bootstrap 返回的结果
    labels : dict, optional
        {'X': '组别', 'M': '连接强度', 'Y': '阅读流畅性'}
    title : str, optional
    ax : matplotlib.axes.Axes, optional

    Returns
    -------
    fig, ax
    """
    if labels is None:
        labels = {'X': 'X (Group)', 'M': 'M (Connectivity)', 'Y': 'Y (Behavior)'}

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    else:
        fig = ax.figure

    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-1, 3.5)
    ax.axis('off')

    # 节点位置
    pos_x = (0, 0.5)
    pos_m = (2, 3)
    pos_y = (4, 0.5)

    # 节点框
    bbox_style = dict(boxstyle='round,pad=0.5', facecolor='lightblue',
                      edgecolor='black', linewidth=1.5)
    ax.text(*pos_x, labels['X'], ha='center', va='center', fontsize=13,
            fontweight='bold', bbox=bbox_style)
    ax.text(*pos_m, labels['M'], ha='center', va='center', fontsize=13,
            fontweight='bold', bbox=bbox_style)
    ax.text(*pos_y, labels['Y'], ha='center', va='center', fontsize=13,
            fontweight='bold', bbox=bbox_style)

    def _sig_stars(p):
        if p < 0.001:
            return '***'
        if p < 0.01:
            return '**'
        if p < 0.05:
            return '*'
        return ''

    def _format_coef(val, p):
        return f"{val:.3f}{_sig_stars(p)}"

    # Path a: X → M
    ax.annotate('', xy=(1.3, 2.6), xytext=(0.6, 1.0),
                arrowprops=dict(arrowstyle='->', lw=2, color='steelblue'))
    ax.text(0.6, 2.0, f"a = {_format_coef(result['a'], result['a_p'])}",
            ha='center', fontsize=11, color='steelblue')

    # Path b: M → Y
    ax.annotate('', xy=(3.4, 1.0), xytext=(2.7, 2.6),
                arrowprops=dict(arrowstyle='->', lw=2, color='steelblue'))
    ax.text(3.4, 2.0, f"b = {_format_coef(result['b'], result['b_p'])}",
            ha='center', fontsize=11, color='steelblue')

    # Path c': X → Y (直接效应)
    ax.annotate('', xy=(3.3, 0.5), xytext=(0.7, 0.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='gray'))
    ax.text(2.0, 0.15, f"c' = {_format_coef(result['c_prime'], result['c_prime_p'])}",
            ha='center', fontsize=11, color='gray')

    # Path c (总效应) - 显示在底部
    ax.text(2.0, -0.5,
            f"Total: c = {_format_coef(result['c'], result['c_p'])}",
            ha='center', fontsize=10, style='italic', color='dimgray')

    # 间接效应
    ci_lo, ci_hi = result['indirect_ci']
    sig = ' *' if (ci_lo > 0 or ci_hi < 0) else ''
    ax.text(2.0, -0.85,
            f"Indirect: ab = {result['indirect']:.3f} "
            f"[{ci_lo:.3f}, {ci_hi:.3f}]{sig}",
            ha='center', fontsize=10, fontweight='bold',
            color='darkred' if sig else 'dimgray')

    if title:
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)

    fig.tight_layout()
    return fig, ax


def plot_indirect_effects(summary_df, figsize=None):
    """
    间接效应 forest plot (含 CI 误差棒)。

    Parameters
    ----------
    summary_df : pd.DataFrame
        batch_mediation 或 nbs_component_mediation 返回的汇总表

    Returns
    -------
    fig, ax
    """
    if summary_df.empty:
        print("无数据可绘图")
        return None, None

    df = summary_df.copy()

    # 标签
    if 'component' in df.columns:
        df['label'] = df.apply(
            lambda r: f"C{int(r['component'])} → {r['behavior']}", axis=1)
    else:
        df['label'] = df.apply(
            lambda r: f"{r['mediator']} → {r['behavior']}", axis=1)

    n = len(df)
    if figsize is None:
        figsize = (8, max(3, n * 0.5))

    fig, ax = plt.subplots(figsize=figsize)

    y_pos = np.arange(n)
    indirect = df['indirect'].values
    ci_lo = df['indirect_ci_lo'].values
    ci_hi = df['indirect_ci_hi'].values

    # 颜色: 显著 = 红, 不显著 = 灰
    is_sig = (ci_lo > 0) | (ci_hi < 0)
    colors = ['darkred' if s else 'gray' for s in is_sig]

    xerr_lo = indirect - ci_lo
    xerr_hi = ci_hi - indirect

    ax.errorbar(indirect, y_pos, xerr=[xerr_lo, xerr_hi],
                fmt='o', color='black', ecolor=colors, elinewidth=2,
                capsize=4, markersize=6)

    # 着色点
    for i, (x, y, c) in enumerate(zip(indirect, y_pos, colors)):
        ax.plot(x, y, 'o', color=c, markersize=8, zorder=5)

    ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df['label'].values)
    ax.set_xlabel('Indirect Effect (a × b)')
    ax.set_title('Mediation Indirect Effects (95% CI)')
    ax.invert_yaxis()

    fig.tight_layout()
    return fig, ax


def summarize_mediation_results(results_list, labels=None):
    """
    将多个 mediation_bootstrap 结果汇总为 DataFrame。

    Parameters
    ----------
    results_list : list of dict
        每个 dict 需额外包含 'mediator_name' 和 'behavior_name' 键
    labels : dict, optional
        列名映射

    Returns
    -------
    pd.DataFrame
    """
    rows = []
    for res in results_list:
        ci_lo, ci_hi = res['indirect_ci']
        rows.append({
            'mediator': res.get('mediator_name', ''),
            'behavior': res.get('behavior_name', ''),
            'n': res['n'],
            'a': res['a'], 'a_p': res['a_p'],
            'b': res['b'], 'b_p': res['b_p'],
            'c': res['c'], 'c_p': res['c_p'],
            'c_prime': res['c_prime'], 'c_prime_p': res['c_prime_p'],
            'indirect': res['indirect'],
            'indirect_ci_lo': ci_lo,
            'indirect_ci_hi': ci_hi,
            'indirect_p': res['indirect_p'],
            'proportion_mediated': res['proportion_mediated'],
        })

    df = pd.DataFrame(rows)

    if len(df) > 1:
        reject, p_fdr, _, _ = multipletests(df['indirect_p'].values,
                                             method='fdr_bh')
        df['indirect_p_fdr'] = p_fdr
        df['significant_fdr'] = reject

    return df
