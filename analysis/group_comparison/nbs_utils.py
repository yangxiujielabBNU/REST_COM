"""
nbs_utils.py — Network-Based Statistic (NBS) 独立样本分析模块

基于 Zalesky et al. (2010) 的 NBS 方法，适配三组独立样本设计。
参考代码: d:\LYW\pre10_backup\analysis\connectivity\all2all_channel_connectivity.ipynb

关键改动 (相对参考代码):
- compute_t_matrix: paired t → independent samples t-test (scipy.stats.ttest_ind)
- _single_permutation: swap condition labels → shuffle group labels
- nbs_permutation_test: 适配不等样本量的独立样本设计
"""

from collections import defaultdict, deque

import numpy as np
from joblib import Parallel, delayed


# =============================================================================
# 核心统计函数
# =============================================================================
def compute_t_matrix_independent(group1, group2):
    """
    计算两组之间每条边的独立样本 t 统计量 (向量化 Welch's t)。

    Parameters
    ----------
    group1 : ndarray (n1, n_channels, n_channels)
    group2 : ndarray (n2, n_channels, n_channels)

    Returns
    -------
    t_matrix : ndarray (n_channels, n_channels)
        对称 t 值矩阵 (无向, 上三角 + 对称填充)
    """
    n1, n2 = group1.shape[0], group2.shape[0]

    mean1 = group1.mean(axis=0)
    mean2 = group2.mean(axis=0)
    var1 = group1.var(axis=0, ddof=1)
    var2 = group2.var(axis=0, ddof=1)

    denom = np.sqrt(var1 / n1 + var2 / n2)
    # 避免除零
    denom[denom == 0] = np.inf
    t_matrix = (mean1 - mean2) / denom

    # 清理 NaN (方差为零时)
    t_matrix = np.nan_to_num(t_matrix, nan=0.0)
    # 保持对称、对角线为零
    np.fill_diagonal(t_matrix, 0.0)

    return t_matrix


# =============================================================================
# 连通分量检测 (直接复用参考代码)
# =============================================================================
def find_nbs_components(t_matrix, threshold, tail='both'):
    """
    找到超阈值边的连通分量 (BFS)。

    Parameters
    ----------
    t_matrix : ndarray (n_channels, n_channels)
    threshold : float
        t 值阈值
    tail : str
        'both': |t| > threshold
        'positive': t > threshold
        'negative': t < -threshold
    """
    n_ch = t_matrix.shape[0]

    # Step 1: 阈值化
    supra_edges = []
    for i in range(n_ch):
        for j in range(i + 1, n_ch):
            t_val = t_matrix[i, j]
            if tail == 'both' and np.abs(t_val) > threshold:
                supra_edges.append((i, j))
            elif tail == 'positive' and t_val > threshold:
                supra_edges.append((i, j))
            elif tail == 'negative' and t_val < -threshold:
                supra_edges.append((i, j))

    if len(supra_edges) == 0:
        return [], []

    # Step 2: 构建边的邻接关系
    node_to_edges = defaultdict(list)
    for idx, (u, v) in enumerate(supra_edges):
        node_to_edges[u].append(idx)
        node_to_edges[v].append(idx)

    # Step 3: BFS 找连通分量
    visited = [False] * len(supra_edges)
    components = []

    for start in range(len(supra_edges)):
        if visited[start]:
            continue
        queue = deque([start])
        visited[start] = True
        component = []
        while queue:
            edge_idx = queue.popleft()
            component.append(supra_edges[edge_idx])
            u, v = supra_edges[edge_idx]
            for neighbor_idx in node_to_edges[u] + node_to_edges[v]:
                if not visited[neighbor_idx]:
                    visited[neighbor_idx] = True
                    queue.append(neighbor_idx)
        components.append(component)

    return components, supra_edges


def compute_component_stat(component, t_matrix, stat_type='size'):
    """计算连通分量的统计量。"""
    if stat_type == 'size':
        return len(component)
    elif stat_type == 'intensity':
        return sum(np.abs(t_matrix[i, j]) for i, j in component)
    else:
        raise ValueError(f"Unknown stat_type: {stat_type}")


# =============================================================================
# 置换检验 (适配独立样本)
# =============================================================================
def _single_permutation_independent(perm_idx, all_matrices, n1, threshold,
                                     tail, stat_type, rng_seed):
    """
    单次置换 (独立样本): 打乱组标签后计算最大分量统计量。

    当 tail='both' 时，分别计算正尾和负尾的最大统计量，返回 (max_pos, max_neg)。
    当 tail='positive'/'negative' 时，返回单个标量。
    """
    rng = np.random.default_rng(rng_seed + perm_idx)
    n_total = all_matrices.shape[0]

    perm_idx_arr = rng.permutation(n_total)
    g1_perm = all_matrices[perm_idx_arr[:n1]]
    g2_perm = all_matrices[perm_idx_arr[n1:]]

    t_matrix_perm = compute_t_matrix_independent(g1_perm, g2_perm)

    if tail == 'both':
        # 正负方向分别找分量、分别取 max
        comps_pos, _ = find_nbs_components(t_matrix_perm, threshold, 'positive')
        comps_neg, _ = find_nbs_components(t_matrix_perm, threshold, 'negative')
        max_pos = max((compute_component_stat(c, t_matrix_perm, stat_type)
                       for c in comps_pos), default=0)
        max_neg = max((compute_component_stat(c, t_matrix_perm, stat_type)
                       for c in comps_neg), default=0)
        return (max_pos, max_neg)
    else:
        components_perm, _ = find_nbs_components(t_matrix_perm, threshold, tail)
        if len(components_perm) > 0:
            return max(compute_component_stat(comp, t_matrix_perm, stat_type)
                       for comp in components_perm)
        return 0


def nbs_permutation_test(group1, group2, threshold, n_perms=5000,
                         tail='both', stat_type='size', seed=42, n_jobs=-1):
    """
    NBS 置换检验 (独立样本)。

    当 tail='both' 时，分别对正尾 (group1>group2) 和负尾 (group1<group2)
    找连通分量，各自与对应方向的零分布比较。结果中每个分量带 'direction' 标签。

    Returns
    -------
    dict : NBS 结果，含 components, component_stats, component_pvals,
           null_max_stats, t_matrix, threshold。
           tail='both' 时额外含 'directions' 列表 ('positive'/'negative')。
    """
    n1, n2 = group1.shape[0], group2.shape[0]
    print(f"  样本量: group1={n1}, group2={n2}")
    print(f"  计算真实数据的 t 矩阵...")

    t_matrix_real = compute_t_matrix_independent(group1, group2)
    all_matrices = np.concatenate([group1, group2], axis=0)

    if tail == 'both':
        return _nbs_both_tails(t_matrix_real, all_matrices, n1,
                               threshold, n_perms, stat_type, seed, n_jobs)
    else:
        return _nbs_single_tail(t_matrix_real, all_matrices, n1,
                                threshold, n_perms, tail, stat_type, seed, n_jobs)


def _nbs_single_tail(t_matrix_real, all_matrices, n1,
                     threshold, n_perms, tail, stat_type, seed, n_jobs):
    """单方向 NBS 置换检验。"""
    print(f"  寻找超阈值连通分量 (threshold={threshold}, tail={tail})...")
    components_real, supra_edges = find_nbs_components(
        t_matrix_real, threshold, tail
    )

    n_components = len(components_real)
    print(f"    超阈值边数: {len(supra_edges)}")
    print(f"    连通分量数: {n_components}")

    if n_components == 0:
        print("    未找到超阈值边，跳过置换检验")
        return {
            'components': [], 'component_stats': [],
            'component_pvals': [], 'null_max_stats': [],
            't_matrix': t_matrix_real, 'threshold': threshold,
        }

    real_stats = [compute_component_stat(comp, t_matrix_real, stat_type)
                  for comp in components_real]
    print(f"    分量统计量: {real_stats}")

    print(f"  执行置换检验 ({n_perms} 次, n_jobs={n_jobs})...")
    null_max_stats = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_single_permutation_independent)(
            i, all_matrices, n1, threshold, tail, stat_type, seed
        ) for i in range(n_perms)
    )
    null_max_stats = np.array(null_max_stats)

    p_values = [(np.sum(null_max_stats >= stat) + 1) / (n_perms + 1)
                for stat in real_stats]

    print(f"\n  NBS 结果 (tail={tail}):")
    for i, (comp, stat, p) in enumerate(zip(components_real, real_stats, p_values)):
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        n_nodes = len({n for e in comp for n in e})
        print(f"    分量{i+1}: {stat} 条边, {n_nodes} 个节点, p={p:.4f} {sig}")

    return {
        'components': components_real,
        'component_stats': real_stats,
        'component_pvals': p_values,
        'null_max_stats': null_max_stats,
        't_matrix': t_matrix_real,
        'threshold': threshold,
    }


def _nbs_both_tails(t_matrix_real, all_matrices, n1,
                    threshold, n_perms, stat_type, seed, n_jobs):
    """双侧 NBS: 正负方向分别找分量，各自与对应零分布比较。"""
    # 正尾: group1 > group2
    comps_pos, supra_pos = find_nbs_components(t_matrix_real, threshold, 'positive')
    # 负尾: group1 < group2
    comps_neg, supra_neg = find_nbs_components(t_matrix_real, threshold, 'negative')

    print(f"  正尾 (g1>g2): {len(supra_pos)} 超阈值边, {len(comps_pos)} 分量")
    print(f"  负尾 (g1<g2): {len(supra_neg)} 超阈值边, {len(comps_neg)} 分量")

    if len(comps_pos) == 0 and len(comps_neg) == 0:
        print("    两个方向均无超阈值边，跳过置换检验")
        return {
            'components': [], 'component_stats': [],
            'component_pvals': [], 'null_max_stats': [],
            'directions': [],
            't_matrix': t_matrix_real, 'threshold': threshold,
        }

    stats_pos = [compute_component_stat(c, t_matrix_real, stat_type) for c in comps_pos]
    stats_neg = [compute_component_stat(c, t_matrix_real, stat_type) for c in comps_neg]

    if stats_pos:
        print(f"    正尾分量统计量: {stats_pos}")
    if stats_neg:
        print(f"    负尾分量统计量: {stats_neg}")

    # 置换: 每次返回 (max_pos, max_neg)
    print(f"  执行置换检验 ({n_perms} 次, n_jobs={n_jobs})...")
    perm_results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_single_permutation_independent)(
            i, all_matrices, n1, threshold, 'both', stat_type, seed
        ) for i in range(n_perms)
    )
    perm_results = np.array(perm_results)  # (n_perms, 2)
    null_pos = perm_results[:, 0]
    null_neg = perm_results[:, 1]

    # 各自比较
    pvals_pos = [(np.sum(null_pos >= s) + 1) / (n_perms + 1) for s in stats_pos]
    pvals_neg = [(np.sum(null_neg >= s) + 1) / (n_perms + 1) for s in stats_neg]

    # 合并结果
    all_comps = comps_pos + comps_neg
    all_stats = stats_pos + stats_neg
    all_pvals = pvals_pos + pvals_neg
    all_dirs = ['positive'] * len(comps_pos) + ['negative'] * len(comps_neg)

    print(f"\n  NBS 结果 (双侧):")
    for i, (comp, stat, p, d) in enumerate(zip(all_comps, all_stats, all_pvals, all_dirs)):
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        n_nodes = len({n for e in comp for n in e})
        arrow = "g1>g2" if d == 'positive' else "g1<g2"
        print(f"    分量{i+1} ({arrow}): {stat} 条边, {n_nodes} 个节点, p={p:.4f} {sig}")

    return {
        'components': all_comps,
        'component_stats': all_stats,
        'component_pvals': all_pvals,
        'null_max_stats': perm_results,
        'directions': all_dirs,
        't_matrix': t_matrix_real,
        'threshold': threshold,
    }


# =============================================================================
# 结果提取与可视化辅助 (复用参考代码)
# =============================================================================
def get_significant_edges(nbs_results, alpha=0.05, channels=None):
    """提取显著分量中的边。始终返回 index 列表，可选附带通道名。"""
    sig_edges = []
    for comp, p in zip(nbs_results['components'], nbs_results['component_pvals']):
        if p < alpha:
            sig_edges.extend(comp)

    if channels is not None:
        sig_edges_named = [(channels[i], channels[j]) for i, j in sig_edges]
        return sig_edges, sig_edges_named
    return sig_edges, []


def create_nbs_matrix(nbs_results, n_channels, value_type='t'):
    """创建可视化矩阵 (仅显著边有值)。"""
    matrix = np.zeros((n_channels, n_channels))
    sig_edges, _ = get_significant_edges(nbs_results)

    for i, j in sig_edges:
        if value_type == 't':
            val = nbs_results['t_matrix'][i, j]
        else:
            val = 1
        matrix[i, j] = val
        matrix[j, i] = val

    return matrix


# =============================================================================
# 批量运行: 三组两两比较 × 多频段
# =============================================================================
def run_all_pairwise_nbs(group_matrices, channel_names, pairs, bands,
                         threshold=2.0, n_perms=5000, stat_type='size',
                         tail='both', seed=42, n_jobs=-1):
    """
    对所有组对和频段运行 NBS。

    Parameters
    ----------
    group_matrices : dict
        {group_name: {band_name: ndarray (n_subjects, n_ch, n_ch)}}
    channel_names : list
        通道名列表
    pairs : list of tuple
        组对列表, 如 [('adhd', 'com'), ('adhd', 'td'), ('com', 'td')]
    bands : list of str
        频段列表, 如 ['delta', 'theta', 'alpha', 'beta', 'gamma']
    threshold, n_perms, stat_type, tail, seed, n_jobs : NBS 参数

    Returns
    -------
    all_results : dict
        {(g1, g2, band): nbs_result_dict}
    """
    all_results = {}
    total = len(pairs) * len(bands)
    count = 0

    for g1, g2 in pairs:
        for band in bands:
            count += 1
            print(f"\n{'=' * 60}")
            print(f"[{count}/{total}] {g1} vs {g2} — {band}")
            print('=' * 60)

            mat1 = group_matrices[g1][band]
            mat2 = group_matrices[g2][band]

            result = nbs_permutation_test(
                mat1, mat2, threshold=threshold, n_perms=n_perms,
                tail=tail, stat_type=stat_type, seed=seed, n_jobs=n_jobs,
            )
            result['pair'] = (g1, g2)
            result['freq_band'] = band
            result['channel_names'] = channel_names

            all_results[(g1, g2, band)] = result

    return all_results


def summarize_nbs_results(all_results, group_labels=None):
    """
    汇总所有 NBS 结果为 DataFrame。

    当结果包含 'directions' 字段时（tail='both'），会在输出中添加 'direction' 列。

    Parameters
    ----------
    all_results : dict
        run_all_pairwise_nbs 的返回值
    group_labels : dict, optional
        {group: label} 映射

    Returns
    -------
    pd.DataFrame
    """
    import pandas as pd

    rows = []
    for (g1, g2, band), res in all_results.items():
        pair_label = f"{g1} vs {g2}"
        if group_labels:
            pair_label = f"{group_labels.get(g1, g1)} vs {group_labels.get(g2, g2)}"

        has_directions = 'directions' in res and res['directions']

        if len(res['components']) == 0:
            row = {
                'pair': f"{g1}_vs_{g2}", 'pair_label': pair_label,
                'freq_band': band, 'component_id': 0,
                'n_edges': 0, 'n_nodes': 0,
                'stat': 0, 'p_value': 1.0, 'significant': False,
            }
            if has_directions:
                row['direction'] = None
            rows.append(row)
        else:
            directions = res.get('directions', [None] * len(res['components']))
            for i, (comp, stat, p, direction) in enumerate(zip(
                res['components'], res['component_stats'], res['component_pvals'], directions
            )):
                n_nodes = len({n for e in comp for n in e})
                row = {
                    'pair': f"{g1}_vs_{g2}", 'pair_label': pair_label,
                    'freq_band': band, 'component_id': i + 1,
                    'n_edges': len(comp), 'n_nodes': n_nodes,
                    'stat': stat, 'p_value': p, 'significant': p < 0.05,
                }
                if has_directions:
                    row['direction'] = direction
                rows.append(row)

    return pd.DataFrame(rows)


def extract_all_significant_edges(all_results, alpha=0.05):
    """
    从所有 NBS 结果中提取显著边，返回 DataFrame。

    Returns
    -------
    pd.DataFrame with columns: pair, freq_band, ch1, ch2, t_value, component_p
    """
    import pandas as pd

    rows = []
    for (g1, g2, band), res in all_results.items():
        channels = res.get('channel_names', [])
        for comp, p in zip(res['components'], res['component_pvals']):
            if p >= alpha:
                continue
            for i, j in comp:
                ch1 = channels[i] if channels else str(i)
                ch2 = channels[j] if channels else str(j)
                rows.append({
                    'pair': f"{g1}_vs_{g2}",
                    'freq_band': band,
                    'ch1': ch1, 'ch2': ch2,
                    't_value': res['t_matrix'][i, j],
                    'component_p': p,
                })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values('t_value', key=abs, ascending=False)
    return df


# =============================================================================
# 多阈值探索
# =============================================================================
# 10-20 系统脑区映射
CHANNEL_REGION_MAP = {
    # 前额叶
    'Fp1': 'Prefrontal', 'Fp2': 'Prefrontal',
    'AF3': 'Prefrontal', 'AF4': 'Prefrontal',
    'AF7': 'Prefrontal', 'AF8': 'Prefrontal', 'AFz': 'Prefrontal',
    # 额叶
    'F1': 'Frontal', 'F2': 'Frontal', 'F3': 'Frontal', 'F4': 'Frontal',
    'F5': 'Frontal', 'F6': 'Frontal', 'F7': 'Frontal', 'F8': 'Frontal',
    'F9': 'Frontal', 'Fz': 'Frontal',
    # 额-中央
    'FC1': 'Frontal-Central', 'FC2': 'Frontal-Central',
    'FC3': 'Frontal-Central', 'FC4': 'Frontal-Central',
    'FC5': 'Frontal-Central', 'FC6': 'Frontal-Central',
    'FCz': 'Frontal-Central',
    'FT7': 'Frontal-Central', 'FT8': 'Frontal-Central',
    # 中央
    'C1': 'Central', 'C2': 'Central', 'C3': 'Central', 'C4': 'Central',
    'C5': 'Central', 'C6': 'Central', 'Cz': 'Central',
    # 颞叶
    'T7': 'Temporal', 'T8': 'Temporal',
    'TP7': 'Temporal', 'TP8': 'Temporal',
    # 中央-顶叶
    'CP1': 'Central-Parietal', 'CP2': 'Central-Parietal',
    'CP3': 'Central-Parietal', 'CP4': 'Central-Parietal',
    'CP5': 'Central-Parietal', 'CP6': 'Central-Parietal',
    'CPz': 'Central-Parietal',
    # 顶叶
    'P1': 'Parietal', 'P2': 'Parietal', 'P3': 'Parietal', 'P4': 'Parietal',
    'P5': 'Parietal', 'P6': 'Parietal', 'P7': 'Parietal', 'P8': 'Parietal',
    'Pz': 'Parietal',
    # 顶-枕
    'PO3': 'Parietal-Occipital', 'PO4': 'Parietal-Occipital',
    'PO7': 'Parietal-Occipital', 'PO8': 'Parietal-Occipital',
    'POz': 'Parietal-Occipital',
    # 枕叶
    'O1': 'Occipital', 'O2': 'Occipital', 'Oz': 'Occipital',
}


def get_channel_region(ch_name):
    """返回通道所属脑区，未知则返回 'Other'。"""
    return CHANNEL_REGION_MAP.get(ch_name, 'Other')


def multi_threshold_explore(group1, group2, thresholds=(2.0, 2.5, 3.0, 3.5),
                            channel_names=None, tail='both'):
    """
    多阈值探索: 只计算一次 t 矩阵, 用不同阈值切割观察边的分布。
    不做置换检验 (快速探索用)。

    当 tail='both' 时，分别统计正负方向的边数和分量。

    Returns
    -------
    dict : {threshold: {'n_supra_pos': int, 'n_supra_neg': int,
                        'n_components_pos': int, 'n_components_neg': int,
                        'largest_component_pos': int, 'largest_component_neg': int,
                        'top_edges_pos': list, 'top_edges_neg': list, ...}}
           当 tail='positive'/'negative' 时，只有单方向的统计。
    t_matrix : ndarray — 真实 t 值矩阵
    """
    t_matrix = compute_t_matrix_independent(group1, group2)
    n_ch = t_matrix.shape[0]
    channels = channel_names or [str(i) for i in range(n_ch)]

    results = {}
    for thr in thresholds:
        if tail == 'both':
            # 正负方向分别统计
            comps_pos, edges_pos = find_nbs_components(t_matrix, thr, 'positive')
            comps_neg, edges_neg = find_nbs_components(t_matrix, thr, 'negative')

            result = {
                'n_supra_pos': len(edges_pos),
                'n_supra_neg': len(edges_neg),
                'n_components_pos': len(comps_pos),
                'n_components_neg': len(comps_neg),
                'largest_component_pos': max((len(c) for c in comps_pos), default=0),
                'largest_component_neg': max((len(c) for c in comps_neg), default=0),
            }

            # 正向边详情
            edge_list_pos = []
            for i, j in edges_pos:
                edge_list_pos.append({
                    'ch1': channels[i], 'ch2': channels[j],
                    'ch1_idx': i, 'ch2_idx': j,
                    't_value': t_matrix[i, j],
                })
            edge_list_pos.sort(key=lambda x: x['t_value'], reverse=True)

            # 负向边详情
            edge_list_neg = []
            for i, j in edges_neg:
                edge_list_neg.append({
                    'ch1': channels[i], 'ch2': channels[j],
                    'ch1_idx': i, 'ch2_idx': j,
                    't_value': t_matrix[i, j],
                })
            edge_list_neg.sort(key=lambda x: x['t_value'])  # 负向按升序

            result['top_edges_pos'] = edge_list_pos[:20]
            result['top_edges_neg'] = edge_list_neg[:20]
            result['all_edges_pos'] = edge_list_pos
            result['all_edges_neg'] = edge_list_neg

            # 脑区统计（正向）
            region_counts_pos = defaultdict(int)
            for e in edge_list_pos:
                r1 = get_channel_region(e['ch1'])
                r2 = get_channel_region(e['ch2'])
                region_counts_pos[r1] += 1
                region_counts_pos[r2] += 1
            result['region_counts_pos'] = dict(sorted(region_counts_pos.items(),
                                                      key=lambda x: x[1], reverse=True))

            # 脑区统计（负向）
            region_counts_neg = defaultdict(int)
            for e in edge_list_neg:
                r1 = get_channel_region(e['ch1'])
                r2 = get_channel_region(e['ch2'])
                region_counts_neg[r1] += 1
                region_counts_neg[r2] += 1
            result['region_counts_neg'] = dict(sorted(region_counts_neg.items(),
                                                      key=lambda x: x[1], reverse=True))

        else:
            # 单方向统计（保持原逻辑）
            components, supra_edges = find_nbs_components(t_matrix, thr, tail)

            edge_list = []
            for i, j in supra_edges:
                edge_list.append({
                    'ch1': channels[i], 'ch2': channels[j],
                    'ch1_idx': i, 'ch2_idx': j,
                    't_value': t_matrix[i, j],
                })
            edge_list.sort(key=lambda x: abs(x['t_value']), reverse=True)

            region_counts = defaultdict(int)
            for e in edge_list:
                r1 = get_channel_region(e['ch1'])
                r2 = get_channel_region(e['ch2'])
                region_counts[r1] += 1
                region_counts[r2] += 1
            region_counts = dict(sorted(region_counts.items(),
                                        key=lambda x: x[1], reverse=True))

            region_pair_counts = defaultdict(int)
            for e in edge_list:
                r1 = get_channel_region(e['ch1'])
                r2 = get_channel_region(e['ch2'])
                key = tuple(sorted([r1, r2]))
                region_pair_counts[key] += 1
            region_pair_counts = dict(sorted(region_pair_counts.items(),
                                             key=lambda x: x[1], reverse=True))

            result = {
                'n_supra': len(supra_edges),
                'n_components': len(components),
                'largest_component': max((len(c) for c in components), default=0),
                'top_edges': edge_list[:20],
                'region_counts': region_counts,
                'region_pair_counts': region_pair_counts,
                'all_edges': edge_list,
            }

        results[thr] = result

    return results, t_matrix


def summarize_region_involvement(edges_list, channel_names=None):
    """
    统计一组边中每个脑区涉及的边数 (度数)。

    Parameters
    ----------
    edges_list : list of (i, j) 或 list of dict with ch1/ch2

    Returns
    -------
    pd.DataFrame : columns=[region, n_edges, pct]
    """
    import pandas as pd

    region_counts = defaultdict(int)
    total = 0
    for e in edges_list:
        if isinstance(e, dict):
            r1 = get_channel_region(e['ch1'])
            r2 = get_channel_region(e['ch2'])
        elif isinstance(e, (tuple, list)) and channel_names:
            r1 = get_channel_region(channel_names[e[0]])
            r2 = get_channel_region(channel_names[e[1]])
        else:
            continue
        region_counts[r1] += 1
        region_counts[r2] += 1
        total += 1

    rows = [{'region': r, 'n_edges': c, 'pct': c / (total * 2) * 100 if total > 0 else 0}
            for r, c in region_counts.items()]
    df = pd.DataFrame(rows).sort_values('n_edges', ascending=False).reset_index(drop=True)
    return df
