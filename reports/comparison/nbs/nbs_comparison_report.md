# NBS 连接性组间统计比较报告

**生成时间:** 2026-03-02
**数据来源:** `reports/comparison/nbs/nbs_omnibus_summary.csv`, `nbs_significant_edges.csv`
**分析脚本:** `analysis/group_comparison/nbs_stats.ipynb`
**统计方法:** Network-Based Statistic (NBS, Zalesky et al. 2010)
**目的:** 汇总三组被试在五个频段的 wPLI 连接性边级别差异

---

## 1. 研究设计

| 组别 | 标签 | 样本量 |
|------|------|--------|
| adhd | ADHD单纯组 | 12 |
| com | ADHD共患阅读困难组 | 14 |
| td | 正常发展组 | 8 |

## 2. NBS 参数

| 参数 | 值 |
|------|-----|
| 连接性指标 | wPLI |
| t 值阈值 | 3.0 |
| 置换次数 | 5000 |
| 统计量类型 | size (边数) |
| 检验方向 | both (正负方向独立零分布) |
| 频段 | delta, theta, alpha, beta, gamma |
| 随机种子 | 42 |

## 3. 显著结果

共 15 次 NBS 比较 (3 组对 × 5 频段)，**2 个显著子网络**，均在 **gamma 频段**，方向均为 **positive (临床组 > TD)**。

| 比较 | 频段 | 边数 | 节点数 | p 值 | 方向 |
|------|------|------|--------|------|------|
| ADHD vs TD | gamma | 42 | 32 | **0.034** | positive (ADHD > TD) |
| COM vs TD | gamma | 61 | 43 | **0.032** | positive (COM > TD) |

ADHD vs COM 在所有频段均无显著差异 (最小 p = 0.137, beta)。

### 非显著频段概览

| 频段 | adhd vs com | adhd vs td | com vs td |
|------|-------------|------------|-----------|
| delta | p > 0.60 | p > 0.60 | p > 0.40 |
| theta | p > 0.17 | p > 0.13 | p > 0.57 |
| alpha | p > 0.40 | p > 0.28 | p > 0.61 |
| beta | p > 0.14 | p > 0.23 | p > 0.20 |

注: theta adhd vs td 有 11 边分量 (p=0.132)，虽未达显著但值得关注。

## 4. 显著边空间分布

103 条显著边 (adhd_vs_td: 42, com_vs_td: 61)，全部 gamma 频段。

### Top 10 边 (按 t 值)

| 比较 | ch1 | ch2 | t 值 |
|------|-----|-----|------|
| adhd_vs_td | F1 | CP4 | 4.74 |
| com_vs_td | CP5 | PO4 | 4.58 |
| adhd_vs_td | CP5 | PO8 | 4.56 |
| adhd_vs_td | CP5 | P8 | 4.30 |
| adhd_vs_td | CP6 | F1 | 4.26 |
| com_vs_td | FC5 | F8 | 4.16 |
| com_vs_td | C4 | F1 | 4.13 |
| com_vs_td | CP5 | P2 | 4.09 |
| adhd_vs_td | CP5 | P6 | 4.02 |
| com_vs_td | TP7 | CP4 | 3.96 |

*完整 103 条边见 `nbs_significant_edges.csv`*

### 核心节点

- **CP5, CP6, CP3, CP4** — Central-Parietal，出现频率最高
- **P4, P6, P8, PO4, PO8** — Parietal-Occipital
- **F1, F4** — Frontal (与 CP 区形成长程连接)
- **TP7, TP8** — Temporal-Parietal

两个临床组 vs TD 的子网络有大量重叠边 (CP5↔PO8, CP6↔CP3, C4↔F1, FC1↔FC6 等)。

---

## 5. 结论

1. **Gamma Central-Parietal 子网络异常**: ADHD 和 COM 组的 gamma wPLI 连接性均显著高于 TD 组，涉及 32-43 个节点的大范围子网络
2. **ADHD 亚组差异极小**: 单纯组与共患阅读困难组在所有频段均无显著差异，提示共患阅读困难对静息态 wPLI 模式影响有限
3. **频段特异性**: 差异仅在 gamma 频段达显著，低频段未发现显著子网络
4. **方向一致**: 所有显著差异均为临床组 > TD (gamma 连接性增强)

---

*报告由 REST EEG Pipeline 生成，详见 `analysis/group_comparison/nbs_stats.ipynb`*
