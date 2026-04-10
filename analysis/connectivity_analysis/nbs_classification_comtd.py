"""
NBS显著边Gamma频段 - 分组分类分析
用NBS找到的显著边特征预测被试分组 (ADHD vs COM vs TD)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                              roc_auc_score, confusion_matrix)
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_classif
from scipy.stats import spearmanr
from statsmodels.stats.multitest import fdrcorrection
import warnings
warnings.filterwarnings('ignore')


def create_model(use_pca, n_components):
    """创建不含标准化步骤的模型（标准化在CV循环内完成）"""
    return Pipeline([
        ('scaler', StandardScaler()),
        ('pca' if use_pca else 'x',
         PCA(n_components=n_components) if use_pca else 'passthrough'),
        ('svc', LinearSVC(max_iter=5000, class_weight='balanced'))
    ])


# =============================================================================
# 1. 配置路径
# =============================================================================
BASE_PATH = Path('/Users/gaozhenning/Desktop/CodeData/REST_COM-main')

CONN_PATHS = {
    'adhd': BASE_PATH / 'analysis/connectivity_analysis/adhd/results',
    'com': BASE_PATH / 'analysis/connectivity_analysis/com/results',
    'td': BASE_PATH / 'analysis/connectivity_analysis/td/results',
}

NBS_EDGES_PATH = BASE_PATH / 'reports/comparison/nbs/nbs_significant_edges.csv'
BEHAVIOR_PATH = BASE_PATH / '行为数据spss.csv'

# =============================================================================
# 2. 加载数据并建立被试ID映射
# =============================================================================
print("=" * 60)
print("加载数据")
print("=" * 60)

# 加载行为数据，建立 被试ID(去掉BZK) -> 分组 的映射
behav_df = pd.read_csv(BEHAVIOR_PATH)
behav_df = behav_df.dropna(subset=['被试编号', '分组'])

# 行为数据ID格式: BZK018 -> 映射到 '018'
subject_to_group = {}
for _, row in behav_df.iterrows():
    subj_id = str(row['被试编号'])[3:]  # 去掉BZK
    subject_to_group[subj_id] = row['分组']

print(f"行为数据: {len(behav_df)} 人")
print(f"分组分布:\n{behav_df['分组'].value_counts()}")

# 加载连接矩阵，建立 被试ID -> 组别 的映射
conn_subject_to_group = {}
for group, path in CONN_PATHS.items():
    npz = np.load(path / 'connectivity_results.npz', allow_pickle=True)
    for subj_id in npz['subject_ids']:
        conn_subject_to_group[str(subj_id)] = group

# 找匹配的被试
matched_subjects = set(subject_to_group.keys()) & set(conn_subject_to_group.keys())
print(f"\n匹配的被试数: {len(matched_subjects)}")

# 加载NBS显著边
nbs_edges_df = pd.read_csv(NBS_EDGES_PATH)
print(f"NBS显著边总数: {len(nbs_edges_df)}")

# =============================================================================
# 3. 分析配置
# =============================================================================
# 'three_class' - 三分类 ADHD vs COM vs TD
# 'dyslexia' - 阅读困难(COM) vs 非阅读困难(ADHD+TD)
# 'adhd_td' - ADHD vs TD
# 'com_td' - COM(阅读困难) vs TD (正常) 二分类
ANALYSIS_CONFIG = {
    'comparison': 'com_vs_td',  # 用com_vs_td的显著边
    'classification': 'com_td',  # COM(阅读困难) vs TD(正常)
    'n_components': 6,
    'use_pca': True,
}

print("\n" + "=" * 60)
print("分析配置")
print("=" * 60)
for k, v in ANALYSIS_CONFIG.items():
    print(f"  {k}: {v}")

# =============================================================================
# 4. 提取NBS显著边特征
# =============================================================================
print("\n" + "=" * 60)
print("提取NBS显著边特征")
print("=" * 60)

# 筛选特定比较和频段的显著边
sig_edges = nbs_edges_df[(nbs_edges_df['pair'] == ANALYSIS_CONFIG['comparison']) &
                          (nbs_edges_df['freq_band'] == 'gamma')]
print(f"筛选后 ({ANALYSIS_CONFIG['comparison']} vs gamma): {len(sig_edges)} 条显著边")

# 获取通道名称和索引映射
npz_sample = np.load(CONN_PATHS['adhd'] / 'connectivity_results.npz', allow_pickle=True)
channel_names = list(npz_sample['channel_names'])
ch2idx = {ch: i for i, ch in enumerate(channel_names)}

# 提取所有组别的特征
X_list, y_list, subj_ids_list = [], [], []
group_labels_map = {'adhd': 0, 'com': 1, 'td': 2}

for group, path in CONN_PATHS.items():
    npz_path = path / 'connectivity_results.npz'
    data = np.load(npz_path, allow_pickle=True)
    matrices = data['gamma']  # gamma频段

    for subj_idx, subj_id in enumerate(data['subject_ids']):
        subj_id_str = str(subj_id)

        # 只保留匹配的被试
        if subj_id_str not in matched_subjects:
            continue

        mat = matrices[subj_idx]
        features = []
        for _, row in sig_edges.iterrows():
            i = ch2idx.get(row['ch1'])
            j = ch2idx.get(row['ch2'])
            if i is not None and j is not None:
                features.append(mat[i, j])
            else:
                features.append(0)

        X_list.append(features)
        y_list.append(group_labels_map[group])
        subj_ids_list.append(subj_id_str)

X = np.array(X_list)
y = np.array(y_list)

print(f"\n特征矩阵形状: {X.shape}")
print(f"标签分布: ADHD={np.sum(y==0)}, COM={np.sum(y==1)}, TD={np.sum(y==2)}")

# =============================================================================
# 5. 根据分类类型设置标签
# =============================================================================
if ANALYSIS_CONFIG['classification'] == 'three_class':
    y_encoded = y.copy()
    class_names = ['ADHD', 'COM', 'TD']
    n_classes = 3

elif ANALYSIS_CONFIG['classification'] == 'dyslexia':
    # COM = 阅读困难(1), ADHD+TD = 非阅读困难(0)
    y_encoded = np.where(y == 1, 1, 0)
    class_names = ['非阅读困难', '阅读困难']
    n_classes = 2

elif ANALYSIS_CONFIG['classification'] == 'adhd_td':
    # ADHD vs TD (排除COM)
    mask = (y == 0) | (y == 2)
    subj_ids_array = np.array(subj_ids_list)
    X = X[mask]
    y_encoded = y[mask]
    subj_ids_list = list(subj_ids_array[mask])  # 更新被试ID列表
    y_encoded = np.where(y_encoded == 0, 0, 1)
    class_names = ['ADHD', 'TD']
    n_classes = 2

elif ANALYSIS_CONFIG['classification'] == 'com_td':
    # COM(阅读困难) vs TD (排除ADHD)
    mask = (y == 1) | (y == 2)
    subj_ids_array = np.array(subj_ids_list)
    X = X[mask]
    y_encoded = y[mask]
    subj_ids_list = list(subj_ids_array[mask])  # 更新被试ID列表
    y_encoded = np.where(y_encoded == 1, 0, 1)  # com=0, td=1
    class_names = ['阅读困难(COM)', '正常(TD)']
    n_classes = 2

print(f"\n分类: {class_names}")
print(f"类别分布: {np.bincount(y_encoded)}")
print(f"被试数量: {len(y_encoded)}")

# =============================================================================
# 6. 自动调整PCA成分数
# =============================================================================
n_samples = len(y_encoded)
n_features = X.shape[1]
max_pca_components = min(n_samples - 1, n_features, 20)

if ANALYSIS_CONFIG['use_pca']:
    actual_components = min(ANALYSIS_CONFIG['n_components'], max_pca_components)
    # 确保PCA成分数 <= min(样本数-1, 特征数)
    if actual_components < 1:
        actual_components = max_pca_components
    print(f"PCA成分数: {actual_components} (样本数={n_samples}, 特征数={n_features})")
else:
    actual_components = n_features

# =============================================================================
# 7. 模型训练与交叉验证
# =============================================================================
print("\n" + "=" * 60)
print("模型训练与交叉验证")
print("=" * 60)

n_folds = min(5, n_samples - 1)
if n_folds < 2:
    n_folds = n_samples  # Leave-one-out
    print(f"使用 Leave-One-Out 交叉验证 (共{n_folds}折)")
else:
    print(f"使用 {n_folds}-折交叉验证")

skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA降维
if ANALYSIS_CONFIG['use_pca'] and actual_components < n_features:
    pca = PCA(n_components=actual_components)
    X_final = pca.fit_transform(X_scaled)
    print(f"PCA降维: {n_features} → {actual_components}")
    print(f"保留方差: {pca.explained_variance_ratio_.sum():.2%}")
else:
    X_final = X_scaled
    print(f"不使用PCA，特征数: {n_features}")

# 测试模型
models = {
    'LinearSVC': Pipeline([
        ('scaler', StandardScaler()),
        ('pca' if ANALYSIS_CONFIG['use_pca'] else 'x', PCA(n_components=actual_components) if ANALYSIS_CONFIG['use_pca'] else 'passthrough'),
        ('svc', LinearSVC(max_iter=5000, class_weight='balanced'))
    ]),
    'LogisticRegression': Pipeline([
        ('scaler', StandardScaler()),
        ('pca' if ANALYSIS_CONFIG['use_pca'] else 'x', PCA(n_components=actual_components) if ANALYSIS_CONFIG['use_pca'] else 'passthrough'),
        ('lr', LogisticRegression(max_iter=1000, class_weight='balanced', solver='lbfgs'))
    ]),
}

results_summary = []
for model_name, model_name_key in models.items():
    print(f"\n--- {model_name} ---")

    accs, bal_accs = [], []
    all_y_true, all_y_pred = [], []

    for train_idx, test_idx in skf.split(X, y_encoded):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

        # ★ 在训练集内进行标准化（避免数据泄露）
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # ★ PCA降维也在训练集内拟合
        if ANALYSIS_CONFIG['use_pca']:
            pca = PCA(n_components=actual_components)
            X_train_final = pca.fit_transform(X_train_scaled)
            X_test_final = pca.transform(X_test_scaled)
        else:
            X_train_final = X_train_scaled
            X_test_final = X_test_scaled

        # 创建模型（不再内部标准化，传入已处理的数据）
        model = create_model(ANALYSIS_CONFIG['use_pca'], actual_components)
        model.fit(X_train_final, y_train)
        y_pred = model.predict(X_test_final)

        accs.append(accuracy_score(y_test, y_pred))
        bal_accs.append(balanced_accuracy_score(y_test, y_pred))
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

    print(f"  Accuracy: {np.mean(accs):.3f} (±{np.std(accs):.3f})")
    print(f"  Balanced Accuracy: {np.mean(bal_accs):.3f} (±{np.std(bal_accs):.3f})")
    print(f"  Confusion Matrix:\n{confusion_matrix(all_y_true, all_y_pred)}")

    results_summary.append({
        'model': model_name,
        'accuracy': np.mean(accs),
        'balanced_accuracy': np.mean(bal_accs),
        'std': np.std(accs),
    })

# =============================================================================
# 8. 置换检验
# =============================================================================
print("\n" + "=" * 60)
print("置换检验 (2000次)")
print("=" * 60)

n_perm = 2000
perm_accs = []

for i in range(n_perm):
    y_perm = np.random.permutation(y_encoded)
    fold_accs = []
    for train_idx, test_idx in skf.split(X, y_perm):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_perm[train_idx], y_perm[test_idx]

        model = Pipeline([
            ('scaler', StandardScaler()),
            ('pca' if ANALYSIS_CONFIG['use_pca'] else 'x', PCA(n_components=actual_components) if ANALYSIS_CONFIG['use_pca'] else 'passthrough'),
            ('svc', LinearSVC(max_iter=5000, class_weight='balanced'))
        ])
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        fold_accs.append(accuracy_score(y_test, y_pred))

    perm_accs.append(np.mean(fold_accs))

perm_accs = np.array(perm_accs)
best_acc = results_summary[0]['accuracy']
perm_pvalue = (np.sum(perm_accs >= best_acc) + 1) / (n_perm + 1)

print(f"真实准确率: {best_acc:.3f}")
print(f"置换准确率: {np.mean(perm_accs):.3f} (±{np.std(perm_accs):.3f})")
print(f"Permutation p-value: {perm_pvalue:.4f}")

if perm_pvalue < 0.05:
    print("✓ 模型分类能力显著 (p<0.05)")
else:
    print("✗ 模型分类能力不显著 (p>=0.05)")

# =============================================================================
# 9. 特征重要性分析
# =============================================================================
print("\n" + "=" * 60)
print("特征重要性分析 (F-score)")
print("=" * 60)

f_scores, p_vals = f_classif(X_scaled, y_encoded)
fdr_pvals = fdrcorrection(p_vals)[1]

feature_importance = pd.DataFrame({
    'ch1': sig_edges['ch1'].values,
    'ch2': sig_edges['ch2'].values,
    'edge': [f"{sig_edges['ch1'].iloc[i]}-{sig_edges['ch2'].iloc[i]}" for i in range(len(sig_edges))],
    't_value': sig_edges['t_value'].values,
    'f_score': f_scores,
    'p_value': p_vals,
    'fdr_p': fdr_pvals,
})

feature_importance = feature_importance.sort_values('f_score', ascending=False)
print("\nTop 15 特征 (按F-score排序):")
print(feature_importance[['edge', 't_value', 'f_score', 'p_value', 'fdr_p']].head(15).to_string())

n_sig_features = np.sum(fdr_pvals < 0.05)
print(f"\nFDR校正后显著特征数: {n_sig_features}")

# =============================================================================
# 10. 行为相关分析
# =============================================================================
print("\n" + "=" * 60)
print("NBS边特征与行为测量相关")
print("=" * 60)

# 获取行为数据 - 针对dyslexia分类（所有33个被试）
adhd_td_behav = behav_df.copy()
adhd_td_behav['subj_id'] = adhd_td_behav['被试编号'].str.replace('BZK', '', regex=False)
# 只保留在subj_ids_list中的被试
adhd_td_behav = adhd_td_behav[adhd_td_behav['subj_id'].isin(subj_ids_list)]

if len(adhd_td_behav) == 0:
    print("警告: 无法匹配行为数据")
else:
    # 重新排列行为数据顺序以匹配X的顺序
    adhd_td_behav = adhd_td_behav.set_index('subj_id').loc[subj_ids_list].reset_index()
    
    behavior_cols = ['150字', '1分钟阅读平均', '数字RAN均值', '物体RAN均值',
                     '阅读流畅性', '音位删除', '部首意识']

    for col in behavior_cols:
        adhd_td_behav[col] = pd.to_numeric(adhd_td_behav[col], errors='coerce')
        adhd_td_behav.loc[adhd_td_behav[col] == -999, col] = np.nan

    sig_edge_mean = X_scaled.mean(axis=1)

    print(f"\n特征均值与行为测量的Spearman相关 (n={len(sig_edge_mean)}):")
    behavior_correlations = []
    for col in behavior_cols:
        valid_mask = ~np.isnan(adhd_td_behav[col].values)
        if np.sum(valid_mask) < 5:
            continue
        r, p = spearmanr(sig_edge_mean[valid_mask], adhd_td_behav[col].values[valid_mask])
        behavior_correlations.append({'behavior': col, 'r': r, 'p': p, 'n': np.sum(valid_mask)})
        sig = "*" if p < 0.05 else ("†" if p < 0.1 else "")
        print(f"  {col}: r={r:.3f}, p={p:.4f} {sig}")

# =============================================================================
# 11. 保存结果
# =============================================================================
OUTPUT_DIR = BASE_PATH / 'reports' / 'comparison' / 'nbs_classification'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

feature_importance.to_csv(OUTPUT_DIR / 'feature_importance.csv', index=False)
print(f"\n特征重要性已保存: {OUTPUT_DIR / 'feature_importance.csv'}")

results_df = pd.DataFrame(results_summary)
results_df['perm_pvalue'] = perm_pvalue
results_df['n_sig_features'] = n_sig_features
results_df.to_csv(OUTPUT_DIR / 'classification_results.csv', index=False)
print(f"分类结果已保存: {OUTPUT_DIR / 'classification_results.csv'}")

print("\n" + "=" * 60)
print("分析完成!")
print("=" * 60)
