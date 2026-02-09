# 正常被试批量预处理脚本
# 基于 rest-eeg-pipeline skill 的 eeg-preprocessing 流程

import os
import os.path as op
import mne
from mne.preprocessing import ICA
from autoreject import AutoReject
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ============================================================
# 配置参数
# ============================================================

# 正常被试列表 (从 DATA/正常被试/ 目录获取)
# 注意: 100 目录下的文件名是 "125 rest.vhdr"
SUBJECT_INFO = {
    '006': '006 REST.vhdr',
    '010': '010 REST.vhdr',
    '068': '068 REST.vhdr',
    '079': '079 REST.vhdr',
    '080': '080 REST.vhdr',
    '093': '093 REST.vhdr',
    '100': '125 rest.vhdr',  # 特殊情况：目录名和文件名不一致
    '108': '108 REST.vhdr',
    '109': '109 REST.vhdr',
    '112': '112 REST.vhdr',
    '127': '127 REST.vhdr',
    '133': '133 REST.vhdr',
}

# 组别前缀
GROUP_PREFIX = 'std_'  # 正常被试前缀

# 路径配置
BASE_PATH = Path(r'd:\LYW\REST_COM')
DATA_PATH = BASE_PATH / 'DATA' / '正常被试'
PREPROC_PATH = BASE_PATH / 'preprocessing'

# 输出路径
SAVE_PATH_RAW = PREPROC_PATH / 'data' / '1raw_rename'
SAVE_PATH_BAD = PREPROC_PATH / 'data' / '2raw_bad'
SAVE_PATH_ICA = PREPROC_PATH / 'data' / '4raw_ica'
SAVE_PATH_EPOCH = PREPROC_PATH / 'data' / '5epoch'
SAVE_PATH_EPOCH_CLEAN = PREPROC_PATH / 'data' / '6epoch_clean'
SAVE_PATH_EVOKED = PREPROC_PATH / 'data' / '7evoked'
FIG_PATH_ICA = PREPROC_PATH / 'figures' / 'ica_scores'
FIG_PATH_EVOKED = PREPROC_PATH / 'figures' / 'evoked_compare'

# 预处理参数
TARGET_SFREQ = 500           # 目标采样率
TARGET_DURATION = 300        # 目标时长（秒），截取中间部分
EPOCH_DURATION = 10.0        # Epoch 时长（秒）
FILTER_LOW = 0.1             # 高通截止频率
FILTER_HIGH = 40.0           # 低通截止频率
ICA_COMPONENTS = 50          # ICA 成分数
EOG_THRESHOLD = 3.0          # 眼电检测阈值

# ============================================================
# 创建输出目录
# ============================================================

for path in [SAVE_PATH_RAW, SAVE_PATH_BAD, SAVE_PATH_ICA,
             SAVE_PATH_EPOCH, SAVE_PATH_EPOCH_CLEAN, SAVE_PATH_EVOKED,
             FIG_PATH_ICA, FIG_PATH_EVOKED]:
    path.mkdir(parents=True, exist_ok=True)

# ============================================================
# 预处理函数
# ============================================================

def read_brainvision_data(vhdr_path):
    """读取 BrainVision 格式数据"""
    raw = mne.io.read_raw_brainvision(str(vhdr_path), preload=True, verbose=False)
    return raw


def process_channels(raw):
    """处理通道：移除边缘电极、设置 EOG"""
    # 移除边缘电极
    channels_to_drop = ['P9', 'F10', 'P10']
    existing = [ch for ch in channels_to_drop if ch in raw.ch_names]
    if existing:
        raw.drop_channels(existing)

    # 设置 F9 为 EOG
    if 'F9' in raw.ch_names:
        raw.set_channel_types({'F9': 'eog'})

    return raw


def crop_middle(raw, target_duration=300):
    """截取中间部分数据"""
    original_duration = raw.times[-1]

    if original_duration < target_duration:
        print(f'  ⚠ 数据时长不足 {target_duration}秒 (实际: {original_duration:.1f}秒)')
        # 使用全部数据
        return raw

    trim = (original_duration - target_duration) / 2
    raw.crop(tmin=trim, tmax=original_duration - trim)

    return raw


def filter_and_resample(raw, target_sfreq=500):
    """滤波与重采样"""
    # 带通滤波
    raw_filtered = raw.copy().filter(FILTER_LOW, FILTER_HIGH, verbose=False)

    # 陷波滤波
    sample_rate = int(raw_filtered.info['sfreq'])
    freqs = np.arange(50, sample_rate / 2, 50)
    raw_notch = raw_filtered.notch_filter(freqs=freqs, verbose=False)

    # 重采样
    if raw_notch.info['sfreq'] != target_sfreq:
        raw_resampled = raw_notch.resample(target_sfreq, verbose=False)
    else:
        raw_resampled = raw_notch

    # 重参考
    raw_rerefer = raw_resampled.set_eeg_reference('average', verbose=False)

    return raw_rerefer


def remove_eog_ica(raw, sub_id, save_fig=True):
    """ICA 去除眼电伪迹"""
    # 1Hz 高通滤波用于 ICA 训练
    raw_filt = raw.copy()
    raw_filt.load_data().filter(l_freq=1., h_freq=None, verbose=False)

    # 训练 ICA
    ica = ICA(n_components=ICA_COMPONENTS, random_state=97, max_iter=500)
    ica.fit(raw_filt, verbose=False)

    # 检测眼电成分
    eog_inds, eog_scores = ica.find_bads_eog(raw_filt, threshold=EOG_THRESHOLD, verbose=False)

    # 保存 ICA 得分图
    if save_fig:
        fig = ica.plot_scores(eog_scores, show=False)
        fig.savefig(FIG_PATH_ICA / f'{GROUP_PREFIX}{sub_id}_ica_scores.png',
                   dpi=150, bbox_inches='tight')
        plt.close(fig)

    # 应用 ICA
    ica.exclude = eog_inds
    raw_clean = raw_filt.copy()
    ica.apply(raw_clean)

    return raw_clean, ica, eog_inds


def create_epochs(raw, duration=10.0):
    """创建固定时长 epochs"""
    events = mne.make_fixed_length_events(raw, duration=duration)

    epochs = mne.Epochs(
        raw, events,
        tmin=0, tmax=duration,
        baseline=None,
        preload=True,
        proj=False,
        detrend=1,
        verbose=False
    )

    return epochs


def apply_autoreject(epochs):
    """AutoReject 自动去除坏 epochs"""
    n_interpolates = np.array([1, 4, 8, 16])
    consensus_percs = np.linspace(0, 0.3, 11)

    picks = mne.pick_types(epochs.info, eeg=True, eog=False)

    ar = AutoReject(
        n_interpolates, consensus_percs,
        picks=picks,
        thresh_method='random_search',
        random_state=42,
        n_jobs=1,
        verbose=False
    )

    ar.fit(epochs)
    epochs_clean = ar.transform(epochs)

    return epochs_clean, ar


def preprocess_subject(sub_id, vhdr_filename):
    """单被试完整预处理流程"""

    output_prefix = f'{GROUP_PREFIX}{sub_id}'

    # 1. 读取数据
    vhdr_path = DATA_PATH / sub_id / vhdr_filename
    if not vhdr_path.exists():
        raise FileNotFoundError(f'文件不存在: {vhdr_path}')

    raw = read_brainvision_data(vhdr_path)
    original_duration = raw.times[-1]
    original_sfreq = raw.info['sfreq']
    print(f'  原始: {original_duration:.1f}秒, {original_sfreq}Hz, {len(raw.ch_names)}通道')

    # 2. 通道处理
    raw = process_channels(raw)

    # 3. 时间截取
    raw = crop_middle(raw, TARGET_DURATION)
    print(f'  截取后: {raw.times[-1]:.1f}秒')

    # 4. 滤波与重采样
    raw = filter_and_resample(raw, TARGET_SFREQ)
    print(f'  滤波重采样: {raw.info["sfreq"]}Hz')

    # 5. ICA 去眼电
    raw_clean, ica, eog_inds = remove_eog_ica(raw, sub_id)
    print(f'  ICA 眼电成分: {eog_inds}')

    # 保存 ICA 结果
    ica.save(SAVE_PATH_ICA / f'{output_prefix}_ica.fif', overwrite=True)
    raw_clean.save(SAVE_PATH_ICA / f'{output_prefix}_raw.fif', overwrite=True)

    # 6. 设置 montage
    montage = mne.channels.make_standard_montage('standard_1020')
    raw_clean.set_montage(montage, match_case=False, on_missing='warn')

    # 7. Epoch 切分
    epochs = create_epochs(raw_clean, EPOCH_DURATION)
    print(f'  创建 {len(epochs)} 个 epochs')

    # 保存原始 epochs
    epochs.save(SAVE_PATH_EPOCH / f'{output_prefix}-epo.fif', overwrite=True)

    # 8. AutoReject
    epochs_clean, ar = apply_autoreject(epochs)
    rejected = len(epochs) - len(epochs_clean)
    retention = len(epochs_clean) / len(epochs) * 100
    print(f'  AutoReject: {len(epochs)} → {len(epochs_clean)} (保留率: {retention:.1f}%)')

    # 9. 保存清理后的 epochs
    epochs_clean.save(SAVE_PATH_EPOCH_CLEAN / f'{output_prefix}-epo.fif', overwrite=True)

    # 10. 计算 evoked 并保存对比图
    evoked_original = epochs.average()
    evoked_clean = epochs_clean.average()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    evoked_original.plot(axes=axes[0], show=False, spatial_colors=True, gfp=True)
    axes[0].set_title(f'原始 Epochs 平均 (n={len(epochs)})', fontsize=12)
    evoked_clean.plot(axes=axes[1], show=False, spatial_colors=True, gfp=True)
    axes[1].set_title(f'AutoReject 清理后 (n={len(epochs_clean)})', fontsize=12)
    plt.tight_layout()

    fig.savefig(FIG_PATH_EVOKED / f'{output_prefix}_evoked_compare.png',
               dpi=150, bbox_inches='tight')
    plt.close(fig)

    # 保存 evoked
    evoked_clean.save(SAVE_PATH_EVOKED / f'{output_prefix}-ave.fif', overwrite=True)

    return {
        'subject': sub_id,
        'original_epochs': len(epochs),
        'clean_epochs': len(epochs_clean),
        'rejected_epochs': rejected,
        'retention_rate': retention,
        'eog_components': eog_inds
    }


# ============================================================
# 主程序
# ============================================================

if __name__ == '__main__':
    print('='*60)
    print('正常被试批量预处理')
    print('='*60)
    print(f'被试数量: {len(SUBJECT_INFO)}')
    print(f'目标采样率: {TARGET_SFREQ} Hz')
    print(f'目标时长: {TARGET_DURATION} 秒')
    print(f'Epoch 时长: {EPOCH_DURATION} 秒')
    print(f'输出前缀: {GROUP_PREFIX}')
    print('='*60)

    success_count = 0
    failed_list = []
    epoch_stats = []

    for sub_id, vhdr_filename in SUBJECT_INFO.items():
        try:
            print(f'\n[{sub_id}] 开始处理...')
            stats = preprocess_subject(sub_id, vhdr_filename)
            epoch_stats.append(stats)
            success_count += 1
            print(f'[{sub_id}] ✓ 处理完成')

        except Exception as e:
            print(f'[{sub_id}] ✗ 处理失败: {str(e)}')
            failed_list.append(sub_id)
            continue

    # 显示结果汇总
    print('\n' + '='*60)
    print('批量处理完成！')
    print(f'成功: {success_count}/{len(SUBJECT_INFO)}')

    if failed_list:
        print(f'\n失败列表: {failed_list}')
    else:
        print('\n所有被试处理成功！')

    # 显示统计信息
    if epoch_stats:
        print('\n=== Epoch 统计 ===')
        print(f'{"被试":<8} {"原始":<8} {"清理后":<8} {"拒绝":<8} {"保留率":<10} {"眼电成分"}')
        print('-'*70)
        for stat in epoch_stats:
            print(f'{stat["subject"]:<8} {stat["original_epochs"]:<8} {stat["clean_epochs"]:<8} '
                  f'{stat["rejected_epochs"]:<8} {stat["retention_rate"]:<10.1f}% {stat["eog_components"]}')

        # 计算平均值
        avg_retention = np.mean([s['retention_rate'] for s in epoch_stats])
        print('-'*70)
        print(f'平均保留率: {avg_retention:.1f}%')

    print(f'\n输出文件:')
    print(f'  ICA 结果: {SAVE_PATH_ICA}')
    print(f'  原始 epochs: {SAVE_PATH_EPOCH}')
    print(f'  清理后 epochs: {SAVE_PATH_EPOCH_CLEAN}')
    print(f'  Evoked 数据: {SAVE_PATH_EVOKED}')
    print(f'  ICA 得分图: {FIG_PATH_ICA}')
    print(f'  对比图: {FIG_PATH_EVOKED}')
    print('='*60)
