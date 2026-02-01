# 批量预处理脚本
import os
import os.path as op
import mne
from mne.preprocessing import ICA
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import numpy as np

# 设置路径
sub_ids = ['001', '021', '022', '024', '047', '049', '051', '071', '072', '073', '076', '092']
data_path = r'.\data\1raw_rename\\'
fig_path = r'.\figures\ica_scores\\'
save_path_ica = r'.\data\4raw_ica\\'

# 预处理参数
TARGET_SFREQ = 500  # 目标采样率
TARGET_DURATION = 300  # 目标时长（秒），截取中间部分

# 创建目录
os.makedirs(fig_path, exist_ok=True)
os.makedirs(save_path_ica, exist_ok=True)

success_count = 0
failed_list = []

print(f'开始批量处理 {len(sub_ids)} 个被试...\n')
print(f'目标采样率: {TARGET_SFREQ} Hz')
print(f'目标时长: {TARGET_DURATION} 秒（截取中间部分）\n')

for sub_id in sub_ids:
    try:
        print(f'[{sub_id}] 开始处理...')

        # 1. 读取数据
        fname = op.join(data_path + sub_id + '_raw.fif')
        raw = mne.io.read_raw_fif(fname, preload=True, verbose=None)

        original_duration = raw.times[-1]
        original_sfreq = raw.info['sfreq']
        print(f'[{sub_id}]   原始: {original_duration:.1f}秒, {original_sfreq}Hz')

        # 2. 截取中间 300 秒（掐头去尾）
        if original_duration < TARGET_DURATION:
            print(f'[{sub_id}] ⚠ 数据时长不足 {TARGET_DURATION}秒，跳过')
            failed_list.append(sub_id)
            continue

        # 计算截取的起止时间
        trim_duration = (original_duration - TARGET_DURATION) / 2
        tmin = trim_duration
        tmax = original_duration - trim_duration
        raw.crop(tmin=tmin, tmax=tmax)
        print(f'[{sub_id}]   截取: {tmin:.1f}s - {tmax:.1f}s → {raw.times[-1]:.1f}秒')

        # 3. 移除边缘电极
        channels_to_drop = ['P9', 'F10', 'P10']
        existing_channels = [ch for ch in channels_to_drop if ch in raw.ch_names]
        if existing_channels:
            raw.drop_channels(existing_channels)

        # 4. 设置 F9 为 EOG
        if 'F9' in raw.ch_names:
            raw.set_channel_types(mapping={'F9': 'eog'})

        print(f'[{sub_id}]   通道数: {len(raw.ch_names)}, 采样率: {raw.info["sfreq"]} Hz')

        # 5. 插值、滤波、重采样、重参考
        raw_interp = raw.copy().interpolate_bads(reset_bads=True)
        raw_filtered = raw_interp.copy().filter(0.1, 40.)
        sample_rate = int(raw_filtered.info['sfreq'])
        freqs = np.arange(50, sample_rate / 2, 50)
        raw_notch = raw_filtered.copy().notch_filter(freqs=freqs)

        # 重采样到目标采样率
        if raw_notch.info['sfreq'] != TARGET_SFREQ:
            raw_resampled = raw_notch.copy().resample(TARGET_SFREQ)
            print(f'[{sub_id}]   重采样: {raw_notch.info["sfreq"]} Hz → {TARGET_SFREQ} Hz')
        else:
            raw_resampled = raw_notch.copy()
            print(f'[{sub_id}]   采样率已为 {TARGET_SFREQ} Hz')

        raw_rerefer = raw_resampled.copy().set_eeg_reference(ref_channels='average')

        print(f'[{sub_id}]   预处理完成（滤波→重采样→重参考）')

        # 5. ICA 训练
        raw_filt = raw_rerefer.copy()
        raw_filt.load_data().filter(l_freq=1., h_freq=None)
        ica = ICA(n_components=50, random_state=97, max_iter=500)
        ica.fit(raw_filt)

        print(f'[{sub_id}]   ICA 训练完成')

        # 6. 检测眼电成分并保存图片
        eog_inds, eog_scores = ica.find_bads_eog(raw_filt, threshold=3.0)

        fig = ica.plot_scores(eog_scores, show=False)
        fig.savefig(f'{fig_path}{sub_id}_ica_scores.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f'[{sub_id}]   检测到眼电成分: {eog_inds}')
        print(f'[{sub_id}]   已保存图片: {sub_id}_ica_scores.png')

        # 7. 应用 ICA
        ica.exclude = eog_inds
        reconst_raw = raw_filt.copy()
        ica.apply(reconst_raw)

        print(f'[{sub_id}]   已应用 ICA')

        # 8. 保存结果
        ica.save(save_path_ica + sub_id + '_ica.fif')
        reconst_raw.save(save_path_ica + sub_id + '_raw.fif', overwrite=True)

        print(f'[{sub_id}] ✓ 处理完成\n')
        success_count += 1

    except Exception as e:
        print(f'[{sub_id}] ✗ 处理失败: {str(e)}\n')
        failed_list.append(sub_id)
        continue

# 显示结果汇总
print('='*60)
print(f'批量处理完成！')
print(f'成功: {success_count}/{len(sub_ids)}')
print(f'\n处理参数:')
print(f'  - 目标采样率: {TARGET_SFREQ} Hz')
print(f'  - 截取时长: {TARGET_DURATION} 秒（中间部分）')
print(f'  - 滤波范围: 0.1-40 Hz')
if failed_list:
    print(f'\n失败列表: {failed_list}')
else:
    print('\n所有被试处理成功！')
print(f'\nICA 成分得分图已保存到: {fig_path}')
print(f'处理后数据已保存到: {save_path_ica}')
print('='*60)
