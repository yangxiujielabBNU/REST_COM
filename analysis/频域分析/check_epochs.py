import mne
import os.path as op

# 检查失败的被试
failed_subs = ['022', '024', '051', '071', '076']
success_subs = ['001', '021', '047', '049', '072', '073', '092']

data_path = r'..\..\preprocessing\data\6epoch_clean\\'

print("失败的被试:")
for sub_id in failed_subs:
    fname = op.join(data_path, sub_id + '-epo.fif')
    epochs = mne.read_epochs(fname, preload=False, verbose=False)
    print(f"  {sub_id}: {len(epochs.times)} samples, {epochs.tmin} to {epochs.tmax} sec, {len(epochs)} epochs")

print("\n成功的被试:")
for sub_id in success_subs:
    fname = op.join(data_path, sub_id + '-epo.fif')
    epochs = mne.read_epochs(fname, preload=False, verbose=False)
    print(f"  {sub_id}: {len(epochs.times)} samples, {epochs.tmin} to {epochs.tmax} sec, {len(epochs)} epochs")
