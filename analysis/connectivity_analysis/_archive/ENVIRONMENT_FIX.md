# REST EEG Connectivity Analysis - 环境修复指南

## 问题诊断

您遇到的错误是 `xarray` 包的循环导入问题，这通常由以下原因引起：
- `xarray` 版本与 `mne-connectivity` 不兼容
- Python 环境中的包依赖冲突

## 解决方案

### 选项1：升级 xarray（推荐）

在您的 conda 环境中运行：

```bash
conda activate mne12_ADHDproject
pip install --upgrade xarray
# 或
conda install -c conda-forge xarray>=2023.1.0
```

### 选项2：重新安装 mne-connectivity

```bash
conda activate mne12_ADHDproject
pip uninstall mne-connectivity xarray
pip install mne-connectivity
```

### 选项3：创建新的干净环境（如果上述方法无效）

```bash
# 创建新环境
conda create -n rest_connectivity python=3.10 -y
conda activate rest_connectivity

# 安装核心包
conda install -c conda-forge mne mne-connectivity -y
pip install networkx pandas matplotlib seaborn tqdm jupyter
```

## 验证修复

运行以下代码验证是否修复：

```python
import mne
from mne_connectivity import spectral_connectivity_epochs
import networkx as nx
print("环境正常!")
print(f"MNE版本: {mne.__version__}")
```

## 临时解决方案：简化版Notebook

如果您需要立即开始分析而不想修复环境，我可以为您创建一个不依赖 `mne_connectivity` 的简化版本，使用以下替代方案：

1. 使用 `mne.connectivity.spectral_connectivity` (MNE内置版本)
2. 或者使用更底层的方法手动计算PLI

请告诉我您想使用哪个方案？
