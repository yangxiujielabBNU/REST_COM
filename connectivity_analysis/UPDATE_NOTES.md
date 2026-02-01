# Connectivity Analysis - 更新说明

**更新日期**: 2026-01-28

## 主要更新

### ✅ 添加数据对齐功能

已参考 [frequency_analysis_group.ipynb](d:\LYW\REST_COM\analysis\频域分析\frequency_analysis_group.ipynb) 的数据处理方法，在connectivity分析中添加了自动数据对齐功能。

---

## 更新内容详情

### 1. 数据对齐参数配置

在 `cell-5` 添加了统一的数据对齐参数：

```python
TARGET_SFREQ = 500  # 目标采样率
TARGET_DURATION = 10.0  # 目标epoch时长（秒）
TARGET_SAMPLES = int(TARGET_SFREQ * TARGET_DURATION)  # 5000 采样点
```

### 2. 增强的数据加载函数

更新了 `load_epochs()` 函数，添加两步对齐处理：

**步骤1: 采样率对齐**
- 检查采样率是否为500Hz
- 如果不是，自动重采样到500Hz
- 示例输出: `[001] 重采样: 1000.0 -> 500 Hz`

**步骤2: Epoch长度对齐**
- 检查采样点数是否为5000
- 如果不是，裁剪到精确的5000个采样点
- 示例输出: `[022] 统一epoch长度: 5001 -> 5000 采样点`

### 3. 增强的数据验证

更新了 `validate_data()` 函数（`cell-7`），添加：

- 新增 `n_times` 列，显示每个被试的采样点数
- 自动检查数据一致性：
  - 采样率统一性检查
  - 采样点数统一性检查
  - 通道数统一性检查
- 如果检测到不一致，会显示警告信息

**输出示例**：
```
数据一致性检查:
  采样率统一: True (目标: 500 Hz)
  采样点数统一: True (目标: 5000 点)
  通道数统一: True
```

### 4. 改进的加载进度信息

`load_all_subjects()` 函数现在会显示：
- 目标对齐参数
- 实时对齐进度
- 最终对齐确认

---

## 使用说明

### 运行Notebook前的准备

1. **激活conda环境**：
   ```bash
   conda activate mne12_ADHDproject
   ```

2. **检查环境**（可选）：
   ```bash
   python check_environment.py
   ```

3. **如果遇到xarray错误**，参考 [ENVIRONMENT_FIX.md](./ENVIRONMENT_FIX.md) 修复

### 预期行为

运行 `cell-6`（加载数据）时，您会看到：

```
加载被试数据并进行对齐...
目标: 500 Hz, 10.0秒 (5000 采样点)
------------------------------------------------------------
  [001] 重采样: 1000.0 -> 500 Hz
  [022] 统一epoch长度: 5001 -> 5000 采样点
  [024] 统一epoch长度: 5001 -> 5000 采样点
  ...

============================================================
成功加载 12/12 个被试
所有数据已对齐到: 500 Hz, 5000 采样点
============================================================
```

### 数据验证

运行 `cell-7` 后，确认：
- ✅ 所有被试 `sfreq = 500`
- ✅ 所有被试 `n_times = 5000`
- ✅ 所有被试通道数一致

---

## 技术细节

### 对齐逻辑

参考频域分析的批量处理循环（`frequency_analysis_group.ipynb` 的 `cell-6`）：

1. **重采样**使用 `epochs.resample(TARGET_SFREQ)`
2. **裁剪**使用 `epochs.crop(tmin=0, tmax=(TARGET_SAMPLES-1)/TARGET_SFREQ)`
   - 确保精确得到5000个采样点
   - 公式: `tmax = (5000 - 1) / 500 = 9.998秒`

### 为什么需要对齐

您的数据存在两种不一致：

| 被试 | 原始采样率 | 原始采样点数 | 需要处理 |
|------|-----------|-------------|---------|
| 001, 021, 047, 049, 072, 073, 092 | 1000 Hz | - | 重采样 |
| 022, 024, 051, 071, 076 | 500 Hz | 5001 | 裁剪 |

不对齐会导致：
- ❌ 连接性计算时频率分辨率不一致
- ❌ 无法进行跨被试的组平均
- ❌ 网络指标计算结果不可比

---

## 文件清单

| 文件 | 说明 |
|------|------|
| [rest_connectivity.ipynb](./rest_connectivity.ipynb) | **主分析notebook**（已更新） |
| [check_environment.py](./check_environment.py) | 环境诊断脚本 |
| [ENVIRONMENT_FIX.md](./ENVIRONMENT_FIX.md) | 环境问题修复指南 |
| [UPDATE_NOTES.md](./UPDATE_NOTES.md) | 本文档 |

---

## 下一步

1. ✅ 数据对齐已完成 - 您可以开始测试
2. ⏳ 运行notebook，查看数据加载是否正常
3. ⏳ 修复xarray环境问题（如果需要）
4. ⏳ 运行完整的connectivity分析流程

---

## 问题反馈

如果遇到问题，请检查：
1. 数据对齐输出是否正常
2. 数据验证摘要中的一致性检查
3. xarray导入是否成功（参考ENVIRONMENT_FIX.md）
