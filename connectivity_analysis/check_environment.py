#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
环境诊断脚本 - Environment Diagnostic Script

运行此脚本检查您的Python环境是否满足分析要求
Run this script to check if your Python environment meets the analysis requirements
"""

import sys
import importlib
from typing import Dict, Tuple

def check_package(package_name: str, min_version: str = None) -> Tuple[bool, str]:
    """
    检查包是否已安装及版本

    Parameters
    ----------
    package_name : str
        包名
    min_version : str, optional
        最低版本要求

    Returns
    -------
    installed : bool
        是否已安装
    info : str
        版本信息或错误信息
    """
    try:
        module = importlib.import_module(package_name)
        version = getattr(module, '__version__', 'unknown')

        if min_version and version != 'unknown':
            # 简单版本比较 (仅适用于 major.minor.patch 格式)
            try:
                current = tuple(map(int, version.split('.')[:2]))
                required = tuple(map(int, min_version.split('.')[:2]))
                if current >= required:
                    return True, f"✓ {version}"
                else:
                    return False, f"✗ {version} (需要 >= {min_version})"
            except:
                return True, f"✓ {version}"

        return True, f"✓ {version}"
    except ImportError:
        return False, "✗ 未安装"

def main():
    """主诊断流程"""
    print("=" * 60)
    print("REST EEG Connectivity Analysis - 环境诊断")
    print("=" * 60)

    # Python版本检查
    print(f"\nPython版本: {sys.version}")
    py_version = sys.version_info
    if py_version.major < 3 or (py_version.major == 3 and py_version.minor < 8):
        print("⚠️  警告: 建议使用Python 3.8+")
    else:
        print("✓ Python版本满足要求")

    # 必需包检查
    print("\n" + "-" * 60)
    print("核心依赖包检查:")
    print("-" * 60)

    required_packages = {
        'numpy': '1.20.0',
        'pandas': '1.3.0',
        'mne': '1.0.0',
        'mne_connectivity': '0.4.0',
        'xarray': '2023.1.0',  # 关键: xarray需要较新版本
        'networkx': '2.6',
        'matplotlib': '3.4.0',
        'seaborn': '0.11.0',
        'tqdm': None
    }

    all_installed = True
    issues = []

    for package, min_ver in required_packages.items():
        installed, info = check_package(package, min_ver)
        status = "✓" if installed else "✗"
        print(f"  {status} {package:20s} {info}")

        if not installed:
            all_installed = False
            if min_ver:
                issues.append(f"pip install {package}>={min_ver}")
            else:
                issues.append(f"pip install {package}")

    # 特别检查: xarray循环导入问题
    print("\n" + "-" * 60)
    print("特殊兼容性检查:")
    print("-" * 60)

    try:
        import xarray
        import mne_connectivity
        from mne_connectivity import spectral_connectivity_epochs
        print("✓ mne_connectivity导入成功 - 无循环导入错误")
    except ImportError as e:
        print(f"✗ mne_connectivity导入失败:")
        print(f"  错误: {e}")
        print("\n  推荐修复方案:")
        print("  conda activate mne12_ADHDproject")
        print("  pip install --upgrade xarray")
        print("  # 或")
        print("  pip uninstall mne-connectivity xarray")
        print("  pip install mne-connectivity")
        all_installed = False

    # 总结
    print("\n" + "=" * 60)
    if all_installed:
        print("✓ 环境检查通过! 可以运行分析notebook")
    else:
        print("✗ 环境存在问题，请执行以下命令修复:")
        print("\n激活环境:")
        print("  conda activate mne12_ADHDproject")
        print("\n安装/升级缺失的包:")
        for cmd in issues:
            print(f"  {cmd}")
    print("=" * 60)

if __name__ == '__main__':
    main()
