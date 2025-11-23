#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
精简部署配置测试脚本
用于验证依赖兼容性，特别是mediapipe
"""

import sys
import subprocess
import importlib.util

# 测试依赖列表
REQUIRED_PACKAGES = [
    "opencv-python-headless",
    "mediapipe",
    "numpy",
    "scikit-learn",
    "scipy",
    "Pillow",
    "streamlit"
]

def print_header():
    """打印脚本头部信息"""
    print("=" * 60)
    print("精简部署配置测试脚本")
    print("验证Streamlit Cloud部署依赖兼容性")
    print("=" * 60)

def check_package_import(package_name):
    """检查包是否可以导入"""
    try:
        # 尝试导入包
        spec = importlib.util.find_spec(package_name)
        if spec is None:
            return False, f"无法找到包: {package_name}"
        
        # 尝试实际导入
        module = importlib.import_module(package_name)
        # 获取版本信息（如果可用）
        version = getattr(module, '__version__', '未知版本')
        return True, f"成功导入 {package_name} 版本 {version}"
    except ImportError as e:
        return False, f"导入失败: {str(e)}"
    except Exception as e:
        return False, f"发生错误: {str(e)}"

def test_packages():
    """测试所有必需的包"""
    print("\n开始测试包导入兼容性...\n")
    success_count = 0
    failure_count = 0
    failure_details = []
    
    for package in REQUIRED_PACKAGES:
        print(f"测试 {package}...", end=" ")
        success, message = check_package_import(package)
        if success:
            print("✓ 成功")
            print(f"  {message}")
            success_count += 1
        else:
            print("✗ 失败")
            print(f"  {message}")
            failure_count += 1
            failure_details.append((package, message))
        print()
    
    # 打印总结
    print("=" * 60)
    print(f"测试结果: {success_count} 成功, {failure_count} 失败")
    print("=" * 60)
    
    if failure_count > 0:
        print("\n失败详情:")
        for package, message in failure_details:
            print(f"- {package}: {message}")
        
        print("\n建议:")
        if "mediapipe" in [p for p, _ in failure_details]:
            print("1. 确保使用兼容的mediapipe版本 (0.8.9.1是一个稳定版本)")
            print("2. 检查Python版本兼容性 (推荐Python 3.7-3.9)")
            print("3. 对于Streamlit Cloud部署，请参考requirements.txt中的版本约束")
        
        return False
    
    return True

def check_python_version():
    """检查Python版本"""
    print(f"当前Python版本: {sys.version}")
    
    # mediapipe推荐的Python版本是3.7-3.9
    major, minor = sys.version_info[:2]
    if major != 3 or minor < 7 or minor > 9:
        print("⚠️  警告: mediapipe在Python 3.7-3.9上兼容性最好")
        return False
    return True

def main():
    """主函数"""
    print_header()
    
    # 检查Python版本
    print("\nPython版本检查:")
    python_check = check_python_version()
    
    # 测试包导入
    packages_check = test_packages()
    
    # 总体评估
    print("\n部署兼容性评估:")
    if python_check and packages_check:
        print("✅ 兼容性良好，可以尝试部署到Streamlit Cloud")
        print("建议操作:")
        print("1. 确保requirements.txt包含正确的版本约束")
        print("2. 将代码推送到GitHub仓库")
        print("3. 在Streamlit Cloud上创建新应用")
    else:
        print("❌ 存在兼容性问题，需要修复后再部署")
        print("请先在本地环境解决上述问题")

if __name__ == "__main__":
    main()
