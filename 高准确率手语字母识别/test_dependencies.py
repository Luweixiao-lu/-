#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
依赖测试脚本
测试更新后的依赖配置是否能正常工作
"""

import sys

def test_imports():
    """测试所有必要的导入"""
    print("开始测试依赖导入...")
    print(f"Python版本: {sys.version}")
    print("="*50)
    
    imports = {
        'cv2': 'opencv-python-headless',
        'numpy': 'numpy',
        'sklearn': 'scikit-learn',
        'scipy': 'scipy',
        'PIL': 'Pillow',
        'streamlit': 'streamlit',
        'mediapipe': 'mediapipe'
    }
    
    success_count = 0
    fail_count = 0
    
    for import_name, package_name in imports.items():
        try:
            module = __import__(import_name)
            # 获取版本信息
            version = getattr(module, '__version__', '未知版本')
            print(f"✅ 成功导入: {package_name} ({import_name}) - 版本: {version}")
            success_count += 1
        except ImportError as e:
            print(f"❌ 导入失败: {package_name} ({import_name})")
            print(f"   错误信息: {str(e)}")
            fail_count += 1
    
    print("="*50)
    print(f"总计: 成功 {success_count}, 失败 {fail_count}")
    
    if fail_count == 0:
        print("🎉 所有依赖导入成功！依赖配置正常。")
        print("请将更新后的requirements.txt推送到GitHub，然后重新部署Streamlit Cloud应用。")
    else:
        print("❌ 部分依赖导入失败，请检查安装。")
    
    return fail_count == 0

if __name__ == "__main__":
    test_imports()
