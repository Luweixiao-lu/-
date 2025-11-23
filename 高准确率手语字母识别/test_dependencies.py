#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
依赖测试脚本
只检查导入而不安装，避免conda和pip冲突
"""

import sys

def test_imports():
    """测试依赖导入"""
    print("开始测试依赖导入...")
    
    # 尝试导入主要依赖
    imported_modules = {}
    modules_to_test = [
        ('cv2', 'OpenCV'),
        ('mediapipe', 'MediaPipe'),
        ('numpy', 'NumPy'),
        ('sklearn', 'scikit-learn'),
        ('scipy', 'SciPy'),
        ('PIL', 'Pillow'),
        ('streamlit', 'Streamlit')
    ]
    
    all_imported = True
    
    for module_name, display_name in modules_to_test:
        try:
            module = __import__(module_name)
            version = getattr(module, '__version__', '未知版本')
            imported_modules[display_name] = version
            print(f"✓ {display_name}版本: {version}")
        except ImportError as e:
            print(f"✗ 导入{display_name}失败: {e}")
            all_imported = False
    
    return all_imported

if __name__ == "__main__":
    success = test_imports()
    
    print("\n当前的requirements.txt文件内容:")
    with open('requirements.txt', 'r') as f:
        print(f.read())
    
    print("\n--- 部署指南 ---")
    print("1. 将更新后的requirements.txt推送到GitHub仓库")
    print("   git add requirements.txt")
    print("   git commit -m '修复部署依赖问题'")
    print("   git push origin main")
    print("2. Streamlit Cloud将自动重新部署应用")
    print("3. 如果仍然遇到依赖问题，请在Streamlit Cloud管理界面查看详细日志")
    
    # 推荐使用Docker部署作为替代方案
    print("\n--- 推荐替代方案 ---")
    print("如果Streamlit Cloud部署持续遇到问题，建议使用Docker部署:")
    print("1. 确保Dockerfile已正确配置")
    print("2. 构建Docker镜像: docker build -t sign-language-app .")
    print("3. 运行容器: docker run -p 8501:8501 sign-language-app")
    print("4. 然后可以将Docker镜像部署到任何支持容器的云服务上")
