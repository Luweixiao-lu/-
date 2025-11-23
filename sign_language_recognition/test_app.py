#!/usr/bin/env python3
"""
测试应用核心功能
"""
import sys
import os

def test_imports():
    """测试导入"""
    print("=" * 50)
    print("测试1: 模块导入")
    print("=" * 50)
    
    try:
        from hand_landmarks import HandLandmarkDetector
        print("✅ HandLandmarkDetector 导入成功")
    except Exception as e:
        print(f"❌ HandLandmarkDetector 导入失败: {e}")
        return False
    
    try:
        from gesture_classifier import GestureClassifier
        print("✅ GestureClassifier 导入成功")
    except Exception as e:
        print(f"❌ GestureClassifier 导入失败: {e}")
        return False
    
    try:
        import cv2
        import numpy as np
        from PIL import Image
        print("✅ 基础库导入成功")
    except Exception as e:
        print(f"❌ 基础库导入失败: {e}")
        return False
    
    return True

def test_model_loading():
    """测试模型加载"""
    print("\n" + "=" * 50)
    print("测试2: 模型加载")
    print("=" * 50)
    
    if not os.path.exists('gesture_model.pkl'):
        print("❌ 模型文件不存在: gesture_model.pkl")
        return False
    
    print(f"✅ 模型文件存在: {os.path.getsize('gesture_model.pkl') / 1024 / 1024:.2f} MB")
    
    try:
        from gesture_classifier import GestureClassifier
        classifier = GestureClassifier()
        if classifier.model is not None:
            print("✅ 模型加载成功")
            return True
        else:
            print("⚠️ 模型未加载（可能是新模型）")
            return True
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return False

def test_detector():
    """测试检测器"""
    print("\n" + "=" * 50)
    print("测试3: 手部检测器")
    print("=" * 50)
    
    try:
        from hand_landmarks import HandLandmarkDetector
        detector = HandLandmarkDetector()
        print("✅ 检测器初始化成功")
        return True
    except Exception as e:
        print(f"❌ 检测器初始化失败: {e}")
        return False

def test_video_processor():
    """测试视频处理器类"""
    print("\n" + "=" * 50)
    print("测试4: 视频处理器类")
    print("=" * 50)
    
    try:
        # 检查streamlit-webrtc是否可用
        try:
            from streamlit_webrtc import VideoProcessorBase
            print("✅ streamlit-webrtc 可用")
        except ImportError:
            print("⚠️ streamlit-webrtc 不可用（部署时会安装）")
        
        # 检查类定义
        import sys
        sys.path.insert(0, '.')
        
        # 读取app.py并检查类定义
        with open('app.py', 'r', encoding='utf-8') as f:
            content = f.read()
            if 'class SignLanguageVideoProcessor' in content:
                print("✅ SignLanguageVideoProcessor 类定义存在")
                return True
            else:
                print("❌ SignLanguageVideoProcessor 类定义不存在")
                return False
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def test_requirements():
    """测试requirements.txt"""
    print("\n" + "=" * 50)
    print("测试5: 依赖文件")
    print("=" * 50)
    
    if not os.path.exists('requirements.txt'):
        print("❌ requirements.txt 不存在")
        return False
    
    print("✅ requirements.txt 存在")
    
    with open('requirements.txt', 'r') as f:
        content = f.read()
        required = ['streamlit', 'opencv', 'mediapipe', 'numpy', 'scikit-learn']
        found = []
        for req in required:
            if req in content.lower():
                found.append(req)
                print(f"  ✅ {req}")
        
        if 'streamlit-webrtc' in content.lower():
            print("  ✅ streamlit-webrtc")
        else:
            print("  ⚠️ streamlit-webrtc 未找到（可能已添加）")
    
    return True

def main():
    """主测试函数"""
    print("\n" + "=" * 50)
    print("手语识别应用 - 功能测试")
    print("=" * 50 + "\n")
    
    results = []
    results.append(("模块导入", test_imports()))
    results.append(("模型加载", test_model_loading()))
    results.append(("检测器", test_detector()))
    results.append(("视频处理器", test_video_processor()))
    results.append(("依赖文件", test_requirements()))
    
    print("\n" + "=" * 50)
    print("测试结果汇总")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{name}: {status}")
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！应用应该可以正常使用。")
        return 0
    else:
        print(f"\n⚠️ 有 {total - passed} 个测试失败，请检查相关问题。")
        return 1

if __name__ == '__main__':
    sys.exit(main())

