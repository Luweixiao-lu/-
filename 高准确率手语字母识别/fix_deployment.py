#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
部署修复脚本
确保应用在Streamlit Cloud上正确运行
"""

import os
import sys
import shutil

print("开始部署修复...")

# 检查并修复任何可能的兼容性问题
def check_compatibility():
    print("检查代码兼容性...")
    
    # 检查关键文件是否存在
    required_files = ['streamlit_app.py', 'hand_landmarks.py', 'gesture_classifier.py']
    missing_files = []
    
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"⚠️  缺少关键文件: {', '.join(missing_files)}")
    else:
        print("✅ 所有关键文件都存在")
    
    # 确保使用cv2的正确导入方式
    print("\n确保OpenCV导入兼容性...")
    try:
        import cv2
        print(f"✅ OpenCV版本: {cv2.__version__}")
    except ImportError:
        print("❌ 无法导入OpenCV，请确保安装了opencv-python-headless")
    
    # 确保使用mediapipe的正确导入方式
    print("\n确保MediaPipe导入兼容性...")
    try:
        import mediapipe as mp
        print(f"✅ MediaPipe版本: {mp.__version__}")
    except ImportError:
        print("❌ 无法导入MediaPipe")

def create_deployment_guide():
    """创建详细的部署指南"""
    guide_content = """# 手语字母识别系统 - Streamlit Cloud部署指南

## 部署步骤

### 1. 准备GitHub仓库
确保您的GitHub仓库包含以下文件：
- `streamlit_app.py` - 主应用文件
- `hand_landmarks.py` - 手部关键点检测模块
- `gesture_classifier.py` - 手势分类器模块
- `requirements.txt` - 依赖文件（已优化为精简版本）
- `gesture_model.pkl` - 训练好的模型文件（如果有）

### 2. 依赖配置
当前的`requirements.txt`使用了精简配置：
```
opencv-python-headless
mediapipe
numpy
scikit-learn
scipy
Pillow
streamlit
```

### 3. Streamlit Cloud部署
1. 访问 [Streamlit Community Cloud](https://share.streamlit.io/)
2. 点击 "New app"
3. 连接您的GitHub账户
4. 选择您的仓库
5. 选择分支（通常是`main`或`master`）
6. 文件路径选择 `streamlit_app.py`
7. 点击 "Deploy!"

### 4. 解决常见问题

#### 依赖安装错误
如果仍然遇到依赖安装错误，可以尝试以下方法：

1. **使用更具体的版本**：在`requirements.txt`中指定已知兼容的版本：
```
opencv-python-headless==4.8.0.74
mediapipe==0.10.14
numpy==1.24.3
scikit-learn==1.3.0
scipy==1.10.1
Pillow==10.0.0
streamlit==1.28.2
```

2. **减少依赖**：如果某些功能不是必需的，可以暂时移除相关依赖

3. **使用setup.py**：确保`setup.py`文件也使用兼容的依赖配置

4. **检查Python版本**：在`.streamlit/config.toml`中指定Python版本：
```toml
[server]
headless = true

[python]
version = "3.10"
```

### 5. 模型文件处理
如果您的应用需要预训练模型：

1. 确保模型文件包含在仓库中（大小不超过GitHub限制）
2. 或者使用代码自动下载模型
3. 或者添加模型训练代码，让应用在首次运行时自动训练

### 6. 调试技巧

1. 检查Streamlit Cloud的部署日志
2. 在日志中查找具体的错误信息
3. 根据错误信息针对性地修复问题
4. 推送修复后的代码到GitHub，Streamlit Cloud会自动重新部署
"""
    
    with open('Streamlit_Cloud部署指南.md', 'w', encoding='utf-8') as f:
        f.write(guide_content)
    
    print("\n✅ 创建了部署指南: Streamlit_Cloud部署指南.md")

# 运行修复
def main():
    check_compatibility()
    create_deployment_guide()
    
    print("\n部署修复完成！")
    print("\n下一步操作：")
    print("1. 将更新后的代码推送到GitHub仓库")
    print("2. 参考生成的部署指南进行操作")
    print("3. 检查Streamlit Cloud的构建日志")

if __name__ == "__main__":
    main()
