# 手语字母识别系统 - Streamlit Cloud部署指南

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
