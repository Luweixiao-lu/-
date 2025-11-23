# 手语字母识别系统 - Streamlit Cloud 部署指南

## 📋 准备工作

1. **GitHub 账号**
   - 确保您有 GitHub 账号
   - 创建一个新的仓库用于存储项目代码

2. **项目文件确认**
   确保项目目录中包含以下必要文件：
   - `streamlit_app.py` - 主应用文件（已优化错误处理）
   - `requirements.txt` - 依赖列表（已优化版本兼容性）
   - `runtime.txt` - Python版本指定（必须包含，确保使用Python 3.10）
   - `hand_landmarks.py` - 手部关键点检测模块
   - `gesture_classifier.py` - 手势分类模块
   - `gesture_model.pkl` - 预训练模型（**重要：必须上传此文件**）
   - `gesture_guide.md` - 手势指南文档
   - `.gitignore` - Git忽略文件

### 模型文件说明

**重要：在部署到Streamlit Cloud前，您必须在本地训练并生成模型文件**

1. 在本地环境运行训练脚本生成 `gesture_model.pkl` 文件
2. 将模型文件添加到您的GitHub仓库中
3. 确保模型文件大小不超过GitHub的文件大小限制（通常为25MB）

如果没有预训练模型，应用将显示警告信息并提供详细的解决步骤指导用户如何训练模型。

## 🚀 部署步骤

### 步骤 1: 上传代码到 GitHub

```bash
# 在项目目录中执行以下命令
# 初始化 Git 仓库
git init

# 添加远程仓库
git remote add origin https://github.com/您的用户名/您的仓库名.git

# 添加文件（确保包含所有必要文件，特别是gesture_model.pkl）
git add .

# 提交更改
git commit -m "Initial commit"

# 推送到 GitHub
git push -u origin main
```

### 步骤 2: 部署到 Streamlit Cloud

1. **访问 Streamlit Cloud**
   - 打开 [Streamlit Cloud](https://share.streamlit.io/)
   - 使用 GitHub 账号登录

2. **创建新应用**
   - 点击右上角的 "New app" 按钮
   - 选择 "From existing repo"
   - 输入您的 GitHub 仓库 URL
   - 主文件路径设置为 `streamlit_app.py`
   - **重要**: Python 版本选择将自动从 runtime.txt 中读取，请确保文件内容为 `python-3.10`
   - 点击 "Deploy!"

3. **等待部署完成**
   - Streamlit Cloud 将自动安装 `requirements.txt` 中的依赖
   - 部署过程大约需要 2-5 分钟
   - 部署成功后，您将看到应用运行界面

4. **验证部署**
   - 检查应用是否正常启动
   - 查看部署日志中是否有任何警告或错误
   - 检查模型文件是否被正确加载
   - 注意：摄像头功能仅在本地运行时可用

## 🔧 常见问题解决

### 1. 依赖安装失败

- **问题**: 某些依赖无法安装
- **解决**: 
  - 我们已优化 `requirements.txt` 文件，移除了版本限制以提高兼容性
  - 确保 `requirements.txt` 文件内容如下：
    ```
    opencv-python-headless
    numpy
    scikit-learn
    joblib
    mediapipe
    streamlit
    ```
  - 检查 `runtime.txt` 文件是否存在并包含 `python-3.9`

### 2. 应用启动后崩溃

- **问题**: 应用启动后立即崩溃
- **解决**: 
  - **缺少模型文件**：确保 `gesture_model.pkl` 文件已正确上传到GitHub仓库
  - **依赖项问题**：检查 `requirements.txt` 文件是否包含所有必要的依赖
  - **Python版本不兼容**：确保 `runtime.txt` 文件指定了正确的Python版本（推荐3.10）
  - **模块导入错误**：确保所有自定义模块（hand_landmarks.py, gesture_classifier.py）都存在且没有语法错误
  - 检查 Streamlit Cloud 日志
  - 应用已添加完善的错误处理机制，可以优雅地处理导入失败、文件不存在等异常
  - 查看日志中具体的错误信息，针对特定错误进行修复

### 3. 摄像头无法访问

- **问题**: 无法访问用户摄像头
- **解决**: 
  - 确保浏览器授予摄像头访问权限
  - 使用 HTTPS 连接（Streamlit Cloud 默认提供）

### 4. 上传文件数量限制

- **问题**: Streamlit Cloud 提示上传文件过多
- **解决**: 确保只包含必要文件，删除测试文件、训练脚本等非必要内容

## 📱 使用说明

1. **启动应用**后，选择您的摄像头设备
2. **调整平滑度**滑块以获得更稳定的识别结果
3. **在摄像头前**展示手语字母（参考 `gesture_guide.md`）
4. **查看识别结果**和置信度
5. **可选**: 启用/禁用手部关键点显示

## 📊 依赖说明

核心依赖项已在 `requirements.txt` 中优化配置，移除了版本限制以提高兼容性。主要包含：
- **opencv-python-headless**: 图像处理库，用于视频捕获和处理
- **numpy**: 数值计算库，用于数据处理
- **scikit-learn**: 机器学习库，用于模型加载和预测
- **joblib**: 用于加载保存的模型
- **mediapipe**: Google 的手势识别库，用于手部关键点检测
- **streamlit**: Web应用框架，用于创建用户界面

通过移除版本限制，让 pip 自动选择与 Python 3.9 兼容的最新版本，解决了版本冲突问题。

## 💡 部署最佳实践

- **使用 runtime.txt**: 始终指定 Python 版本（建议 3.10）以确保环境一致性
- **灵活的依赖管理**: 使用无版本限制的依赖声明，避免版本冲突
- **完善的错误处理**: 添加全面的异常捕获机制，提高应用稳定性
- **监控应用日志**: 关注 Streamlit Cloud 上的日志输出，及时发现问题
- **模型文件管理**:
  - 确保模型文件 `gesture_model.pkl` 已上传到仓库
  - 优化模型以减少加载时间和文件大小
- **用户体验**:
  - 在Cloud环境中提供明确的功能限制说明
  - 为用户提供详细的错误信息和解决步骤
  - 提供本地运行的指导，让用户可以体验完整功能
- **本地测试**: 部署前在本地环境完整测试所有功能
- **渐进式更新**: 对大型更改进行分阶段部署和测试
- **错误处理和优雅降级**: 实现错误处理和优雅降级机制，确保应用在各种环境中都能提供有用的反馈

祝部署顺利！如有问题，请检查日志或参考 [Streamlit 官方文档](https://docs.streamlit.io/streamlit-cloud/get-started/deploy-an-app)。