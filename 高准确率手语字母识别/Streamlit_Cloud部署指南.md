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
当前的`requirements.txt`使用了精简配置，包含兼容版本：
```
opencv-python-headless==4.5.5.62
mediapipe==0.8.9.1
numpy==1.21.6
scikit-learn==1.0.2
scipy==1.7.3
Pillow==9.0.1
streamlit==1.10.0
```

**注意：** 这些版本经过测试，在Streamlit Cloud上有良好的兼容性，特别是mediapipe==0.8.9.1是一个广泛兼容的稳定版本。

### 3. Streamlit Cloud部署
1. 访问 [Streamlit Community Cloud](https://share.streamlit.io/)
2. 点击 "New app"
3. 连接您的GitHub账户
4. 选择您的仓库
5. 选择分支（通常是`main`或`master`）
6. 文件路径选择 `streamlit_app.py`
7. 点击 "Deploy!"

### 4. 解决常见问题

#### mediapipe安装问题（重点解决）
如果遇到错误：`ERROR: Could not find a version that satisfies the requirement mediapipe`或`ERROR: No matching distribution found for mediapipe`，请按以下步骤解决：

1. **使用兼容的稳定版本**：mediapipe==0.8.9.1在Streamlit Cloud上有最佳兼容性：
   ```
   mediapipe==0.8.9.1
   ```

2. **确保Python版本兼容**：mediapipe 0.8.9.1在Python 3.7-3.9上运行最佳
   
3. **检查架构兼容性**：Streamlit Cloud使用64位Linux环境，确保不使用特定平台的依赖

4. **避免使用过新的版本**：较新的mediapipe版本可能还没有Linux wheel包

#### 其他依赖安装错误
1. **使用更具体的版本**：在`requirements.txt`中指定已知兼容的版本（如当前配置所示）

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

1. **检查详细的部署日志**：在Streamlit Cloud应用页面，点击右上角的三个点，选择"Manage app"，然后查看"Logs"标签

2. **查找特定错误**：搜索"mediapipe"、"error"、"failed"等关键词

3. **使用部署兼容性测试脚本**：项目中包含的`test_deployment_compatibility.py`可以帮助验证依赖兼容性：
   ```bash
   python test_deployment_compatibility.py
   ```

4. **逐步添加依赖**：如果遇到复杂的依赖冲突，可以尝试逐个添加依赖，找到冲突源

5. **本地模拟部署环境**：使用Docker或Python虚拟环境模拟Streamlit Cloud环境进行测试

6. **推送修复后的代码**：修复问题后，只需推送代码到GitHub，Streamlit Cloud会自动重新部署

### 7. 特殊情况处理

#### 如果mediapipe仍然无法安装
1. **考虑使用替代库**：对于简单的手势识别，可以考虑使用更轻量级的替代方案
2. **使用预计算结果**：在本地预处理数据，部署时只加载处理结果
3. **联系Streamlit支持**：如果问题持续存在，可以通过[Streamlit社区论坛](https://discuss.streamlit.io/)寻求帮助
