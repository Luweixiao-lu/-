# 🌐 Streamlit Cloud 部署指南

本指南将帮助您将手语识别应用部署到Streamlit Cloud，让任何人都可以通过链接访问。

## 📋 部署前准备

### 1. 确保以下文件存在

- ✅ `app.py` - Streamlit应用主文件
- ✅ `requirements.txt` - Python依赖
- ✅ `gesture_model.pkl` - 训练好的模型文件（重要！）
- ✅ 所有Python模块文件（`hand_landmarks.py`, `gesture_classifier.py` 等）

### 2. 检查requirements.txt

确保包含所有必要的依赖：

```txt
opencv-python>=4.8.0
mediapipe==0.10.21
numpy>=1.26.0,<2.0.0
scikit-learn>=1.3.2
scipy>=1.13.0
Pillow==10.1.0
streamlit>=1.28.0
```

## 🚀 部署步骤

### 步骤1：将代码推送到GitHub

1. **创建GitHub仓库**（如果还没有）
   ```bash
   git init
   git add .
   git commit -m "Initial commit: Sign Language Recognition App"
   ```

2. **在GitHub上创建新仓库**
   - 访问 https://github.com/new
   - 创建新仓库（例如：`sign-language-recognition`）
   - **不要**初始化README、.gitignore或license

3. **推送代码到GitHub**
   ```bash
   git remote add origin https://github.com/你的用户名/sign-language-recognition.git
   git branch -M main
   git push -u origin main
   ```

### 步骤2：部署到Streamlit Cloud

1. **访问Streamlit Cloud**
   - 打开 https://share.streamlit.io/
   - 使用GitHub账号登录

2. **创建新应用**
   - 点击 "New app"
   - 选择您的GitHub仓库
   - 选择分支（通常是 `main`）
   - 设置主文件路径：`app.py`

3. **配置应用**
   - App URL: 可以自定义（例如：`sign-language-recognition`）
   - Python version: 选择 3.11（推荐）

4. **点击 "Deploy"**

### 步骤3：等待部署完成

- 部署通常需要2-5分钟
- 您可以在部署日志中查看进度
- 部署成功后，您会获得一个URL，例如：
  ```
  https://your-app-name.streamlit.app
  ```

## ⚠️ 重要注意事项

### 1. 模型文件大小

- Streamlit Cloud有文件大小限制
- 如果 `gesture_model.pkl` 太大（>100MB），考虑：
  - 使用Git LFS
  - 或者压缩模型
  - 或者使用外部存储（如GitHub Releases）

### 2. 摄像头访问

- Streamlit Cloud上的应用可以访问用户本地摄像头
- 用户需要授权浏览器访问摄像头
- 建议在应用中添加使用说明

### 3. 性能优化

- 首次加载可能需要一些时间（下载依赖）
- 考虑添加加载提示
- 使用缓存优化性能

## 🔧 故障排除

### 问题1：部署失败

**可能原因**：
- requirements.txt中的依赖版本冲突
- 缺少必要的文件

**解决方案**：
- 检查部署日志
- 确保所有依赖版本兼容
- 确保 `gesture_model.pkl` 已提交到Git

### 问题2：模型文件找不到

**解决方案**：
```bash
# 确保模型文件已提交
git add gesture_model.pkl
git commit -m "Add model file"
git push
```

### 问题3：摄像头无法访问

**解决方案**：
- 确保使用HTTPS（Streamlit Cloud自动提供）
- 在应用中添加清晰的摄像头权限说明
- 建议用户使用Chrome或Firefox浏览器

## 📝 部署检查清单

部署前请确认：

- [ ] 所有代码已推送到GitHub
- [ ] `gesture_model.pkl` 已包含在仓库中
- [ ] `requirements.txt` 包含所有依赖
- [ ] `app.py` 是主文件
- [ ] 测试过本地运行 `streamlit run app.py`
- [ ] 已创建Streamlit Cloud账号
- [ ] 已授权GitHub访问

## 🎉 部署完成！

部署成功后：

1. **分享链接**：将应用URL分享给任何人
2. **更新README**：在项目README中添加应用链接
3. **测试功能**：确保所有功能正常工作

## 🔗 示例应用URL格式

```
https://sign-language-recognition.streamlit.app
```

## 📞 获取帮助

如果遇到问题：
- 查看Streamlit Cloud文档：https://docs.streamlit.io/streamlit-community-cloud
- 检查部署日志
- 提交GitHub Issue

---

**现在任何人都可以通过您分享的链接使用手语识别应用了！** 🎊

