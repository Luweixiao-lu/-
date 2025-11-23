# 🌐 一键部署指南

## 🎯 让任何人都可以通过链接使用您的应用

### 最简单的方法：Streamlit Cloud（推荐）⭐

**3步完成部署**：

1. **推送到GitHub**
   ```bash
   git init
   git add .
   git commit -m "Deploy app"
   git remote add origin https://github.com/你的用户名/仓库名.git
   git push -u origin main
   ```

2. **部署到Streamlit Cloud**
   - 访问：https://share.streamlit.io/
   - 用GitHub登录
   - 点击 "New app"
   - 选择仓库和 `app.py`
   - 点击 "Deploy"

3. **分享链接**
   - 获得URL：`https://your-app.streamlit.app`
   - 分享给任何人使用！

**详细步骤**：查看 [DEPLOY_QUICK.md](DEPLOY_QUICK.md) 或 [STREAMLIT_CLOUD_DEPLOY.md](STREAMLIT_CLOUD_DEPLOY.md)

---

## 📋 部署前检查清单

- [ ] `app.py` 文件存在
- [ ] `requirements.txt` 包含所有依赖
- [ ] `gesture_model.pkl` 已包含（重要！）
- [ ] 所有Python模块文件都在仓库中
- [ ] 本地测试通过：`streamlit run app.py`

---

## 🚀 其他部署选项

### 选项1：Streamlit Cloud（最简单）
- ✅ 免费
- ✅ 自动HTTPS
- ✅ 自动更新（Git推送后）
- 📖 查看：[STREAMLIT_CLOUD_DEPLOY.md](STREAMLIT_CLOUD_DEPLOY.md)

### 选项2：Heroku
- ✅ 免费套餐可用
- ⚠️ 需要配置Procfile
- 📖 查看：[DEPLOYMENT.md](DEPLOYMENT.md)

### 选项3：Docker
- ✅ 可部署到任何支持Docker的平台
- ⚠️ 需要Docker知识
- 📖 查看：[DEPLOYMENT.md](DEPLOYMENT.md)

---

## ⚠️ 重要提示

1. **模型文件**：确保 `gesture_model.pkl` 已提交到Git
2. **文件大小**：如果模型文件>100MB，考虑使用Git LFS
3. **摄像头**：用户需要授权浏览器访问摄像头
4. **浏览器**：建议使用Chrome或Firefox

---

## 🎉 部署完成后

您的应用URL格式：
```
https://your-app-name.streamlit.app
```

**现在任何人都可以通过这个链接使用您的手语识别应用了！**

---

需要帮助？查看：
- [DEPLOY_QUICK.md](DEPLOY_QUICK.md) - 快速部署（3步）
- [STREAMLIT_CLOUD_DEPLOY.md](STREAMLIT_CLOUD_DEPLOY.md) - 详细部署指南
- [DEPLOYMENT.md](DEPLOYMENT.md) - 完整部署文档

