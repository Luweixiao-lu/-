# ⚡ 快速部署到Streamlit Cloud（3步完成）

## 🎯 目标
让任何人都可以通过一个链接访问您的手语识别应用。

## 📝 步骤

### 1️⃣ 准备GitHub仓库

```bash
# 在项目目录下执行
git init
git add .
git commit -m "Deploy Sign Language Recognition App"

# 在GitHub创建新仓库，然后：
git remote add origin https://github.com/你的用户名/仓库名.git
git push -u origin main
```

**重要**：确保 `gesture_model.pkl` 文件已包含！

### 2️⃣ 部署到Streamlit Cloud

1. 访问：https://share.streamlit.io/
2. 用GitHub登录
3. 点击 "New app"
4. 选择您的仓库和 `app.py`
5. 点击 "Deploy"

### 3️⃣ 分享链接

部署完成后，您会得到一个URL，例如：
```
https://your-app-name.streamlit.app
```

**就这么简单！** 现在任何人都可以通过这个链接使用您的应用了。

---

## ⚠️ 常见问题

**Q: 模型文件太大怎么办？**
A: 如果超过100MB，可以使用Git LFS或压缩模型。

**Q: 摄像头无法使用？**
A: 确保用户授权浏览器访问摄像头，建议使用Chrome浏览器。

**Q: 部署失败？**
A: 检查 `requirements.txt` 中的依赖是否正确，查看部署日志找出问题。

---

详细说明请查看：[STREAMLIT_CLOUD_DEPLOY.md](STREAMLIT_CLOUD_DEPLOY.md)

