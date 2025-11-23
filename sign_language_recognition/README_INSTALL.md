# 🚀 快速安装和使用指南

## 方式一：Web应用（最简单，推荐）✨

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 启动Web应用
```bash
# 使用启动脚本（推荐）
chmod +x run_web.sh
./run_web.sh

# 或直接运行
streamlit run app.py
```

### 3. 在浏览器中使用
- 浏览器会自动打开
- 或访问：`http://localhost:8501`
- 点击"实时识别"开始使用

**优点**：
- ✅ 界面美观，易于使用
- ✅ 跨平台，无需安装
- ✅ 支持实时识别
- ✅ 包含完整的使用说明

---

## 方式二：作为Python包安装 📦

### 1. 安装包
```bash
pip install .
```

### 2. 使用命令行工具
```bash
# 运行识别程序
sign-language-recognition

# 收集训练数据
sign-language-collect

# 训练模型
sign-language-train
```

**优点**：
- ✅ 全局可用，无需进入项目目录
- ✅ 命令行工具，适合自动化
- ✅ 可以作为依赖安装到其他项目

---

## 方式三：直接运行Python脚本 💻

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 首次使用（训练模型）
```bash
# 收集训练数据
python data_collector.py

# 训练模型
python train_model.py
```

### 3. 运行识别
```bash
# 使用启动脚本
chmod +x run.sh
./run.sh

# 或直接运行
python main.py
```

**优点**：
- ✅ 简单直接
- ✅ 适合开发和调试
- ✅ 完全控制

---

## 📋 系统要求

- Python 3.8+
- 摄像头（内置或USB）
- 2GB+ RAM
- Windows / macOS / Linux

## ⚠️ 首次使用必读

1. **模型文件**：首次使用需要训练模型
   ```bash
   python train_model.py
   ```

2. **摄像头权限**：
   - macOS: 系统偏好设置 → 安全性与隐私 → 摄像头
   - Windows: 设置 → 隐私 → 摄像头
   - Linux: 确保用户有 `/dev/video0` 访问权限

3. **Web应用摄像头**：
   - 浏览器会请求摄像头权限，请点击"允许"
   - 建议使用Chrome或Firefox浏览器

## 🎯 推荐使用方式

- **普通用户**：使用Web应用（方式一）
- **开发者**：作为Python包安装（方式二）
- **学习/调试**：直接运行脚本（方式三）

## 📚 更多信息

- 详细安装说明：[INSTALL.md](INSTALL.md)
- 部署指南：[DEPLOYMENT.md](DEPLOYMENT.md)
- 用户手册：[用户手册.md](用户手册.md)

---

**选择最适合您的方式，开始使用手语识别系统吧！** 🎉

