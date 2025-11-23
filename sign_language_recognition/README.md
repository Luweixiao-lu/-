# 汉语手指字母识别软件

这是一个基于计算机视觉和机器学习的汉语手指字母（手语字母）识别软件，可以实时识别30个不同的手语字母手势（A-Z, ZH, CH, SH, NG）。

## ✨ 功能特点

- 🎥 实时摄像头手语识别
- 📚 支持30个汉语手指字母
- 🔍 可视化手部关键点检测
- 🎨 简洁易用的图形界面（桌面版 + Web版）
- 🌐 Web应用，易于共享和部署
- 📦 可作为Python包安装
- 🤖 基于机器学习的智能识别
- 📊 置信度评估和结果平滑

## 📋 系统要求

- Python 3.12+
- 摄像头（内置或外接 USB）
- macOS / Windows / Linux

## 🚀 快速开始

### 方式一：Web应用（推荐）✨

最简单的方式，适合所有用户：

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 启动Web应用
streamlit run app.py
# 或使用启动脚本
chmod +x run_web.sh && ./run_web.sh

# 3. 浏览器会自动打开，或访问 http://localhost:8501
```

### 方式二：作为Python包安装 📦

```bash
# 安装包
pip install .

# 使用命令行工具
sign-language-recognition  # 运行识别
sign-language-collect      # 收集数据
sign-language-train        # 训练模型
```

### 方式三：直接运行脚本 💻

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 首次使用（训练模型）
python data_collector.py  # 收集训练数据
python train_model.py     # 训练模型

# 3. 运行识别软件
python main.py
# 或使用启动脚本
chmod +x run.sh && ./run.sh
```

📖 **详细安装说明**: 查看 [README_INSTALL.md](README_INSTALL.md) 或 [INSTALL.md](INSTALL.md)

## 📖 详细文档

**完整用户手册**: 请查看 [`用户手册.md`](用户手册.md)

手册包含：
- 📘 详细的安装指南
- 📗 完整的使用说明
- 📙 手势动作详解
- 📕 常见问题解答
- 🔧 故障排除指南

## 🎯 支持的手势

软件支持识别以下30个手语字母：
- **A-Z**（26个英文字母）
- **ZH, CH, SH, NG**（4个汉语拼音声母）

详细手势说明请参考 [`gesture_guide.md`](gesture_guide.md)

## 💡 使用提示

1. ✅ 确保摄像头正常工作
2. ✅ 在良好的光照条件下使用
3. ✅ 将手放在摄像头前，保持手势清晰
4. ✅ 每个手势保持2-3秒以便识别
5. ✅ 确保手部完全在摄像头视野内

## 📁 项目结构

```
sign_language_recognition/
├── main.py                 # 主程序（桌面版手势识别）
├── app.py                  # Web应用（Streamlit）
├── data_collector.py       # 数据收集工具
├── train_model.py          # 模型训练脚本
├── hand_landmarks.py       # 手部关键点检测
├── gesture_classifier.py   # 手势分类器
├── setup.py                # Python包安装配置
├── requirements.txt        # 依赖库列表
├── run.sh                  # 桌面版启动脚本
├── run_web.sh              # Web版启动脚本
├── README.md               # 项目说明（本文件）
├── README_INSTALL.md       # 快速安装指南
├── INSTALL.md              # 详细安装说明
├── DEPLOYMENT.md           # 部署指南
├── 用户手册.md            # 完整用户手册
├── gesture_guide.md        # 手势指南
└── IMPROVEMENTS.md         # 改进说明
```

## 🔧 技术栈

- **计算机视觉**: OpenCV, MediaPipe
- **机器学习**: scikit-learn (Random Forest)
- **Web框架**: Streamlit
- **数据处理**: NumPy
- **编程语言**: Python 3.8+

## 🌐 部署选项

本项目支持多种部署方式：

- ✅ **本地运行**: 桌面应用或Web应用
- ✅ **云端部署**: Streamlit Cloud（推荐，一键部署）⭐
- ✅ **打包安装**: Python包安装，命令行工具
- ✅ **其他平台**: Heroku, Docker

### 🚀 快速部署到云端（让任何人通过链接使用）

**最简单的方法**：3步完成部署到Streamlit Cloud

1. 推送到GitHub
2. 在 https://share.streamlit.io/ 部署
3. 获得链接并分享

详细步骤请查看：
- 📖 [DEPLOY_QUICK.md](DEPLOY_QUICK.md) - 快速部署指南（3步）
- 📖 [STREAMLIT_CLOUD_DEPLOY.md](STREAMLIT_CLOUD_DEPLOY.md) - 详细部署说明
- 📖 [DEPLOYMENT.md](DEPLOYMENT.md) - 完整部署文档

## 📝 许可证

本项目仅供学习和研究使用。

## 🤝 贡献

欢迎提交 Issue 或 Pull Request！

---

**详细使用说明请查看 [`用户手册.md`](用户手册.md)**

