# 部署指南

本指南将帮助您将手语识别系统部署为可安装的软件包或Web应用。

## 📦 方式一：作为Python包安装

### 安装步骤

1. **克隆或下载项目**
   ```bash
   cd sign_language_recognition
   ```

2. **安装包**
   ```bash
   pip install .
   ```
   或者使用开发模式安装（推荐，便于修改）：
   ```bash
   pip install -e .
   ```

3. **使用命令行工具**
   安装后，您可以使用以下命令：
   ```bash
   # 运行识别程序
   sign-language-recognition
   
   # 收集训练数据
   sign-language-collect
   
   # 训练模型
   sign-language-train
   ```

## 🌐 方式二：作为Web应用运行

### 本地运行

1. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

2. **启动Web应用**
   ```bash
   streamlit run app.py
   ```

3. **访问应用**
   浏览器会自动打开，或手动访问：`http://localhost:8501`

### 部署到云端

#### 选项1：Streamlit Cloud（推荐）

1. **准备GitHub仓库**
   - 将代码推送到GitHub
   - 确保包含 `requirements.txt` 和 `app.py`

2. **部署到Streamlit Cloud**
   - 访问 [streamlit.io/cloud](https://streamlit.io/cloud)
   - 使用GitHub账号登录
   - 点击 "New app"
   - 选择您的仓库和 `app.py` 文件
   - 点击 "Deploy"

3. **注意事项**
   - Streamlit Cloud需要摄像头访问，可能需要使用WebRTC
   - 确保模型文件 `gesture_model.pkl` 已包含在仓库中

#### 选项2：Heroku

1. **创建Procfile**
   ```
   web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
   ```

2. **创建runtime.txt**
   ```
   python-3.11.0
   ```

3. **部署**
   ```bash
   heroku create your-app-name
   git push heroku main
   ```

#### 选项3：Docker部署

1. **创建Dockerfile**
   ```dockerfile
   FROM python:3.11-slim
   
   WORKDIR /app
   
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt
   
   COPY . .
   
   EXPOSE 8501
   
   CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
   ```

2. **构建和运行**
   ```bash
   docker build -t sign-language-app .
   docker run -p 8501:8501 sign-language-app
   ```

## 📱 方式三：打包为桌面应用

### 使用PyInstaller

1. **安装PyInstaller**
   ```bash
   pip install pyinstaller
   ```

2. **打包主程序**
   ```bash
   pyinstaller --onefile --windowed --name="手语识别" main.py
   ```

3. **打包Web应用（可选）**
   ```bash
   pyinstaller --onefile --name="手语识别Web" --add-data "gesture_model.pkl:." app.py
   ```

## 🔧 配置说明

### 环境变量

可以设置以下环境变量：

- `SIGN_LANGUAGE_MODEL_PATH`: 模型文件路径（默认：`gesture_model.pkl`）
- `SIGN_LANGUAGE_DATA_DIR`: 训练数据目录（默认：`training_data`）

### 端口配置

Web应用默认使用8501端口，可以通过以下方式修改：

```bash
streamlit run app.py --server.port=8080
```

## 📋 系统要求

- Python 3.8+
- 摄像头（用于实时识别）
- 至少2GB RAM
- 支持的操作系统：Windows, macOS, Linux

## 🚀 快速启动脚本

项目包含 `run.sh` 脚本，可以快速启动Web应用：

```bash
chmod +x run.sh
./run.sh
```

## ⚠️ 注意事项

1. **模型文件**
   - 确保 `gesture_model.pkl` 文件存在
   - 如果不存在，需要先运行 `python train_model.py`

2. **摄像头权限**
   - Web应用需要浏览器摄像头权限
   - 确保在HTTPS环境下运行（某些浏览器要求）

3. **性能优化**
   - 对于生产环境，考虑使用GPU加速
   - 可以调整MediaPipe的检测参数以提高性能

## 📞 获取帮助

如果遇到问题，请查看：
- [README.md](README.md) - 项目说明
- [用户手册.md](用户手册.md) - 详细使用说明
- GitHub Issues - 报告问题

