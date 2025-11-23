# 安装指南

本指南提供了多种安装和使用手语识别系统的方法。

## 📦 方法一：作为Python包安装（推荐）

### 安装步骤

1. **下载或克隆项目**
   ```bash
   git clone <repository-url>
   cd sign_language_recognition
   ```
   或者直接下载ZIP文件并解压。

2. **安装包**
   ```bash
   pip install .
   ```
   
   或者使用开发模式（推荐，便于修改代码）：
   ```bash
   pip install -e .
   ```

3. **验证安装**
   ```bash
   sign-language-recognition --help
   ```

### 使用命令行工具

安装后，您可以使用以下命令：

```bash
# 运行实时识别程序
sign-language-recognition

# 收集训练数据
sign-language-collect

# 训练模型
sign-language-train
```

## 🌐 方法二：运行Web应用

### 快速启动

1. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

2. **启动Web应用**
   ```bash
   # 方式1：使用启动脚本（推荐）
   chmod +x run_web.sh
   ./run_web.sh
   
   # 方式2：直接运行
   streamlit run app.py
   ```

3. **访问应用**
   - 浏览器会自动打开
   - 或手动访问：`http://localhost:8501`

### Web应用功能

- ✅ 实时手语识别
- ✅ 数据收集指导
- ✅ 模型训练说明
- ✅ 使用说明和帮助

## 💻 方法三：直接运行Python脚本

### 基本使用

1. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

2. **首次使用（训练模型）**
   ```bash
   # 收集训练数据
   python data_collector.py
   
   # 训练模型
   python train_model.py
   ```

3. **运行识别程序**
   ```bash
   python main.py
   ```

### 使用启动脚本

项目提供了便捷的启动脚本：

```bash
# 运行桌面应用
chmod +x run.sh
./run.sh

# 运行Web应用
chmod +x run_web.sh
./run_web.sh
```

## 🔧 系统要求

### 最低要求
- Python 3.8 或更高版本
- 2GB RAM
- 摄像头（内置或USB外接）
- 100MB 可用磁盘空间

### 推荐配置
- Python 3.10+
- 4GB+ RAM
- 高清摄像头（720p或更高）
- 良好的光照条件

### 支持的操作系统
- ✅ Windows 10/11
- ✅ macOS 10.14+
- ✅ Linux (Ubuntu 18.04+, Debian 10+, etc.)

## 📋 依赖项说明

主要依赖包括：
- **OpenCV**: 图像处理和摄像头访问
- **MediaPipe**: 手部关键点检测
- **scikit-learn**: 机器学习模型
- **Streamlit**: Web应用框架（仅Web版本需要）
- **NumPy**: 数值计算

完整列表请查看 `requirements.txt`

## ⚠️ 常见问题

### 1. 安装失败

**问题**: `pip install` 失败

**解决方案**:
```bash
# 升级pip
pip install --upgrade pip

# 使用国内镜像源（如果网络较慢）
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2. 摄像头无法访问

**问题**: 无法打开摄像头

**解决方案**:
- 检查摄像头是否被其他程序占用
- 在macOS上，确保已授予终端/IDE摄像头权限
- 在Linux上，确保用户有访问 `/dev/video0` 的权限

### 3. 模型文件不存在

**问题**: 提示 `gesture_model.pkl` 不存在

**解决方案**:
```bash
# 先收集数据（可选，如果已有数据可跳过）
python data_collector.py

# 训练模型
python train_model.py
```

### 4. Web应用无法访问摄像头

**问题**: 浏览器提示无法访问摄像头

**解决方案**:
- 确保使用HTTPS或localhost（某些浏览器要求）
- 检查浏览器权限设置
- 尝试使用Chrome或Firefox浏览器

### 5. 依赖冲突

**问题**: 某些包版本冲突

**解决方案**:
```bash
# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/macOS
# 或
venv\Scripts\activate  # Windows

# 然后安装依赖
pip install -r requirements.txt
```

## 🚀 快速开始检查清单

- [ ] Python 3.8+ 已安装
- [ ] 依赖已安装 (`pip install -r requirements.txt`)
- [ ] 模型文件存在 (`gesture_model.pkl`)
- [ ] 摄像头正常工作
- [ ] 已阅读使用说明

## 📞 获取帮助

如果遇到问题：
1. 查看 [README.md](README.md)
2. 查看 [用户手册.md](用户手册.md)
3. 查看 [DEPLOYMENT.md](DEPLOYMENT.md)
4. 提交GitHub Issue

## 🎉 安装完成！

安装完成后，您可以：
- 运行 `python main.py` 使用桌面应用
- 运行 `streamlit run app.py` 使用Web应用
- 或使用命令行工具 `sign-language-recognition`

祝您使用愉快！

