#!/bin/bash
# 手语识别Web应用启动脚本

echo "=========================================="
echo "  手语识别系统 - Web应用"
echo "=========================================="
echo ""

# 检查Python是否安装
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到Python3，请先安装Python"
    exit 1
fi

# 检查依赖是否安装
echo "检查依赖..."
if ! python3 -c "import streamlit" 2>/dev/null; then
    echo "正在安装依赖..."
    pip install -r requirements.txt
fi

# 检查模型文件
if [ ! -f "gesture_model.pkl" ]; then
    echo ""
    echo "⚠️  警告: 未找到模型文件 gesture_model.pkl"
    echo "   请先运行以下命令训练模型："
    echo "   python train_model.py"
    echo ""
    read -p "是否继续启动Web应用？(y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 启动Streamlit应用
echo ""
echo "正在启动Web应用..."
echo "浏览器将自动打开，或手动访问: http://localhost:8501"
echo ""
echo "按 Ctrl+C 停止应用"
echo "=========================================="
echo ""

streamlit run app.py

