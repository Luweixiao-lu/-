#!/bin/bash

# 手语识别软件启动脚本

echo "=========================================="
echo "手语识别软件"
echo "=========================================="
echo ""

# 检查是否已安装依赖
if ! python3 -c "import cv2" 2>/dev/null; then
    echo "正在安装依赖..."
    pip3 install -r requirements.txt
fi

# 检查模型是否存在
if [ ! -f "gesture_model.pkl" ]; then
    echo "模型文件不存在，正在训练模型..."
    echo "（首次运行会生成示例数据）"
    python3 train_model.py
fi

# 启动识别软件
echo ""
echo "启动手语识别软件..."
python3 main.py

