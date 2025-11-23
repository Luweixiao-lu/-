#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简单的应用启动测试脚本
"""

import subprocess
import sys
import time

print("开始测试Streamlit应用启动...")

# 尝试启动Streamlit应用
process = subprocess.Popen(
    [sys.executable, "-m", "streamlit", "run", "streamlit_app.py"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)

try:
    # 等待10秒钟，看是否能成功启动
    start_time = time.time()
    timeout = 10
    
    print("正在启动Streamlit应用，请稍候...")
    
    # 实时输出启动日志
    while time.time() - start_time < timeout:
        if process.poll() is not None:
            # 进程已结束，说明启动失败
            print("应用启动失败!")
            stderr = process.stderr.read()
            print(f"错误信息: {stderr}")
            sys.exit(1)
        
        # 使用非阻塞方式读取输出
        try:
            import select
            if select.select([process.stdout], [], [], 0.5)[0]:
                line = process.stdout.readline()
                if line:
                    print(line.strip())
                    # 检查是否成功启动
                    if "You can now view your Streamlit app in your browser" in line:
                        print("\n✅ 应用成功启动！")
                        break
        except ImportError:
            # 如果select模块不可用，使用简单的超时方式
            time.sleep(0.5)
    
    else:
        print("\n⏰ 启动超时，应用可能没有成功启动")
        
finally:
    # 无论如何都终止进程
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()

print("\n测试完成。")
print("\n建议操作:")
print("1. 如果本地测试成功，但部署失败，可能是Streamlit Cloud环境问题")
print("2. 尝试在GitHub仓库中使用requirements.txt的简化版本")
print("3. 考虑使用requirements-lite.txt作为部署依赖文件")
