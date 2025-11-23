"""
手语识别软件安装配置
"""
from setuptools import setup, find_packages
import os

# 读取README文件
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

setup(
    name="sign-language-recognition",
    version="1.0.0",
    author="Sign Language Recognition Team",
    description="基于计算机视觉和机器学习的汉语手指字母识别软件",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/sign_language_recognition",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Video :: Capture",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "opencv-python>=4.8.0",
        "mediapipe==0.10.21",
        "numpy>=1.26.0,<2.0.0",
        "scikit-learn>=1.3.2",
        "scipy>=1.13.0",
        "Pillow==10.1.0",
        "streamlit>=1.28.0",
        "streamlit-webrtc>=0.44.0",
    ],
    entry_points={
        "console_scripts": [
            "sign-language-recognition=main:main",
            "sign-language-collect=data_collector:main",
            "sign-language-train=train_model:train_model",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.pkl"],
    },
)

