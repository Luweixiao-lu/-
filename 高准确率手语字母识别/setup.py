#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
手语字母识别软件 - 安装配置
"""

from setuptools import setup, find_packages
import os

# 读取README.md文件内容作为长描述
with open(os.path.join(os.path.dirname(__file__), 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# 读取requirements.txt文件内容作为依赖项
with open(os.path.join(os.path.dirname(__file__), 'requirements.txt'), encoding='utf-8') as f:
    requirements = [line.strip() for line in f.readlines() if line.strip() and not line.startswith('#')]

setup(
    name='sign_language_recognizer',
    version='1.0.0',
    description='汉语手指字母（手语字母）实时识别软件',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='手语识别项目组',
    author_email='contact@example.com',
    url='https://github.com/yourusername/sign_language_recognizer',  # 可替换为实际的项目URL
    packages=find_packages(),
    include_package_data=True,
    package_data={
        '': ['*.md', '*.py'],
    },
    install_requires=requirements,
    extras_require={
        'web': ['streamlit>=1.28.0'],
        'dev': ['pytest>=7.4.0', 'black>=23.10.0'],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Multimedia :: Video :: Capture',
    ],
    python_requires='>=3.12',
    entry_points={
        'console_scripts': [
            'sign-language-recognizer=main:main',
            'sign-language-collector=data_collector:main',
            'sign-language-trainer=train_model:train_model',
        ],
    },
    license='MIT',
    keywords='sign language, gesture recognition, hand tracking, computer vision',
)
