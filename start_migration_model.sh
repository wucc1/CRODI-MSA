#!/bin/bash

# 迁移模型启动脚本
# 该脚本用于启动项目中的迁移学习（微调）模型

echo "===== 启动迁移模型 ====="
echo "正在进入项目目录..."
cd /root/COLARE/COLARE-op5-toF1/commit_classifier

echo "正在执行迁移学习脚本..."
python finetune.py

echo "===== 迁移模型启动完成 ====="