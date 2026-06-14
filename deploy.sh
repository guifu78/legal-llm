#!/bin/bash
# 法律大模型服务器部署脚本

set -e

echo "=========================================="
echo "  法律大模型服务器部署脚本"
echo "=========================================="

# 检查 Python
echo "
[1/6] 检查 Python 环境..."
if command -v python3 &> /dev/null; then
    python3 --version
else
    echo "错误: 未找到 Python3"
    exit 1
fi

# 检查 NVIDIA
echo "
[2/6] 检查 NVIDIA 驱动..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "警告: 未找到 nvidia-smi"
fi

# 创建虚拟环境
echo "
[3/6] 创建虚拟环境..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "虚拟环境创建成功"
else
    echo "虚拟环境已存在"
fi

# 激活虚拟环境
source venv/bin/activate

# 安装依赖
echo "
[4/6] 安装依赖..."
pip install --upgrade pip
pip install -r requirements.txt

# 准备数据
echo "
[5/6] 准备训练数据..."
python scripts/prepare_data.py

# 创建目录
echo "
[6/6] 创建目录结构..."
mkdir -p data models finetuned logs

echo "
=========================================="
echo "部署完成！"
echo "=========================================="
echo ""
echo "下一步操作："
echo "  1. 下载模型: python scripts/download_model.py"
echo "  2. 开始训练: python scripts/finetune.py"
echo "  3. 启动服务: python scripts/serve.py"
