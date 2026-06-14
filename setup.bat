@echo off
chcp 65001 >/dev/null
echo ====================================
echo   法律大模型 - 一键启动
echo ====================================
echo.

REM 检查conda环境
echo [1/4] 检查conda环境...
conda info --envs | findstr legal-llm >/dev/null
if errorlevel 1 (
    echo 创建conda环境...
    conda create -n legal-llm python=3.10 -y
)

REM 激活环境并安装依赖
echo [2/4] 安装依赖...
call conda run -n legal-llm pip install transformers peft accelerate datasets bitsandbytes pyyaml fastapi uvicorn

REM 准备数据
echo [3/4] 准备训练数据...
call conda run -n legal-llm python scripts/prepare_data.py

echo [4/4] 环境准备完成！
echo.
echo ====================================
echo 下一步操作：
echo   1. 下载模型: conda run -n legal-llm python scripts/download_model.py
echo   2. 开始训练: conda run -n legal-llm python scripts/finetune.py
echo   3. 启动服务: conda run -n legal-llm python scripts/serve.py
echo ====================================
echo.
pause
