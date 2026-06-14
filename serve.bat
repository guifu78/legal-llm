@echo off
chcp 65001 >/dev/null
echo ====================================
echo   法律大模型 - 启动API服务
echo ====================================
echo.

echo 激活conda环境...
call conda activate legal-llm

echo 启动服务...
echo API文档地址: http://localhost:8000/docs
echo.

python scripts/serve.py

pause
