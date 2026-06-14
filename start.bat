@echo off
echo ====================================
echo   法律大模型项目启动脚本
echo ====================================
echo.
echo 1. 安装依赖
echo 2. 下载模型
echo 3. 准备数据
echo 4. 开始训练
echo 5. 启动服务
echo.
set /p choice="请选择操作 (1-5): "
if "%choice%"=="1" pip install -r requirements.txt
if "%choice%"=="2" python scripts/download_model.py
if "%choice%"=="3" python scripts/prepare_data.py
if "%choice%"=="4" python scripts/finetune.py
if "%choice%"=="5" python scripts/serve.py
pause
