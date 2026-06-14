快速启动脚本
@echo off
echo 法律大模型项目 - 快速启动
echo.
echo 步骤1: 安装依赖
pip install -r requirements.txt
echo.
echo 步骤2: 准备示例数据
python scripts/prepare_data.py
echo.
echo 项目初始化完成！
echo 请运行 start.bat 选择后续操作
pause
