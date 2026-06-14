@echo off
chcp 65001 >/dev/null
echo ====================================
echo   法律大模型训练
echo ====================================
echo.

call conda activate legal-llm

echo 环境已激活
echo 开始训练...
echo.

python train_simple.py

echo.
echo ====================================
echo 训练完成！
echo ====================================
pause
