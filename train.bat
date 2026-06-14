@echo off
chcp 65001 >/dev/null
echo ====================================
echo   法律大模型 - 开始训练
echo ====================================
echo.

echo 激活conda环境...
call conda activate legal-llm

echo 检查GPU...
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

echo.
echo 开始训练（这可能需要几分钟到几小时）...
echo.

python scripts/finetune.py

echo.
echo ====================================
echo 训练完成！
echo.
echo 下一步：python scripts/serve.py
echo ====================================
echo.
pause
