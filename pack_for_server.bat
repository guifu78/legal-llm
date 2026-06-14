@echo off
echo ====================================
echo   打包项目用于服务器部署
echo ====================================
echo.

REM 创建临时目录
if not exist temp_deploy mkdir temp_deploy

REM 复制必要文件
echo 复制项目文件...
xcopy /E /I /Y configs temp_deploy\configs
xcopy /E /I /Y scripts temp_deploy\scripts  
xcopy /E /I /Y data temp_deploy\data
copy /Y requirements.txt temp_deploycopy /Y README.md temp_deploycopy /Y USAGE_GUIDE.md temp_deploycopy /Y Dockerfile temp_deploycopy /Y docker-compose.yml temp_deploycopy /Y deploy.sh temp_deploy
REM 创建压缩包
echo 创建压缩包...
tar -czf legal-llm-server.tar.gz -C temp_deploy .

REM 清理
rmdir /S /Q temp_deploy

echo.
echo ====================================
echo 打包完成: legal-llm-server.tar.gz
echo ====================================
echo.
echo 上传到服务器:
echo   scp legal-llm-server.tar.gz user@server:/home/user/
echo.
echo 在服务器上部署:
echo   tar -xzf legal-llm-server.tar.gz
echo   chmod +x deploy.sh
echo   ./deploy.sh
echo.
pause
