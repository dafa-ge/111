@echo off
setlocal
title RAG 知识库面板
cd /d "%~dp0"

set "PY=%~dp0.venv\Scripts\python.exe"
set "URL=http://127.0.0.1:8765"
set "SYS=%SystemRoot%\System32"

if not exist "%PY%" (
    echo [错误] 没有找到 Python 环境：%PY%
    echo 请先按 README 第 1 节建好虚拟环境并安装依赖。
    echo.
    pause
    exit /b 1
)

rem 1. 面板已在运行，直接打开浏览器，不重复启动
"%SYS%\netstat.exe" -ano -p TCP | "%SYS%\findstr.exe" ":8765" | "%SYS%\findstr.exe" "LISTENING" >nul 2>&1
if not errorlevel 1 goto already_running

rem 2. 未运行，启动服务，轮询到就绪后再开浏览器
echo.
echo   RAG 知识库面板
echo   地址：%URL%
echo   正在启动服务，就绪后自动打开浏览器（首次约 5-15 秒）
echo   关闭此窗口即停止服务
echo.

start "" /b "%SYS%\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "%~dp0scripts\wait_and_open.ps1"

"%PY%" -m app.web.server
set "CODE=%ERRORLEVEL%"

echo.
if %CODE% neq 0 (
    echo [服务异常退出，代码 %CODE%] 请把上方报错截图发给维护者。
) else (
    echo [服务已正常停止]
)
pause
exit /b %CODE%

:already_running
echo 面板已经在运行，正在打开浏览器……
start "" "%URL%"
exit /b 0