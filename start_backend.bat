@echo off
REM SolaceOS Backend Startup Script
REM This script uses the virtual environment Python directly (no activation needed)

cd /d "%~dp0"

REM Set the Python executable path
set PYTHON_EXE=%~dp0llama-gpu\Scripts\python.exe

REM Start Browser Proxy Server in background (port 8001)
start "BrowserProxy" /b "%PYTHON_EXE%" browser_proxy.py

REM Start MCP Tool Server in background (port 8000)
start "ToolServer" /b "%PYTHON_EXE%" tool_server.py

REM Start main server (port 5000)
"%PYTHON_EXE%" main_server.py
