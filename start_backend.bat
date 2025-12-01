@echo off
REM SolaceOS Backend Startup Script
REM This script activates the virtual environment and starts the backend servers

cd /d "%~dp0"

REM Activate virtual environment
call llama-gpu\Scripts\activate.bat

REM Start Browser Proxy Server in background (port 8001)
start /b python browser_proxy.py

REM Start MCP Tool Server in background (port 8000)
start /b python tool_server.py

REM Start main server (port 5000)
python main_server.py
