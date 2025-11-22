@echo off
cd /d "%~dp0"

echo --- Building React Frontend ---
cd frontend
call npm install
call npm run build
cd ..
echo --- Frontend Build Complete ---

REM Activate virtual environment
call llama-gpu\Scripts\activate.bat

REM Start MCP Tool Server in background (same window)
start /b python tool_server.py

REM Start main server in foreground
python main_server.py

pause