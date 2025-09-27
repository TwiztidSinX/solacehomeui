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

REM Run Nova
echo --- Starting Solace Home UI ---
python main_server.py
pause
