@echo off
set VENV_DIR=mcpServers\voice_server\stt_tts_venv

echo Checking for voice server virtual environment...
if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo Virtual environment not found. Creating one...
    python -m venv "%VENV_DIR%"
    if %errorlevel% neq 0 (
        echo Failed to create virtual environment.
        pause
        exit /b %errorlevel%
    )
    echo Virtual environment created.
)

echo Activating virtual environment...
call "%VENV_DIR%\Scripts\activate.bat"

echo Installing dependencies from requirements.txt...
pip install -r "mcpServers\voice_server\requirements.txt"
if %errorlevel% neq 0 (
    echo Failed to install requirements.
    pause
    exit /b %errorlevel%
)

echo Installing llama-cpp-python with CUDA support for GPU acceleration...
set CMAKE_ARGS="-DGGML_CUDA=on"
set FORCE_CMAKE=1
pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir
if %errorlevel% neq 0 (
    echo Failed to install llama-cpp-python with CUDA.
    echo Please ensure you have the CUDA Toolkit installed and properly configured.
    pause
    exit /b %errorlevel%
)

echo Starting the Voice Server...
uvicorn mcpServers.voice_server.main:app --host 0.0.0.0 --port 8880
if %errorlevel% neq 0 (
    echo Failed to start the voice server.
    pause
    exit /b %errorlevel%
)

pause
