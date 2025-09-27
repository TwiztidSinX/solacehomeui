@echo off
setlocal enabledelayedexpansion

:: =============================================================================
:: Solace Home UI - Final Minimal Local Builder
:: =============================================================================
:: This script builds the essential images for a minimal local test:
:: 1. The slim main-server image.
:: 2. The CPU-based STT and TTS images.
:: =============================================================================

set "DOCKER_REPO=twiztidsinx/solace-home-ui"

echo.
echo =====================================================
echo  Building Final Minimal Images Locally
echo =====================================================
echo.

:: --- Main Server ---
set "IMAGE_TAG=%DOCKER_REPO%:main-server-latest"
echo [1/3] Building main-server...
docker build -t %IMAGE_TAG% -f Dockerfile .
if !errorlevel! neq 0 ( echo ERROR: Failed to build main-server. & pause & goto :eof )
echo --- Successfully built main-server.

:: --- STT Whisper Tiny (CPU) ---
set "IMAGE_TAG=%DOCKER_REPO%:stt-whisper-tiny-latest"
echo.
echo [2/3] Building STT Whisper Tiny...
docker build -t %IMAGE_TAG% -f mcpServers/stt_server_whisper_tiny/Dockerfile mcpServers/stt_server_whisper_tiny
if !errorlevel! neq 0 ( echo ERROR: Failed to build STT Whisper Tiny. & pause & goto :eof )
echo --- Successfully built STT Whisper Tiny.

:: --- TTS Kokoro 82M (CPU) ---
set "IMAGE_TAG=%DOCKER_REPO%:tts-kokoro-82m-latest"
echo.
echo [3/3] Building TTS Kokoro 82M...
docker build -t %IMAGE_TAG% -f mcpServers/tts_server_kokoro_82m/Dockerfile mcpServers/tts_server_kokoro_82m
if !errorlevel! neq 0 ( echo ERROR: Failed to build TTS Kokoro 82M. & pause & goto :eof )
echo --- Successfully built TTS Kokoro 82M.

echo.
echo =====================================================
echo  All necessary local images have been built!
echo  You can now run the main 'setup.bat' script.
echo =====================================================
echo.
pause
endlocal