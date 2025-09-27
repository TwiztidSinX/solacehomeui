@echo off
setlocal enabledelayedexpansion

:: =============================================================================
:: Solace Home UI - Docker Image Publishing Script (v7 - Simple & Reliable)
:: =============================================================================
:: This script builds and pushes only the images that are not already on
:: Docker Hub. It uses a simple, repetitive structure to be foolproof.
:: =============================================================================

set "DOCKER_REPO=twiztidsinx/solace-home-ui"
set "LOG_FILE=publish_log.txt"

:: Clear previous log file and start new log
(
    echo =====================================================
    echo  Starting Solace Home UI Image Publish at !date! !time!
    echo =====================================================
    echo.
) > %LOG_FILE%

:: --- Main Script Execution ---
echo.
echo =====================================================
echo  Building and Pushing All Solace Home UI Images
echo  Repository: %DOCKER_REPO%
echo  (Skipping images that already exist on Docker Hub)
echo =====================================================
echo.

:: --- Main Server ---
set "IMAGE_TAG=%DOCKER_REPO%:main-server-latest"
echo [1/9] Checking for Main Server...
docker manifest inspect %IMAGE_TAG% >nul 2>&1
if %errorlevel% equ 0 (
    echo --- Image already exists on Docker Hub. Rebuilding locally due to code changes...
    docker build -t %IMAGE_TAG% -f Dockerfile .
    if !errorlevel! neq 0 ( echo ERROR: Failed to build Main Server. & pause & goto :eof )
    docker push %IMAGE_TAG%
    if !errorlevel! neq 0 ( echo ERROR: Failed to push Main Server. & pause & goto :eof )
    echo --- Successfully published Main Server.
) else (
    echo --- Image not found. Building and pushing Main Server...
    docker build -t %IMAGE_TAG% -f Dockerfile .
    if !errorlevel! neq 0 ( echo ERROR: Failed to build Main Server. & pause & goto :eof )
    docker push %IMAGE_TAG%
    if !errorlevel! neq 0 ( echo ERROR: Failed to push Main Server. & pause & goto :eof )
    echo --- Successfully published Main Server.
)

:: --- STT Whisper Tiny ---
set "IMAGE_TAG=%DOCKER_REPO%:stt-whisper-tiny-latest"
echo.
echo [2/9] Checking for STT Whisper Tiny...
docker manifest inspect %IMAGE_TAG% >nul 2>&1
if %errorlevel% equ 0 (
    echo --- Image already exists on Docker Hub. Skipping.
) else (
    echo --- Image not found. Building and pushing STT Whisper Tiny...
    docker build -t %IMAGE_TAG% -f mcpServers/stt_server_whisper_tiny/Dockerfile mcpServers/stt_server_whisper_tiny
    if !errorlevel! neq 0 ( echo ERROR: Failed to build STT Whisper Tiny. & pause & goto :eof )
    docker push %IMAGE_TAG%
    if !errorlevel! neq 0 ( echo ERROR: Failed to push STT Whisper Tiny. & pause & goto :eof )
    echo --- Successfully published STT Whisper Tiny.
)

:: --- STT Parakeet NeMo ---
set "IMAGE_TAG=%DOCKER_REPO%:stt-parakeet-nemo-latest"
echo.
echo [3/9] Checking for STT Parakeet NeMo...
docker manifest inspect %IMAGE_TAG% >nul 2>&1
if %errorlevel% equ 0 (
    echo --- Image already exists on Docker Hub. Skipping.
) else (
    echo --- Image not found. Building and pushing STT Parakeet NeMo...
    docker build -t %IMAGE_TAG% -f mcpServers/stt_server_parakeet_nemo/Dockerfile mcpServers/stt_server_parakeet_nemo
    if !errorlevel! neq 0 ( echo ERROR: Failed to build STT Parakeet NeMo. & pause & goto :eof )
    docker push %IMAGE_TAG%
    if !errorlevel! neq 0 ( echo ERROR: Failed to push STT Parakeet NeMo. & pause & goto :eof )
    echo --- Successfully published STT Parakeet NeMo.
)

:: --- STT Whisper Large v3 ---
set "IMAGE_TAG=%DOCKER_REPO%:stt-whisper-large-v3-latest"
echo.
echo [4/9] Checking for STT Whisper Large v3...
docker manifest inspect %IMAGE_TAG% >nul 2>&1
if %errorlevel% equ 0 (
    echo --- Image already exists on Docker Hub. Skipping.
) else (
    echo --- Image not found. Building and pushing STT Whisper Large v3...
    docker build -t %IMAGE_TAG% -f mcpServers/stt_server_whisper_large_v3/Dockerfile mcpServers/stt_server_whisper_large_v3
    if !errorlevel! neq 0 ( echo ERROR: Failed to build STT Whisper Large v3. & pause & goto :eof )
    docker push %IMAGE_TAG%
    if !errorlevel! neq 0 ( echo ERROR: Failed to push STT Whisper Large v3. & pause & goto :eof )
    echo --- Successfully published STT Whisper Large v3.
)

:: --- STT Kyutai High ---
set "IMAGE_TAG=%DOCKER_REPO%:stt-kyutai-high-latest"
echo.
echo [5/9] Checking for STT Kyutai High...
docker manifest inspect %IMAGE_TAG% >nul 2>&1
if %errorlevel% equ 0 (
    echo --- Image already exists on Docker Hub. Skipping.
) else (
    echo --- Image not found. Building and pushing STT Kyutai High...
    docker build -t %IMAGE_TAG% -f mcpServers/stt_server_kyutai_high/Dockerfile mcpServers/stt_server_kyutai_high
    if !errorlevel! neq 0 ( echo ERROR: Failed to build STT Kyutai High. & pause & goto :eof )
    docker push %IMAGE_TAG%
    if !errorlevel! neq 0 ( echo ERROR: Failed to push STT Kyutai High. & pause & goto :eof )
    echo --- Successfully published STT Kyutai High.
)

:: --- TTS Kokoro 82M ---
set "IMAGE_TAG=%DOCKER_REPO%:tts-kokoro-82m-latest"
echo.
echo [6/9] Checking for TTS Kokoro 82M...
docker manifest inspect %IMAGE_TAG% >nul 2>&1
if %errorlevel! equ 0 (
    echo --- Image already exists on Docker Hub. Skipping.
) else (
    echo --- Image not found. Building and pushing TTS Kokoro 82M...
    docker build -t %IMAGE_TAG% -f mcpServers/tts_server_kokoro_82m/Dockerfile mcpServers/tts_server_kokoro_82m
    if !errorlevel! neq 0 ( echo ERROR: Failed to build TTS Kokoro 82M. & pause & goto :eof )
    docker push %IMAGE_TAG%
    if !errorlevel! neq 0 ( echo ERROR: Failed to push TTS Kokoro 82M. & pause & goto :eof )
    echo --- Successfully published TTS Kokoro 82M.
)

:: --- TTS Kyutai Safetensors ---
set "IMAGE_TAG=%DOCKER_REPO%:tts-kyutai-safetensors-latest"
echo.
echo [7/9] Checking for TTS Kyutai Safetensors...
docker manifest inspect %IMAGE_TAG% >nul 2>&1
if %errorlevel% equ 0 (
    echo --- Image already exists on Docker Hub. Skipping.
) else (
    echo --- Image not found. Building and pushing TTS Kyutai Safetensors...
    docker build -t %IMAGE_TAG% -f mcpServers/tts_server_kyutai_safetensors/Dockerfile mcpServers/tts_server_kyutai_safetensors
    if !errorlevel! neq 0 ( echo ERROR: Failed to build TTS Kyutai Safetensors. & pause & goto :eof )
    docker push %IMAGE_TAG%
    if !errorlevel! neq 0 ( echo ERROR: Failed to push TTS Kyutai Safetensors. & pause & goto :eof )
    echo --- Successfully published TTS Kyutai Safetensors.
)

:: --- TTS Kyutai Medium ---
set "IMAGE_TAG=%DOCKER_REPO%:tts-kyutai-medium-latest"
echo.
echo [8/9] Checking for TTS Kyutai Medium...
docker manifest inspect %IMAGE_TAG% >nul 2>&1
if %errorlevel% equ 0 (
    echo --- Image already exists on Docker Hub. Skipping.
) else (
    echo --- Image not found. Building and pushing TTS Kyutai Medium...
    docker build -t %IMAGE_TAG% -f mcpServers/tts_server_kyutai_medium/Dockerfile mcpServers/tts_server_kyutai_medium
    if !errorlevel! neq 0 ( echo ERROR: Failed to build TTS Kyutai Medium. & pause & goto :eof )
    docker push %IMAGE_TAG%
    if !errorlevel! neq 0 ( echo ERROR: Failed to push TTS Kyutai Medium. & pause & goto :eof )
    echo --- Successfully published TTS Kyutai Medium.
)

:: --- TTS Orpheus GGUF ---
set "IMAGE_TAG=%DOCKER_REPO%:tts-orpheus-gguf-latest"
echo.
echo [9/9] Checking for TTS Orpheus GGUF...
docker manifest inspect %IMAGE_TAG% >nul 2>&1
if %errorlevel% equ 0 (
    echo --- Image already exists on Docker Hub. Skipping.
) else (
    echo --- Image not found. Building and pushing TTS Orpheus GGUF...
    docker build -t %IMAGE_TAG% -f mcpServers/tts_server_orpheus_gguf/Dockerfile mcpServers/tts_server_orpheus_gguf
    if !errorlevel! neq 0 ( echo ERROR: Failed to build TTS Orpheus GGUF. & pause & goto :eof )
    docker push %IMAGE_TAG%
    if !errorlevel! neq 0 ( echo ERROR: Failed to push TTS Orpheus GGUF. & pause & goto :eof )
    echo --- Successfully published TTS Orpheus GGUF.
)

echo.
echo =====================================================
echo  Image check and publish process complete!
echo =====================================================
echo.
pause
endlocal
