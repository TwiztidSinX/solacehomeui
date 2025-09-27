@echo off
setlocal

:: =============================================================================
:: Solace Home UI - Final Local Rebuilder
:: =============================================================================
:: This script rebuilds the main-server and SearXNG images from scratch
:: to apply critical fixes, and ensures the minimal STT/TTS images exist.
:: =============================================================================

echo.
echo =====================================================
echo  Rebuilding All Necessary Local Images
echo  This will take some time.
echo =====================================================
echo.

:: --- Main Server ---
set "IMAGE_TAG=twiztidsinx/solace-home-ui:main-server-latest"
echo [1/4] Rebuilding main-server...
docker build --no-cache -t %IMAGE_TAG% -f Dockerfile .
if !errorlevel! neq 0 ( echo ERROR: Failed to build main-server. & pause & goto :eof )
echo --- Successfully rebuilt main-server.

:: --- SearXNG Builder ---
echo.
echo [2/4] Rebuilding the SearXNG builder image...
wsl sh -c "cd /mnt/f/solace-home-ui-v3/searxng && docker build -t localhost/searxng/searxng:builder -f container/builder.dockerfile ."
if !errorlevel! neq 0 ( echo ERROR: Failed to build the SearXNG builder image. & pause & goto :eof )
echo --- Builder image built successfully.

:: --- SearXNG Final ---
echo.
echo [3/4] Rebuilding the final SearXNG image...
wsl sh -c "cd /mnt/f/solace-home-ui-v3/searxng && docker build --no-cache -t searxng/searxng:latest -f container/dist.dockerfile ."
if !errorlevel! neq 0 ( echo ERROR: Failed to build the final SearXNG image. & pause & goto :eof )
echo --- Final SearXNG image built successfully.

:: --- Minimal STT/TTS ---
echo.
echo [4/4] Building minimal STT/TTS images...
set "STT_IMAGE_TAG=twiztidsinx/solace-home-ui:stt-whisper-tiny-latest"
docker build -t %STT_IMAGE_TAG% -f mcpServers/stt_server_whisper_tiny/Dockerfile mcpServers/stt_server_whisper_tiny
if !errorlevel! neq 0 ( echo ERROR: Failed to build STT Whisper Tiny. & pause & goto :eof )

set "TTS_IMAGE_TAG=twiztidsinx/solace-home-ui:tts-kokoro-82m-latest"
docker build -t %TTS_IMAGE_TAG% -f mcpServers/tts_server_kokoro_82m/Dockerfile mcpServers/tts_server_kokoro_82m
if !errorlevel! neq 0 ( echo ERROR: Failed to build TTS Kokoro 82M. & pause & goto :eof )
echo --- Minimal STT/TTS images built successfully.


echo.
echo =====================================================
echo  All local images have been rebuilt successfully!
echo  You can now run the 'start_local.bat' script.
echo =====================================================
echo.
pause
endlocal
