@echo off
setlocal enabledelayedexpansion

:: =============================================================================
:: Solace Home UI - Interactive Docker Setup Script v2
:: =============================================================================
:: This script will ask you a series of questions to customize your Docker
:: setup, configure credentials, and generate the required .env file.
:: =============================================================================

:: --- Initialization ---
set "DOCKER_COMMAND=docker-compose -f docker/docker-compose.yml"
set "ENV_CONTENT="

echo.
echo  Welcome to the Solace Home UI Setup!
echo.
echo  This script will guide you through a custom installation.
echo  For each component, please enter 'y' for yes or 'n' for no.
echo.
echo -----------------------------------------------------
echo  Core Components & Credentials
echo -----------------------------------------------------

:: --- MongoDB ---
set "INSTALL_MONGO="
set /p INSTALL_MONGO="Do you want to run the local MongoDB container? (Required if you don't have your own) (y/n): "
if /i "!INSTALL_MONGO!"=="y" (
    set "DOCKER_COMMAND=!DOCKER_COMMAND! -f docker/docker-compose.mongo.yml"
    echo.
    echo    Please provide credentials for the local MongoDB instance.
    set /p MONGO_USER_INPUT="   Enter MongoDB Username (default: admin): "
    if not defined MONGO_USER_INPUT set MONGO_USER_INPUT=admin
    set /p MONGO_PASS_INPUT="   Enter MongoDB Password (default: password): "
    if not defined MONGO_PASS_INPUT set MONGO_PASS_INPUT=password
    set "ENV_CONTENT=!ENV_CONTENT!MONGO_USER=!MONGO_USER_INPUT!&echo!MONGO_PASSWORD=!MONGO_PASS_INPUT!"
)

echo.
echo -----------------------------------------------------
echo  API Keys (Optional)
echo -----------------------------------------------------
set "SETUP_API_KEYS="
set /p SETUP_API_KEYS="Do you want to configure API keys now? (e.g., for wake word) (y/n): "
if /i "!SETUP_API_KEYS!"=="y" (
    echo.
    echo    Enter the API key for the 'Hey Nova' wake word.
    echo    You can get a free key from https://console.picovoice.ai/
    set /p PICOVOICE_KEY_INPUT="   Picovoice API Key: "
    if defined PICOVOICE_KEY_INPUT (
        set "ENV_CONTENT=!ENV_CONTENT!&echo!PICOVOICE_API_KEY=!PICOVOICE_KEY_INPUT!"
    )
)

echo.
echo -----------------------------------------------------
echo  Optional Services
echo -----------------------------------------------------

:: --- Ollama ---
set "INSTALL_OLLAMA="
set /p INSTALL_OLLAMA="Do you want to run the local Ollama container? (y/n): "
if /i "!INSTALL_OLLAMA!"=="y" (
    set "DOCKER_COMMAND=!DOCKER_COMMAND! -f docker/docker-compose.ollama.yml"
)

:: --- SearXNG ---
set "INSTALL_SEARXNG="
set /p INSTALL_SEARXNG="Do you want to run the local SearXNG container for web searches? (y/n): "
if /i "!INSTALL_SEARXNG!"=="y" (
    set "DOCKER_COMMAND=!DOCKER_COMMAND! -f docker/docker-compose.searxng.yml"
)

echo.
echo -----------------------------------------------------
echo  Speech-to-Text (STT) Pipeline
echo -----------------------------------------------------
echo  Please choose ONE STT pipeline to install.
echo.
echo  1. CPU / Low VRAM (Whisper-Tiny, ~1GB VRAM)
echo  2. Low VRAM (Parakeet-NeMo, ~2GB VRAM)
echo  3. Medium VRAM (Whisper-large-v3, ~4GB VRAM)
echo  4. High VRAM (Kyutai-STT-1b, ~6GB VRAM)
echo  5. None
echo.
set "STT_CHOICE="
set /p STT_CHOICE="Enter your choice (1-5): "

if "!STT_CHOICE!"=="1" set "DOCKER_COMMAND=!DOCKER_COMMAND! -f docker/docker-compose.cpu.stt.yml"
if "!STT_CHOICE!"=="2" set "DOCKER_COMMAND=!DOCKER_COMMAND! -f docker/docker-compose.low.stt.yml"
if "!STT_CHOICE!"=="3" set "DOCKER_COMMAND=!DOCKER_COMMAND! -f docker/docker-compose.med.stt.yml"
if "!STT_CHOICE!"=="4" set "DOCKER_COMMAND=!DOCKER_COMMAND! -f docker/docker-compose.high.stt.yml"

echo.
echo -----------------------------------------------------
echo  Text-to-Speech (TTS) Pipeline
echo -----------------------------------------------------
echo  Please choose ONE TTS pipeline to install.
echo.
echo  1. CPU / Low VRAM (Kokoro-82M, ~1.5GB VRAM)
echo  2. Low VRAM (Kyutai-TTS, ~2.5GB VRAM)
echo  3. Medium VRAM (Kyutai-TTS-1.6b, ~5GB VRAM)
echo  4. High VRAM (Orpheus-TTS, ~7GB VRAM)
echo  5. None
echo.
set "TTS_CHOICE="
set /p TTS_CHOICE="Enter your choice (1-5): "

if "!TTS_CHOICE!"=="1" set "DOCKER_COMMAND=!DOCKER_COMMAND! -f docker/docker-compose.cpu.tts.yml"
if "!TTS_CHOICE!"=="2" set "DOCKER_COMMAND=!DOCKER_COMMAND! -f docker/docker-compose.low.tts.yml"
if "!TTS_CHOICE!"=="3" set "DOCKER_COMMAND=!DOCKER_COMMAND! -f docker/docker-compose.med.tts.yml"
if "!TTS_CHOICE!"=="4" set "DOCKER_COMMAND=!DOCKER_COMMAND! -f docker/docker-compose.high.tts.yml"

:: --- Finalization ---

:: Generate the .env file
echo Generating .env file...
if exist .env del .env
if /i "!INSTALL_MONGO!"=="y" (
    echo MONGO_USER=!MONGO_USER_INPUT! > .env
    echo MONGO_PASSWORD=!MONGO_PASS_INPUT! >> .env
)
if /i "!SETUP_API_KEYS!"=="y" (
    if defined PICOVOICE_KEY_INPUT (
        echo PICOVOICE_API_KEY=!PICOVOICE_KEY_INPUT! >> .env
    )
)

:: Append the final command arguments
set "DOCKER_COMMAND=!DOCKER_COMMAND! --env-file .env up --build -d"

echo.
echo -----------------------------------------------------
echo  Setup Complete!
echo -----------------------------------------------------
echo.
echo  Generated .env file with your credentials.
echo  Based on your selections, the following command will be executed:
echo.
echo  !DOCKER_COMMAND!
echo.
echo  Press any key to start the services...
pause > nul

:: Execute the final command in a new window
start "Solace Home UI Docker" cmd /k !DOCKER_COMMAND!

echo.
echo  Your Solace Home UI services are being started in a new window.
echo  It may take a few minutes for all containers to become healthy.
echo.
pause
endlocal
