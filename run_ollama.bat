@echo off
echo Stopping any running Ollama instance...
taskkill /F /IM ollama.exe /T > nul 2>&1
echo Waiting for Ollama to shut down...
timeout /t 5 /nobreak > nul

echo Setting environment variable OLLAMA_KV_CACHE_TYPE=%1
setx OLLAMA_KV_CACHE_TYPE %1

echo Starting Ollama server in the background...
start "" "ollama" serve

echo Ollama is restarting. It may take a moment to become available.
exit /b
