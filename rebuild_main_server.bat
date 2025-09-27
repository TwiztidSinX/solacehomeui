@echo off
setlocal

:: =============================================================================
:: Solace Home UI - Main Server Rebuilder
:: =============================================================================
:: This script rebuilds only the main-server image to apply recent code fixes.
:: =============================================================================

set "DOCKER_REPO=twiztidsinx/solace-home-ui"

echo.
echo =====================================================
echo  Rebuilding the Main Server Image Locally
echo =====================================================
echo.

set "IMAGE_TAG=%DOCKER_REPO%:main-server-latest"
docker build -t %IMAGE_TAG% -f Dockerfile .
if !errorlevel! neq 0 (
    echo ERROR: Failed to build main-server.
    pause
    goto :eof
)

echo.
echo =====================================================
echo  Main server image has been rebuilt successfully!
echo  You can now run 'rebuild_searxng_nocache.bat'
echo  or 'start_local.bat' if SearXNG is fixed.
echo =====================================================
echo.
pause
endlocal
