@echo off
echo =====================================================
echo  Starting Solace Home UI - Local Test Environment
echo =====================================================
echo.
echo Using local images and bypassing Docker Hub...
echo.

docker-compose -f docker-compose.local.yml --env-file .env up -d

echo.
echo Your local environment has been started.
echo Use 'docker-compose -f docker-compose.local.yml ps' to check status.
pause
