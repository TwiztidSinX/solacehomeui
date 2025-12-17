@echo off
echo ====================================
echo  SolaceOS - Tauri Production Build
echo ====================================
echo.
echo Building production Tauri executable...
echo This may take several minutes...
echo.

REM Check if src-tauri directory exists
if not exist "src-tauri" (
    echo ERROR: src-tauri directory not found!
    echo Make sure you're running this from the project root.
    echo.
    pause
    exit /b 1
)

cargo tauri build
echo.
echo Build complete! Check src-tauri\target\release\ for the executable.
pause
