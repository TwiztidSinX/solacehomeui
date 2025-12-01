@echo off
echo ====================================
echo  SolaceOS - Tauri Development Mode
echo ====================================
echo.
echo Starting Tauri app with hot reload...
echo Frontend will build automatically
echo Python backend will launch automatically
echo.

REM Check if src-tauri directory exists
if not exist "src-tauri" (
    echo ERROR: src-tauri directory not found!
    echo Make sure you're running this from the project root.
    echo Current directory: %CD%
    echo.
    pause
    exit /b 1
)

echo Running Tauri development mode...
echo.
cargo tauri dev

REM If we get here, the command finished (possibly with error)
echo.
echo Process finished. Check above for any errors.
pause
