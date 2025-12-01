@echo off
echo ====================================
echo  Simple Tauri Debug
echo ====================================
echo.

echo Step 1: Current directory
cd
echo.
pause

echo Step 2: Listing root files
dir /b
echo.
pause

echo Step 3: Checking if frontend exists
dir frontend
echo.
pause

echo Step 4: Checking Rust
rustc --version
echo.
pause

echo Step 5: Checking Cargo Tauri
cargo tauri --version
echo.
pause

echo Step 6: Checking Node
node --version
echo.
pause

echo Step 7: Checking NPM
npm --version
echo.
pause

echo Step 8: Going to frontend directory
cd frontend
dir /b
echo.
pause

echo Step 9: Checking node_modules
dir node_modules 2>nul
echo.
pause

echo Done!
pause
