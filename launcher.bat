@echo off
:restart
call run_nova.bat
echo Nova exited. Restarting in 5 seconds...
timeout /t 5
goto restart