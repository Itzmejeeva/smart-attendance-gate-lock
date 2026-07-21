@echo off
echo Stopping Face Recognition Gate Lock server on port 5000...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":5000" ^| findstr "LISTENING"') do (
    taskkill /f /pid %%a
    echo Server process %%a terminated.
)
pause
