import os
import sys

def setup_startup():
    print("Setting up automatic startup on Windows boot...")
    
    project_dir = os.path.dirname(os.path.abspath(__file__))
    run_bat_path = os.path.join(project_dir, 'run_server.bat')
    stop_bat_path = os.path.join(project_dir, 'stop_server.bat')
    
    # 1. Create run_server.bat
    # Starts python app.py with a window title so we can identify it if needed
    run_content = f'''@echo off
title AuraGuardServer
cd /d "{project_dir}"
python app.py
'''
    with open(run_bat_path, 'w') as f:
        f.write(run_content)
    print(f"Created run launcher: {run_bat_path}")

    # 2. Create stop_server.bat
    # Finds the PID listening on port 5000 and terminates it
    stop_content = '''@echo off
echo Stopping Face Recognition Gate Lock server on port 5000...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":5000" ^| findstr "LISTENING"') do (
    taskkill /f /pid %%a
    echo Server process %%a terminated.
)
pause
'''
    with open(stop_bat_path, 'w') as f:
        f.write(stop_content)
    print(f"Created stop launcher: {stop_bat_path}")

    # 3. Create VBScript launcher in Windows Startup directory
    startup_dir = os.path.join(os.environ['APPDATA'], 'Microsoft', 'Windows', 'Start Menu', 'Programs', 'Startup')
    vbs_path = os.path.join(startup_dir, 'launch_gate_lock.vbs')
    
    # The VBScript runs the run_server.bat file in hidden mode (0) so there is no command prompt window.
    vbs_content = f'''Set WshShell = CreateObject("WScript.Shell")
WshShell.Run """{run_bat_path}""", 0, False
'''
    with open(vbs_path, 'w') as f:
        f.write(vbs_content)
    print(f"Created Windows startup shortcut: {vbs_path}")
    print("SUCCESS: Setup complete. The server will run automatically in the background every time you boot your computer!")

if __name__ == '__main__':
    setup_startup()
