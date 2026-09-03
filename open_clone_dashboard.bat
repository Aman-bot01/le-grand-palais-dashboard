@echo off
setlocal

set PORT=8504
set CLONE_DIR=C:\AI\le-grand-palais-dashboard
set URL=http://localhost:%PORT%

rem Check if the clone dashboard is already running on this port
netstat -ano | findstr /R /C:":%PORT% .*LISTENING" >nul
if %ERRORLEVEL%==0 (
    echo Clone dashboard already running on port %PORT%. Opening browser...
    start "" "%URL%"
    goto :eof
)

echo Starting Le Grand Palais dashboard on port %PORT%...
start "Le Grand Palais Dashboard" cmd /k "cd /d "%CLONE_DIR%" && streamlit run launch_dashboard_v2.py --server.port %PORT%"

echo Waiting for server to start...
timeout /t 8 /nobreak >nul

start "" "%URL%"

endlocal
