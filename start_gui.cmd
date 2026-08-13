@echo off
setlocal
cd /d "%~dp0"

set "FOOTBALL_GUI_PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%FOOTBALL_GUI_PYTHON%" (
  echo [ERROR] Local Python environment was not found: .venv
  echo.
  echo Run these commands from this directory first:
  echo   py -3.11 -m venv .venv
  echo   .venv\Scripts\python.exe -m pip install -r requirements.txt
  echo   .venv\Scripts\python.exe -m pip install -r requirements-gui.txt
  echo.
  pause
  exit /b 1
)

echo Starting Football Analysis GUI at http://127.0.0.1:8765 ...
"%FOOTBALL_GUI_PYTHON%" "%~dp0gui_server.py"
set "FOOTBALL_GUI_EXIT=%ERRORLEVEL%"

if not "%FOOTBALL_GUI_EXIT%"=="0" (
  echo.
  echo GUI server exited with code %FOOTBALL_GUI_EXIT%.
  pause
)
exit /b %FOOTBALL_GUI_EXIT%
