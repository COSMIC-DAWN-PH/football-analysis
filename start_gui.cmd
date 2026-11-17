@echo off
setlocal
cd /d "%~dp0"

set "FOOTBALL_GUI_POWERSHELL=%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe"
if not exist "%FOOTBALL_GUI_POWERSHELL%" (
  echo [ERROR] Windows PowerShell was not found.
  pause
  exit /b 1
)

"%FOOTBALL_GUI_POWERSHELL%" -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_gui.ps1" %*
exit /b %ERRORLEVEL%
