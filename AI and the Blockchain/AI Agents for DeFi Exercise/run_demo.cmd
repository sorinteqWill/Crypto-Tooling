@echo off
setlocal
REM No PowerShell, no pip required.
REM Requires: Python 3.8+ installed (py launcher preferred)

where py >nul 2>&1
if %ERRORLEVEL%==0 (
  py ai_chain_bot.py --once
  goto :eof
)

where python >nul 2>&1
if %ERRORLEVEL%==0 (
  python ai_chain_bot.py --once
  goto :eof
)

echo Python not found. Install Python 3.8+ and ensure it is on PATH.
exit /b 1
