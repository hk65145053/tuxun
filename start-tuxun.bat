@echo off
rem Double-click to start Tuxun and open it in the browser. Close this window to quit.
cd /d "%~dp0"
if not defined HF_ENDPOINT set HF_ENDPOINT=https://hf-mirror.com
if not exist ".venv\Scripts\tuxun.exe" (
  echo .venv not found. Please install Tuxun first, see README.md
  pause
  exit /b 1
)
".venv\Scripts\tuxun.exe" serve --open
pause
