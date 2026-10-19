@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Test runtime not found. See README.md
  pause
  exit /b 1
)
".venv\Scripts\python.exe" launch.py
