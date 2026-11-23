@echo off
cd /d "%~dp0"
python app_desktop.py
if errorlevel 1 pause
