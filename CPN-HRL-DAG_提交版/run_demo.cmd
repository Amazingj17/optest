@echo off
python "%~dp0run_demo.py" %*
if errorlevel 1 pause
