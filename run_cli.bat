@echo off
REM Launcher for CLI Inference using .venv
call .venv\Scripts\activate.bat
python infer.py %*
