@echo off
REM Launcher for Web UI Inference Studio using .venv
call .venv\Scripts\activate.bat
echo Starting ONNX Inference Studio on http://localhost:8501 ...
streamlit run app.py
