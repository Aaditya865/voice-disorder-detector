@echo off
title Laryngeal Voice Disorder - FastAPI Backend
echo =====================================================================
echo  Starting FastAPI Backend Server on http://localhost:8000
echo =====================================================================
echo.

cd /d "%~dp0backend"
set PYTHONIOENCODING=utf-8
set PYTHONPATH=%~dp0backend

call "C:\Users\adity\Desktop\Final Year Project\.venv\Scripts\activate.bat"
echo Backend starting at http://localhost:8000 ...
python -m uvicorn main:app --host 127.0.0.1 --port 8000
pause
