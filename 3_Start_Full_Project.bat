@echo off
title Launching Full Laryngeal Voice Disorder System...
echo =====================================================================
echo  Starting Backend (FastAPI :8000) and Frontend (React :5173)...
echo =====================================================================
echo.

start "Backend Server" cmd /k "1_Start_Backend.bat"
timeout /t 2 /nobreak >nul
start "Frontend UI" cmd /k "2_Start_Frontend.bat"

echo.
echo Both servers are starting up:
echo - Backend API:  http://localhost:8000/docs
echo - Frontend UI:  http://localhost:5173
echo.
pause
