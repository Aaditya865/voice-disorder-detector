@echo off
title Testing Laryngeal Voice Disorder Audio Samples
echo =====================================================================
echo  Testing Audio Samples with Wav2Vec 2.0 + Random Forest
echo =====================================================================
echo.

cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PYTHONPATH=%~dp0backend

call "C:\Users\adity\Desktop\Final Year Project\.venv\Scripts\activate.bat"
python test_samples.py
echo.
pause
