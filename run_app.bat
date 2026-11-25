@echo off
title Sign Language Recognition Studio
cd /d "%~dp0Sign-Language-Recogination\project code"
echo Starting Sign Language Recognition Studio...
python Dashboard.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Application stopped or encountered an error.
    pause
)
