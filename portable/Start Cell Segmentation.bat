@echo off
rem Double-click to start the app. Keep this window open while presenting; close it to stop.
title Cell Segmentation and Counting
cd /d "%~dp0app"
set GRADIO_ANALYTICS_ENABLED=False
set HF_HUB_OFFLINE=1
set PYTHONUTF8=1
echo.
echo  Cell Segmentation and Counting
echo  Starting... your web browser will open automatically (10-30 s on first start).
echo  Keep this window open while using the app. Close it to stop the app.
echo.
"%~dp0python\python.exe" app.py
if errorlevel 1 (
    echo.
    echo  The app stopped with an error. See the message above and app\outputs\app.log
    pause
)
