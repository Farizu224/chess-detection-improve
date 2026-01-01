@echo off
title Identify Cameras - Find DroidCam
cls

echo.
echo ========================================
echo    CAMERA IDENTIFIER
echo ========================================
echo.
echo This will help you find which camera is:
echo   - Laptop webcam (built-in)
echo   - DroidCam (from phone)
echo.
echo Make sure DroidCam Client is running!
echo.
pause

python identify_cameras_quick.py

echo.
echo ========================================
echo.
echo After identifying cameras:
echo   1. Update droidcam_config.py with correct index
echo   2. Run START_APP.bat
echo.
pause
