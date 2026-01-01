@echo off
title DroidCam Simple Test
cls

echo.
echo ========================================
echo    DROIDCAM SIMPLE TEST
echo ========================================
echo.
echo Using the WORKING method (no complex timeout)
echo.
echo Make sure:
echo  1. DroidCam Client is running
echo  2. Phone connected (USB or WiFi)
echo  3. Preview video visible in DroidCam Client
echo.
pause

python test_droidcam_simple.py

echo.
echo ========================================
echo If DroidCam works, start the web app!
echo ========================================
pause
