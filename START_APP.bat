@echo off
echo =========================================
echo  Chess Detection - Quick Start Guide
echo =========================================
echo.
echo [1/3] Checking Python version...
python --version
echo.

echo [2/3] Checking dependencies...
python -c "import cv2, numpy, ultralytics, sklearn, chess, filterpy; print('✅ All core dependencies OK!')" 2>nul
if errorlevel 1 (
    echo ❌ Missing dependencies detected!
    echo.
    echo Installing required packages...
    pip install flask flask-socketio flask-login flask-bcrypt ultralytics opencv-python scikit-learn python-chess filterpy albumentations
    echo.
)

echo [3/3] Starting application...
echo.
echo =========================================
echo  Application will start at:
echo  http://localhost:5000
echo =========================================
echo.
echo Default login:
echo   Username: admin
echo   Password: admin123
echo.
echo Press Ctrl+C to stop the server
echo =========================================
echo.

cd app
python app.py

pause
