@echo off
echo ======================================
echo TESTING USB CAMERA (INDEX 1)
echo ======================================
echo.

cd /d "%~dp0"

python -c "import cv2; cap = cv2.VideoCapture(1); print('Opening camera 1...'); print('Opened:', cap.isOpened()); ret, frame = cap.read() if cap.isOpened() else (False, None); print('Can read:', ret); print('Shape:', frame.shape if ret else 'N/A'); cap.release()"

if errorlevel 1 (
    echo.
    echo ERROR: Failed to test camera
    pause
    exit /b 1
)

echo.
echo Creating test window...
python -c "import cv2, numpy as np; cv2.namedWindow('Test'); cv2.imshow('Test', np.zeros((480,640,3), dtype=np.uint8)); print('Window shown! Close it to continue...'); cv2.waitKey(0); cv2.destroyAllWindows()"

pause
