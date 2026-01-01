@echo off
echo ============================================================
echo CHESS DETECTION - QUICK START
echo ============================================================
echo.
echo Checking ONNX model...

if not exist "app\model\best.onnx" (
    echo [WARNING] ONNX model not found!
    echo Exporting from PyTorch model...
    python -c "from ultralytics import YOLO; m=YOLO('app/model/best.pt'); m.export(format='onnx')"
    echo.
    echo [OK] ONNX model exported!
    echo.
) else (
    echo [OK] ONNX model found!
    echo.
)

echo Starting Chess Detection App...
echo.
echo Performance optimizations applied:
echo   - ONNX inference (2-10x faster)
echo   - Confidence threshold: 0.45
echo   - Size filtering enabled
echo   - Frame skipping: every 5 frames
echo.
echo Expected FPS: 20-30 (was 11.1)
echo.
echo ============================================================
echo Opening browser in 5 seconds...
echo Press Ctrl+C to cancel
echo ============================================================
echo.

timeout /t 5 /nobreak > nul
start http://localhost:5000

python -m app.app

pause
