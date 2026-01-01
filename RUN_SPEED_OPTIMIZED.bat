@echo off
echo ============================================================
echo SPEED-OPTIMIZED CHESS DETECTION
echo ============================================================
echo.
echo Optimizations Applied:
echo   [+] Ultra-lightweight overlay (15+ text -^> 3 text)
echo   [+] Removed crop_to_square overhead
echo   [+] Pure frame caching (2/3 frames = zero processing)
echo   [+] Reduced inference frequency
echo.
echo Expected Performance:
echo   Baseline Camera: 23.6 FPS
echo   OLD App FPS: 11.3 FPS (48%% efficiency)
echo   NEW App FPS: 18-22 FPS (76-93%% efficiency)
echo.
echo Expected Improvement: +60-95%% FPS!
echo ============================================================
echo.
echo Starting optimized app in 3 seconds...
timeout /t 3 /nobreak > nul

cd chess-detection-improve
python -m app.app

pause
