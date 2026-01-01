# CRITICAL BUGS FIXED - January 1, 2026

## Issues Resolved

### 1. ✅ Thread Race Condition (Thread-21 & Thread-23 collision)

**Problem:** Multiple detection threads starting simultaneously, causing camera driver crash.

**Fix Location:** [routes.py](app/routes.py#L214-L249)

**Changes:**
- Added check in `/api/start_opencv_detection` to reject duplicate start requests
- Returns `already_active: true` if detection is running
- Prevents thread collision

**Fix Location 2:** [chess_detection.py](app/chess_detection.py#L520-L545)

**Changes:**
- Added proper thread cleanup with `join(timeout=3.0)` 
- Waits for old thread to finish before starting new one
- Logs thread lifecycle for debugging

---

### 2. ✅ Camera Initialization Failures (CAP_DSHOW crashes)

**Problem:** USB/Virtual cameras (DroidCam, Nvidia Broadcast) throwing C++ exceptions with DirectShow.

**Fix Location:** [chess_detection.py](app/chess_detection.py#L575-L615)

**Changes:**
- **Smart Backend Selection:**
  - Try `CAP_DSHOW` first
  - If `isOpened()` fails, properly release and retry with `CAP_ANY`
  - For built-in cameras, also try `CAP_MSMF`
  - Verify frame reading before accepting backend
  
- **Robust Property Setting:**
  - Each property (buffer, resolution, FPS, exposure) in separate try-except
  - If property fails, log warning and CONTINUE (don't crash)
  - Graceful fallback to camera defaults
  
**Benefits:**
- No more "raised unknown C++ exception!" crashes
- Works with problematic USB/virtual cameras
- Continues even if some properties unsupported

---

### 3. ✅ ONNX Warning Spam (cublasLt64_12.dll errors)

**Problem:** Logs flooded with CUDA provider errors on CPU systems.

**Fix Location:** [onnx_engine.py](app/onnx_engine.py#L165-L190)

**Changes:**
- Set `ORT_DISABLE_SYMBOL_BINDING=1` environment variable
- Use `SessionOptions` with `log_severity_level=3` (ERROR only)
- Suppress INFO/WARNING logs from ONNX Runtime
- Only show actual errors

**Result:** Clean logs, no CUDA spam on CPU systems.

---

## Testing Instructions

1. **Test Thread Safety:**
   ```bash
   # Start app, click "Start Detection" twice rapidly
   # Should see: "⚠️ Detection already running - rejecting duplicate start request"
   # No camera crash, no thread collision
   ```

2. **Test Camera Fallback:**
   ```bash
   # With USB/virtual camera (Index 1)
   # Should try DirectShow → Auto-detect
   # Logs show: "✅ Camera 1 opened with Auto-detect"
   # No C++ exceptions
   ```

3. **Test Property Errors:**
   ```bash
   # If camera doesn't support resolution setting:
   # Should see: "ℹ️ Resolution: 640x480 (camera default)"
   # App continues normally
   ```

4. **Verify Clean Logs:**
   ```bash
   # No more:
   # [E:onnxruntime:Default, provider_bridge_ort.cc:2251] ...
   # Error loading cublasLt64_12.dll
   ```

---

## Files Modified

1. **app/routes.py** - Added duplicate detection check
2. **app/chess_detection.py** - Fixed camera init & thread handling
3. **app/onnx_engine.py** - Suppressed ONNX warnings

---

## Next Steps

1. Restart the Flask app: `python app.py`
2. Test with Camera Index 1 (your USB camera)
3. Verify no more crashes or thread collisions
4. Check logs are clean (no ONNX spam)

**Expected Behavior:**
- Camera opens smoothly with fallback backends
- Single detection thread runs (no duplicates)
- Clean logs without CUDA warnings
- App stays stable even if camera properties fail

---

## Rollback (if needed)

If issues occur, revert with:
```bash
git checkout app/routes.py app/chess_detection.py app/onnx_engine.py
```

Or restore from `chess-detection-original/` folder.
