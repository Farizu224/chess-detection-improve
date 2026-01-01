"""
Try to fix camera exposure and brightness settings
"""
import cv2
import numpy as np

print("=" * 70)
print("FIXING CAMERA EXPOSURE & BRIGHTNESS")
print("=" * 70)

# Try index 1 (your USB camera that showed settings but black frame)
idx = 1

print(f"\nOpening camera {idx} with DirectShow...")
cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("❌ Cannot open camera")
    exit(1)

print("✅ Camera opened\n")

# Get current settings
print("Current settings:")
print(f"  Brightness: {cap.get(cv2.CAP_PROP_BRIGHTNESS)}")
print(f"  Contrast: {cap.get(cv2.CAP_PROP_CONTRAST)}")
print(f"  Exposure: {cap.get(cv2.CAP_PROP_EXPOSURE)}")
print(f"  Gain: {cap.get(cv2.CAP_PROP_GAIN)}")

# Try AUTO exposure first
print("\n1. Setting AUTO_EXPOSURE = 0.75 (auto mode)...")
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)

# Increase brightness to maximum
print("2. Setting BRIGHTNESS = 255 (maximum)...")
cap.set(cv2.CAP_PROP_BRIGHTNESS, 255)

# Increase contrast
print("3. Setting CONTRAST = 128...")
cap.set(cv2.CAP_PROP_CONTRAST, 128)

# Increase gain
print("4. Setting GAIN = 100...")
cap.set(cv2.CAP_PROP_GAIN, 100)

# Set exposure to higher value (less negative = brighter)
print("5. Setting EXPOSURE = -3 (brighter)...")
cap.set(cv2.CAP_PROP_EXPOSURE, -3)

# Wait for camera to adjust
import time
print("\n⏳ Waiting 2 seconds for camera to adjust...")
time.sleep(2)

# Verify new settings
print("\nNew settings:")
print(f"  Auto Exposure: {cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)}")
print(f"  Brightness: {cap.get(cv2.CAP_PROP_BRIGHTNESS)}")
print(f"  Contrast: {cap.get(cv2.CAP_PROP_CONTRAST)}")
print(f"  Exposure: {cap.get(cv2.CAP_PROP_EXPOSURE)}")
print(f"  Gain: {cap.get(cv2.CAP_PROP_GAIN)}")

# Capture frame
print("\nCapturing test frame...")
ret, frame = cap.read()

if not ret or frame is None:
    print("❌ Cannot read frame")
    cap.release()
    exit(1)

# Calculate brightness
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
avg_brightness = np.mean(gray)

print(f"✅ Frame captured!")
print(f"📊 Brightness: {avg_brightness:.1f}")

# Save frame
cv2.imwrite("camera_after_fix.jpg", frame)
print(f"💾 Saved: camera_after_fix.jpg")

# If still too dark, try manual exposure
if avg_brightness < 30:
    print("\n⚠️ Still too dark! Trying MANUAL exposure mode...")
    
    # Switch to manual exposure
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Manual mode
    cap.set(cv2.CAP_PROP_EXPOSURE, -1)  # Even brighter
    
    time.sleep(2)
    
    ret, frame = cap.read()
    if ret and frame is not None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray)
        
        print(f"📊 New brightness: {avg_brightness:.1f}")
        cv2.imwrite("camera_manual_exposure.jpg", frame)
        print(f"💾 Saved: camera_manual_exposure.jpg")

# Test detection if brightness is good
if avg_brightness > 30:
    print(f"\n✅ BRIGHTNESS OK! ({avg_brightness:.1f} > 30)")
    print("Testing detection...")
    
    from ultralytics import YOLO
    model = YOLO('app/model/best.pt')
    
    results = model.predict(frame, conf=0.25, verbose=False)
    detections = len(results[0].boxes)
    
    print(f"🎯 Detections: {detections}")
    
    if detections > 0:
        annotated = results[0].plot()
        cv2.imwrite("camera_detection_result.jpg", annotated)
        print(f"💾 Saved annotated: camera_detection_result.jpg")
else:
    print(f"\n❌ STILL TOO DARK! ({avg_brightness:.1f})")
    print("\nPOSSIBLE CAUSES:")
    print("  1. Camera lens is physically covered/blocked")
    print("  2. Camera driver not responding to settings")
    print("  3. Wrong camera selected (try index 0)")
    print("  4. Camera hardware issue")
    print("\nTRY:")
    print("  - Check camera physically")
    print("  - Try: python fix_camera_exposure.py --index 0")
    print("  - Test camera in Windows Camera app first")

cap.release()
print("\n" + "="*70)
