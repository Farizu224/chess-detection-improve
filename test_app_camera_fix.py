"""
Quick test of camera with exposure fix integrated
"""
import sys
sys.path.insert(0, r'd:\chess-detection-improve\chess-detection-improve')

from app.chess_detection import ChessDetectionService
import time

print("="*70)
print("TESTING CAMERA WITH EXPOSURE FIX")
print("="*70)

service = ChessDetectionService()

# Set camera index to 1 (your USB camera)
service.camera_index = 1

print("\n1. Initializing detection service...")
print(f"   Camera: {service.camera_index}")
print(f"   Mode: {service.detection_mode}")

print("\n2. Starting camera thread...")
print("   This will open camera with manual exposure fix")
print("   Wait ~3 seconds...\n")

# Start in background
import threading
thread = threading.Thread(target=service.start_opencv_detection, daemon=True)
thread.start()

# Wait for initialization
time.sleep(3)

print("\n3. Checking if camera is working...")
if service.cap and service.cap.isOpened():
    print("   ✅ Camera opened successfully!")
    
    # Try to read frame
    ret, frame = service.cap.read()
    if ret and frame is not None:
        import cv2
        import numpy as np
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        
        print(f"   ✅ Frame captured!")
        print(f"   📊 Brightness: {brightness:.1f}")
        
        if brightness > 30:
            print(f"   ✅✅ BRIGHTNESS IS GOOD! ({brightness:.1f} > 30)")
            
            # Save test frame
            cv2.imwrite("app_camera_test.jpg", frame)
            print(f"   💾 Saved: app_camera_test.jpg")
            
            print("\n4. ✅ CAMERA FIX SUCCESSFUL!")
            print("   Your app should now work properly.")
            print("\n   To run full app:")
            print("   python -m app.app")
        else:
            print(f"   ⚠️ Brightness still low: {brightness:.1f}")
    else:
        print("   ❌ Cannot read frame")
else:
    print("   ❌ Camera not opened")

# Cleanup
print("\n5. Cleaning up...")
service.stop_detection()
time.sleep(1)

print("="*70)
