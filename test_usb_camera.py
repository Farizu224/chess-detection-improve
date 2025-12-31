"""
Test detection langsung dengan USB camera (index 1)
"""
import cv2
import sys
import os

# Add app directory to path
sys.path.insert(0, 'app')

from chess_detection import ChessDetectionService

print("="*70)
print("DIRECT DETECTION TEST - USB CAMERA")
print("="*70)

print("\n[1/3] Initializing Chess Detection Service...")
try:
    detector = ChessDetectionService(
        model_path='app/model/best.pt',
        use_onnx=False
    )
    print("   ✅ Detection service initialized")
except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[2/3] Camera Selection:")
print("   Available cameras:")
print("   - Camera 0: Built-in laptop camera")
print("   - Camera 1: USB camera ⭐")

camera_index = 1  # USB camera
print(f"\n   Using Camera {camera_index} (USB)")

print("\n[3/3] Starting detection...")
print("   This will open an OpenCV window")
print("   Press 'Q' in the window to stop")
print("="*70)

try:
    # Set camera index
    detector.camera_index = camera_index
    
    # Start detection
    success = detector.start_opencv_detection(
        camera_index=camera_index,
        mode='raw',
        show_bbox=True
    )
    
    if success:
        print("\n✅ Detection started successfully!")
        print("   Look for the OpenCV window...")
        print("   Press Ctrl+C here to stop if window doesn't appear")
        
        # Keep script running
        import time
        while detector.detection_active:
            time.sleep(1)
    else:
        print("\n❌ Failed to start detection")
        
except KeyboardInterrupt:
    print("\n\n⏹️  Interrupted by user")
    detector.stop_opencv_detection()
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
finally:
    print("\n🔄 Stopping detection...")
    detector.stop_opencv_detection()
    print("✅ Done")
