"""
Quick test: Open USB camera dan tampilkan window sederhana
"""
import cv2
import time

print("="*70)
print("QUICK USB CAMERA TEST")
print("="*70)

camera_index = 1  # USB camera
print(f"\n[1/2] Opening camera {camera_index} (USB)...")

cap = cv2.VideoCapture(camera_index)

if not cap.isOpened():
    print(f"   ❌ Cannot open camera {camera_index}")
    print("\n   Try camera 0 instead? (Y/n): ", end="")
    # Use camera 0 as fallback
    camera_index = 0
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"   ❌ Cannot open camera {camera_index} either")
        exit(1)

# Test read
ret, frame = cap.read()
if not ret or frame is None:
    print(f"   ❌ Camera opened but cannot read frame")
    cap.release()
    exit(1)

h, w = frame.shape[:2]
print(f"   ✅ Camera {camera_index} opened successfully!")
print(f"   Resolution: {w}x{h}")

print(f"\n[2/2] Creating window and showing video...")
print(f"   Press 'Q' to quit\n")

cv2.namedWindow('USB Camera Test', cv2.WINDOW_NORMAL)
cv2.resizeWindow('USB Camera Test', 640, 480)

frame_count = 0
start_time = time.time()

try:
    while True:
        ret, frame = cap.read()
        
        if not ret or frame is None:
            print("   ⚠️ Failed to read frame")
            break
        
        # Add text overlay
        frame_count += 1
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0
        
        cv2.putText(frame, f"Camera {camera_index} - USB Test", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Press Q to quit", (10, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.imshow('USB Camera Test', frame)
        
        # FPS display every 30 frames
        if frame_count % 30 == 0:
            print(f"   📹 FPS: {fps:.1f}")
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            print("\n   ⏹️  Quit requested")
            break
            
except KeyboardInterrupt:
    print("\n   ⏹️  Interrupted by user")
except Exception as e:
    print(f"\n   ❌ Error: {e}")
    import traceback
    traceback.print_exc()
finally:
    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Camera closed")
    print("="*70)
