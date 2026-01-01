"""
Test camera dengan minimal code - isolate masalahnya
"""
import cv2
import time

print("=" * 60)
print("BASIC CAMERA TEST")
print("=" * 60)

camera_index = 1
backend = cv2.CAP_DSHOW

print(f"\n1. Opening camera {camera_index} with DirectShow...")
cap = cv2.VideoCapture(camera_index, backend)

if not cap.isOpened():
    print(f"❌ Failed to open camera {camera_index}")
    exit(1)

print(f"✅ Camera opened successfully")

# Set properties
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

print(f"\n2. Reading frames...")
cv2.namedWindow('Camera Test', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Camera Test', 640, 480)

frame_count = 0
start_time = time.time()

try:
    while True:
        ret, frame = cap.read()
        
        if not ret or frame is None:
            print(f"❌ Failed to read frame {frame_count}")
            break
        
        frame_count += 1
        
        # Add text
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Camera Test', frame)
        
        # Check every 30 frames
        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            fps = frame_count / elapsed
            print(f"Frame {frame_count} | FPS: {fps:.1f}")
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\nQuit requested")
            break
        
        if frame_count >= 100:
            print(f"\n✅ Test complete: read {frame_count} frames successfully")
            break

except KeyboardInterrupt:
    print("\nInterrupted by user")
except Exception as e:
    print(f"\n❌ Exception: {e}")
    import traceback
    traceback.print_exc()
finally:
    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Cleanup complete")

print("=" * 60)
