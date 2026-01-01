"""Quick camera test dengan display"""
import cv2
import sys

camera_index = int(sys.argv[1]) if len(sys.argv) > 1 else 1

print(f"Testing camera {camera_index}...")
cap = cv2.VideoCapture(camera_index)

if not cap.isOpened():
    print(f"❌ Cannot open camera {camera_index}")
    sys.exit(1)

ret, frame = cap.read()
if not ret:
    print(f"❌ Cannot read frame")
    cap.release()
    sys.exit(1)

print(f"✅ Camera {camera_index} OK - Resolution: {frame.shape[1]}x{frame.shape[0]}")
print(f"📹 Press Q to quit...")

cv2.namedWindow(f"Camera {camera_index} Test", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    cv2.putText(frame, f"Camera {camera_index} - Press Q to quit", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow(f"Camera {camera_index} Test", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Test complete")
