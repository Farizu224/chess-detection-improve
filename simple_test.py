import cv2
import numpy as np

# Test camera 1 (USB)
print("Opening USB camera (index 1)...")
cap = cv2.VideoCapture(1)
print("Camera opened:", cap.isOpened())

if cap.isOpened():
    ret, frame = cap.read()
    print("Can read frame:", ret)
    if ret:
        print("Frame shape:", frame.shape)
    cap.release()

# Test window
print("\nTesting window...")
test = np.zeros((480, 640, 3), dtype=np.uint8)
cv2.putText(test, "Test Window - Press Q", (50, 240), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

cv2.namedWindow('Test', cv2.WINDOW_NORMAL)
cv2.imshow('Test', test)
print("Window created! Press Q to close...")
cv2.waitKey(0)
cv2.destroyAllWindows()
print("Done!")
