import cv2
import numpy as np
import time

print("Testing camera exposure fix with frame warm-up...")

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("❌ Cannot open camera")
    exit(1)

print("✅ Camera opened")

# Set exposure settings
print("Setting manual exposure mode...")
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Manual
cap.set(cv2.CAP_PROP_EXPOSURE, -1)
cap.set(cv2.CAP_PROP_BRIGHTNESS, 255)
cap.set(cv2.CAP_PROP_GAIN, 100)

print("Waiting 2 seconds...")
time.sleep(2)

# Try reading multiple frames (camera might need warm-up)
print("\nReading test frames:")
for i in range(5):
    ret, frame = cap.read()
    if ret and frame is not None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        print(f"  Frame {i+1}: brightness = {brightness:.1f}")
        
        if i == 4:  # Save last frame
            cv2.imwrite("quick_test_frame.jpg", frame)
            if brightness > 30:
                print(f"\n✅✅ BRIGHTNESS OK: {brightness:.1f}")
            else:
                print(f"\n❌ Still dark: {brightness:.1f}")
    else:
        print(f"  Frame {i+1}: Failed to read")
    
    time.sleep(0.2)

cap.release()
