"""Advanced camera diagnostic with brightness adjustment"""
import cv2
import numpy as np
from ultralytics import YOLO

print("=" * 70)
print("ADVANCED CAMERA DIAGNOSTIC")
print("=" * 70)

# Load model
model = YOLO('app/model/best.pt')

# Open camera with DirectShow (more stable than MSMF)
print("\n1. Opening camera with DirectShow...")
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("   ⚠️ DirectShow failed, trying CAP_ANY...")
    cap = cv2.VideoCapture(1, cv2.CAP_ANY)
    if not cap.isOpened():
        print("❌ Cannot open camera")
        exit(1)

print("   ✅ Camera opened")

# Get current settings (don't try to set, just read)
print("\n2. Current camera settings:")
print(f"   Brightness: {cap.get(cv2.CAP_PROP_BRIGHTNESS)}")
print(f"   Exposure: {cap.get(cv2.CAP_PROP_EXPOSURE)}")
print(f"   Gain: {cap.get(cv2.CAP_PROP_GAIN)}")

# Capture frame
print("\n3. Capturing frame...")
ret, frame = cap.read()
if not ret:
    print("❌ Cannot read frame")
    exit(1)
print("   ✅ Frame captured")

# Preprocess
h, w = frame.shape[:2]
min_dim = min(h, w)
start_x = (w - min_dim) // 2
start_y = (h - min_dim) // 2
cropped = frame[start_y:start_y+min_dim, start_x:start_x+min_dim]
resized = cv2.resize(cropped, (736, 736))

brightness_raw = np.mean(resized)
print(f"\n4. Raw frame brightness: {brightness_raw:.1f}")

# Save raw image
cv2.imwrite('camera_raw.jpg', resized)
print(f"   Saved: camera_raw.jpg")

# Try histogram equalization to boost brightness
print(f"\n5. Applying brightness enhancements...")

# Method 1: Simple brightness boost
brightened = cv2.convertScaleAbs(resized, alpha=3.0, beta=50)
brightness_1 = np.mean(brightened)
cv2.imwrite('camera_brightened.jpg', brightened)
print(f"   Method 1 (boost): {brightness_1:.1f} - saved to camera_brightened.jpg")

# Method 2: CLAHE (Contrast Limited Adaptive Histogram Equalization)
lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
l_clahe = clahe.apply(l)
enhanced = cv2.merge([l_clahe, a, b])
enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
brightness_2 = np.mean(enhanced)
cv2.imwrite('camera_enhanced.jpg', enhanced)
print(f"   Method 2 (CLAHE): {brightness_2:.1f} - saved to camera_enhanced.jpg")

# Test detection on all versions
print(f"\n6. Testing detection on different brightness levels...")
test_images = [
    ("Raw", resized, brightness_raw),
    ("Brightened", brightened, brightness_1),
    ("Enhanced (CLAHE)", enhanced, brightness_2)
]

best_result = None
best_count = 0

for name, img, bright in test_images:
    results = model.predict(img, conf=0.25, verbose=False)
    detections = len(results[0].boxes)
    print(f"\n   {name} (brightness={bright:.1f}):")
    print(f"      Detections: {detections}")
    
    if detections > 0:
        for i, box in enumerate(results[0].boxes[:3]):
            cls_name = model.names[int(box.cls[0])]
            conf = float(box.conf[0])
            print(f"      [{i+1}] {cls_name} (conf={conf:.3f})")
        
        if detections > best_count:
            best_count = detections
            best_result = (name, img, results)

# Save best result
if best_result:
    name, img, results = best_result
    annotated = results[0].plot()
    cv2.imwrite('detection_best.jpg', annotated)
    print(f"\n7. ✅ BEST RESULT: {name} with {best_count} detections")
    print(f"   Saved annotated image: detection_best.jpg")
else:
    print(f"\n7. ❌ NO DETECTIONS in any brightness level!")
    print(f"\n   RECOMMENDATIONS:")
    print(f"   1. Turn on room lights / desk lamp")
    print(f"   2. Open DroidCam Settings → increase Brightness/Exposure")
    print(f"   3. Move camera closer to light source")
    print(f"   4. Check if camera lens is covered/blocked")

cap.release()
print("\n" + "=" * 70)
