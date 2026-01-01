"""Test detection quality and see what's being detected"""
import cv2
import numpy as np
from ultralytics import YOLO

print("=" * 70)
print("DETECTION QUALITY DIAGNOSTIC")
print("=" * 70)

# Load model
model = YOLO('app/model/best.pt')
print(f"\n1. Model loaded: {len(model.names)} classes")

# Open camera
print("\n2. Opening camera 1...")
cap = cv2.VideoCapture(1, cv2.CAP_ANY)
if not cap.isOpened():
    print("❌ Camera failed")
    exit(1)

ret, frame = cap.read()
if not ret:
    print("❌ Can't read frame")
    exit(1)

# Preprocess (same as app)
h, w = frame.shape[:2]
min_dim = min(h, w)
start_x = (w - min_dim) // 2
start_y = (h - min_dim) // 2
cropped = frame[start_y:start_y+min_dim, start_x:start_x+min_dim]
resized = cv2.resize(cropped, (736, 736))

print(f"\n3. Frame info:")
print(f"   Original: {frame.shape}")
print(f"   Cropped: {cropped.shape}")
print(f"   Resized: {resized.shape}")
print(f"   Brightness: {np.mean(resized):.1f}")

# Run detection at multiple confidence levels
print(f"\n4. Testing detection at multiple confidence levels...")
for conf in [0.25, 0.15, 0.10, 0.05, 0.01]:
    results = model.predict(resized, conf=conf, verbose=False)
    detections = results[0].boxes
    print(f"   conf={conf:.2f} → {len(detections)} detections")
    
    if len(detections) > 0 and conf == 0.01:
        # Save results from lowest confidence
        results_to_show = results

# Use results from conf=0.01 for detailed analysis
results = model.predict(resized, conf=0.01, verbose=False)
detections = results[0].boxes

print(f"\n5. DETAILED RESULTS (conf=0.01): {len(detections)} detections")
if len(detections) == 0:
    print("   ⚠️ NO DETECTIONS - Check lighting and camera position!")
else:
    print("\n   Detected objects:")
    for i, box in enumerate(detections):
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        conf = float(box.conf[0])
        coords = box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = coords
        
        # Calculate center and size
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1
        
        print(f"\n   [{i+1}] {cls_name}")
        print(f"       Confidence: {conf:.3f}")
        print(f"       Position: ({cx:.0f}, {cy:.0f})")
        print(f"       Size: {width:.0f}x{height:.0f}")
        print(f"       Coords: ({x1:.0f},{y1:.0f}) to ({x2:.0f},{y2:.0f})")
        
        # Check if detection is at image edge (likely false positive)
        if x1 < 50 or y1 < 50 or x2 > 686 or y2 > 686:
            print(f"       ⚠️ WARNING: Detection at image edge (likely FALSE POSITIVE)")

# Save annotated image
annotated = results[0].plot()
cv2.imwrite('detection_diagnostic.jpg', annotated)
print(f"\n6. Saved annotated image: detection_diagnostic.jpg")
print("   Open this file to see what's being detected!")

cap.release()
print("\n" + "=" * 70)
