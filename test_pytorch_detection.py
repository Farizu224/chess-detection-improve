"""Test PyTorch detection to compare with ONNX"""
import cv2
import numpy as np
from ultralytics import YOLO

print("=" * 60)
print("PYTORCH DETECTION TEST")
print("=" * 60)

# Load PyTorch model
print("\n1. Loading PyTorch model...")
model = YOLO('app/model/best.pt')
print(f"✅ PyTorch model loaded")
print(f"   Model type: {type(model)}")
print(f"   Classes: {len(model.names)} ({list(model.names.values())[:5]}...)")

# Open camera
print("\n2. Opening camera 1...")
cap = cv2.VideoCapture(1, cv2.CAP_ANY)
if not cap.isOpened():
    print("❌ Could not open camera 1")
    exit(1)

ret, frame = cap.read()
if not ret:
    print("❌ Could not read frame")
    exit(1)

print(f"✅ Camera opened")
print(f"   Frame shape: {frame.shape}")
print(f"   Frame brightness: {np.mean(frame):.1f}")

# Crop to square
print("\n3. Preprocessing frame...")
h, w = frame.shape[:2]
min_dim = min(h, w)
start_x = (w - min_dim) // 2
start_y = (h - min_dim) // 2
cropped = frame[start_y:start_y+min_dim, start_x:start_x+min_dim]
resized = cv2.resize(cropped, (736, 736))
print(f"   Cropped to: {cropped.shape}")
print(f"   Resized to: {resized.shape}")

# Run inference with various confidence levels
print("\n4. Testing PyTorch inference...")
for conf in [0.25, 0.15, 0.1, 0.05, 0.01]:
    results = model.predict(resized, conf=conf, verbose=False)
    num_detections = len(results[0].boxes)
    print(f"   Conf={conf:.2f} → {num_detections} detections")
    
    if num_detections > 0:
        print(f"   ✅ DETECTIONS FOUND at conf={conf}!")
        for i, box in enumerate(results[0].boxes[:3]):  # Show first 3
            cls = int(box.cls[0])
            confidence = float(box.conf[0])
            coords = box.xyxy[0].cpu().numpy()
            print(f"      Box {i+1}: class={model.names[cls]}, conf={confidence:.3f}, coords={coords}")
        break
else:
    print(f"   ❌ NO DETECTIONS even at conf=0.01")

cap.release()
print("\n" + "=" * 60)
print("Test complete!")
print("=" * 60)
