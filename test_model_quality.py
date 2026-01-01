"""Test model quality - check for false positives"""
import cv2
import numpy as np
from ultralytics import YOLO

print("=" * 70)
print("MODEL QUALITY TEST - FALSE POSITIVE CHECK")
print("=" * 70)

# Load model
print("\n1. Loading model...")
model = YOLO('app/model/best.pt')
print(f"✅ Model loaded: {len(model.names)} classes")
print(f"   Classes: {list(model.names.values())}")

# Test 1: Empty black image (should detect NOTHING)
print("\n2. TEST 1: Black image (should detect 0)")
black_img = np.zeros((736, 736, 3), dtype=np.uint8)
results = model.predict(black_img, conf=0.05, verbose=False)
detections = len(results[0].boxes)
if detections > 0:
    print(f"   ❌ FAIL: Detected {detections} objects in black image!")
    for box in results[0].boxes[:3]:
        cls_name = model.names[int(box.cls[0])]
        conf = float(box.conf[0])
        print(f"      - {cls_name} (conf={conf:.3f})")
else:
    print(f"   ✅ PASS: 0 detections")

# Test 2: Random noise (should detect NOTHING or very few)
print("\n3. TEST 2: Random noise (should detect 0-1)")
noise_img = np.random.randint(0, 256, (736, 736, 3), dtype=np.uint8)
results = model.predict(noise_img, conf=0.05, verbose=False)
detections = len(results[0].boxes)
if detections > 2:
    print(f"   ❌ FAIL: Detected {detections} objects in noise!")
    for box in results[0].boxes[:3]:
        cls_name = model.names[int(box.cls[0])]
        conf = float(box.conf[0])
        print(f"      - {cls_name} (conf={conf:.3f})")
else:
    print(f"   ✅ PASS: {detections} detections (acceptable)")

# Test 3: White image (should detect NOTHING)
print("\n4. TEST 3: White image (should detect 0)")
white_img = np.ones((736, 736, 3), dtype=np.uint8) * 255
results = model.predict(white_img, conf=0.05, verbose=False)
detections = len(results[0].boxes)
if detections > 0:
    print(f"   ❌ FAIL: Detected {detections} objects in white image!")
    for box in results[0].boxes[:3]:
        cls_name = model.names[int(box.cls[0])]
        conf = float(box.conf[0])
        print(f"      - {cls_name} (conf={conf:.3f})")
else:
    print(f"   ✅ PASS: 0 detections")

# Test 4: Real camera feed WITHOUT chess pieces
print("\n5. TEST 4: Live camera WITHOUT chess pieces")
print("   Opening camera 1...")
cap = cv2.VideoCapture(1, cv2.CAP_ANY)
if cap.isOpened():
    ret, frame = cap.read()
    if ret:
        # Preprocess
        h, w = frame.shape[:2]
        min_dim = min(h, w)
        start_x = (w - min_dim) // 2
        start_y = (h - min_dim) // 2
        cropped = frame[start_y:start_y+min_dim, start_x:start_x+min_dim]
        resized = cv2.resize(cropped, (736, 736))
        
        brightness = np.mean(resized)
        print(f"   Frame brightness: {brightness:.1f}")
        
        # Test at different confidence levels
        print(f"\n   Testing at different confidence thresholds:")
        for conf in [0.25, 0.15, 0.10, 0.05]:
            results = model.predict(resized, conf=conf, verbose=False)
            num_det = len(results[0].boxes)
            print(f"   conf={conf:.2f} → {num_det} detections", end="")
            
            if num_det > 0:
                print(" (FALSE POSITIVES!)")
                for box in results[0].boxes[:2]:
                    cls_name = model.names[int(box.cls[0])]
                    box_conf = float(box.conf[0])
                    print(f"      - {cls_name} (conf={box_conf:.3f})")
            else:
                print(" ✓")
        
        # Save image for inspection
        cv2.imwrite('test_camera_frame.jpg', resized)
        print(f"\n   📸 Saved frame to: test_camera_frame.jpg")
    cap.release()
else:
    print("   ⚠️ Could not open camera")

print("\n" + "=" * 70)
print("SUMMARY:")
print("=" * 70)
print("If model detects objects in black/white/empty images,")
print("it has HIGH FALSE POSITIVE RATE and needs:")
print("  1. Higher confidence threshold (0.25 minimum)")
print("  2. Model retraining with better dataset")
print("  3. More negative samples during training")
print("=" * 70)
