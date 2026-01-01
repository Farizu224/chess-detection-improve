"""
FINAL COMPREHENSIVE TEST
Tests camera with exposure fix + detection
"""
import cv2
import numpy as np
import time
from ultralytics import YOLO

print("="*70)
print("FINAL COMPREHENSIVE TEST - Camera + Detection")
print("="*70)

# Load model
print("\n1. Loading YOLO model...")
model = YOLO('app/model/best.pt')
print("   ✅ Model loaded")

# Open camera with fix
print("\n2. Opening camera with manual exposure fix...")
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("   ❌ Cannot open camera")
    exit(1)

print("   ✅ Camera opened")

# Apply exposure settings
print("\n3. Applying manual exposure settings...")
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Manual mode
cap.set(cv2.CAP_PROP_EXPOSURE, -1)         # Brighter
cap.set(cv2.CAP_PROP_BRIGHTNESS, 255)      # Max
cap.set(cv2.CAP_PROP_GAIN, 100)            # Boost

print("   ⏳ Waiting 1.5s for camera adjustment...")
time.sleep(1.5)

# Discard warm-up frame
print("\n4. Discarding warm-up frame...")
ret, _ = cap.read()
print("   ✅ Warm-up frame discarded")

# Capture actual frame
print("\n5. Capturing frame for detection...")
ret, frame = cap.read()

if not ret or frame is None:
    print("   ❌ Cannot capture frame")
    cap.release()
    exit(1)

# Calculate brightness
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
brightness = np.mean(gray)

print(f"   ✅ Frame captured")
print(f"   📊 Brightness: {brightness:.1f}")

# Save raw frame
cv2.imwrite("final_test_raw.jpg", frame)
print(f"   💾 Saved: final_test_raw.jpg")

# Check brightness
if brightness < 30:
    print(f"\n   ⚠️ WARNING: Brightness too low ({brightness:.1f})")
    print("   Detection may not work well")
elif brightness > 200:
    print(f"\n   ⚠️ WARNING: Brightness too high ({brightness:.1f})")
    print("   Image may be overexposed")
else:
    print(f"\n   ✅ Brightness is GOOD ({brightness:.1f} in range 30-200)")

# Run detection
print("\n6. Running YOLO detection...")
results = model.predict(frame, conf=0.25, verbose=False)
detections = results[0].boxes

print(f"   🎯 Detections: {len(detections)}")

if len(detections) > 0:
    print("\n   Detected pieces:")
    for i, box in enumerate(detections):
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        class_name = model.names[cls]
        
        print(f"     {i+1}. {class_name} (conf={conf:.2f}) at [{int(x1)},{int(y1)}] to [{int(x2)},{int(y2)}]")
    
    # Save annotated frame
    annotated = results[0].plot()
    cv2.imwrite("final_test_annotated.jpg", annotated)
    print(f"\n   💾 Saved annotated: final_test_annotated.jpg")
    print(f"   ✅✅ DETECTION WORKING!")
    
else:
    print("\n   ⚠️ No detections found")
    print("   Possible reasons:")
    print("     - No chess pieces in frame")
    print("     - Pieces too small")
    print("     - Confidence threshold too high (try 0.15)")
    print("     - Lighting issues")

# Cleanup
cap.release()

print("\n" + "="*70)
print("TEST COMPLETE")
print("="*70)

if brightness > 30 and len(detections) > 0:
    print("\n✅✅✅ ALL SYSTEMS WORKING!")
    print("\nYour app is ready to run:")
    print("  python -m app.app")
elif brightness > 30:
    print("\n✅ Camera working but no detections")
    print("Make sure chess piece is visible in frame")
else:
    print("\n❌ Camera brightness issue persists")
    print("Check camera physically and try different USB port")

print("="*70)
