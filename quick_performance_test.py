"""
Quick Performance Test - Test inference speed with ONNX
"""
import cv2
import numpy as np
import time
from ultralytics import YOLO

print("="*70)
print("PERFORMANCE TEST - ONNX vs PyTorch")
print("="*70)

# Open camera quickly
print("\n1. Opening camera...")
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("❌ Cannot open camera")
    exit(1)

# Apply exposure settings
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
cap.set(cv2.CAP_PROP_EXPOSURE, -1)
cap.set(cv2.CAP_PROP_BRIGHTNESS, 255)
cap.set(cv2.CAP_PROP_GAIN, 100)
time.sleep(1.5)

ret, _ = cap.read()  # Discard warm-up
ret, frame = cap.read()

if not ret:
    print("❌ Cannot capture frame")
    cap.release()
    exit(1)

cap.release()
print("✅ Frame captured\n")

# Test PyTorch
print("2. Testing PyTorch inference...")
model_pt = YOLO('app/model/best.pt')

times_pt = []
for i in range(3):
    start = time.time()
    results = model_pt.predict(frame, conf=0.45, verbose=False)
    elapsed = (time.time() - start) * 1000
    times_pt.append(elapsed)
    print(f"   Run {i+1}: {elapsed:.1f}ms")

avg_pt = np.mean(times_pt)
detections_pt = len(results[0].boxes)

print(f"\n   📊 PyTorch Average: {avg_pt:.1f}ms")
print(f"   🎯 Detections: {detections_pt}\n")

# Test ONNX
print("3. Testing ONNX inference...")
try:
    import sys
    sys.path.insert(0, 'app')
    from onnx_engine import ONNXInferenceEngine
    
    engine = ONNXInferenceEngine('app/model/best.onnx', 'app/model/best.pt', input_size=736)
    
    times_onnx = []
    for i in range(3):
        start = time.time()
        results = engine.infer(frame, conf_threshold=0.45)
        elapsed = (time.time() - start) * 1000
        times_onnx.append(elapsed)
        print(f"   Run {i+1}: {elapsed:.1f}ms")
    
    avg_onnx = np.mean(times_onnx)
    detections_onnx = len(results[0].boxes)
    
    print(f"\n   📊 ONNX Average: {avg_onnx:.1f}ms")
    print(f"   🎯 Detections: {detections_onnx}\n")
    
    # Comparison
    speedup = avg_pt / avg_onnx
    print("="*70)
    print("COMPARISON RESULTS")
    print("="*70)
    print(f"\nPyTorch:  {avg_pt:.1f}ms  |  {1000/avg_pt:.1f} FPS")
    print(f"ONNX:     {avg_onnx:.1f}ms  |  {1000/avg_onnx:.1f} FPS")
    print(f"\n🚀 SPEEDUP: {speedup:.2f}x faster with ONNX!")
    
    if avg_onnx < 150:
        print(f"\n✅✅ EXCELLENT! Inference time < 150ms")
        print(f"   Expected FPS: {1000/avg_onnx:.0f} (very smooth!)")
    elif avg_onnx < 200:
        print(f"\n✅ GOOD! Inference time < 200ms")
        print(f"   Expected FPS: {1000/avg_onnx:.0f} (smooth)")
    else:
        print(f"\n⚠️ Slow inference: {avg_onnx:.1f}ms")
        print(f"   Expected FPS: {1000/avg_onnx:.0f} (may be choppy)")
    
except Exception as e:
    print(f"❌ ONNX test failed: {e}")
    print("\nONNX might not be available. Using PyTorch only.")
    print(f"PyTorch performance: {avg_pt:.1f}ms | {1000/avg_pt:.1f} FPS")

print("\n" + "="*70)
