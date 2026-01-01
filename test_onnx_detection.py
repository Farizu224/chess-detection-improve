"""
Quick test script to verify ONNX inference is working correctly
Run this before starting the full app to diagnose detection issues
"""

import cv2
import numpy as np
from app.onnx_engine import ONNXInferenceEngine

print("="*60)
print("ONNX DETECTION TEST")
print("="*60)

# Test 1: Load ONNX engine
print("\n1. Testing ONNX Engine Initialization...")
try:
    engine = ONNXInferenceEngine(
        onnx_path='app/model/best.onnx',
        pytorch_model='app/model/best.pt',
        input_size=736  # EXPLICITLY set to 736
    )
    print(f"   ✅ Engine loaded")
    print(f"   Input size: {engine.input_size}")
    print(f"   Using ONNX: {engine.use_onnx}")
    print(f"   Has session: {engine.session is not None}")
    print(f"   Has PyTorch fallback: {engine.pytorch_model is not None}")
except Exception as e:
    print(f"   ❌ Failed to load engine: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 2: Create test image (must be 736x736 to match model)
print("\n2. Creating test image (736x736 with random noise)...")
test_image = np.random.randint(0, 255, (736, 736, 3), dtype=np.uint8)
print(f"   Image shape: {test_image.shape}, dtype: {test_image.dtype}")

# Test 3: Run inference
print("\n3. Running inference...")
try:
    results = engine.infer(test_image, conf_threshold=0.1)
    print(f"   ✅ Inference completed")
    print(f"   Result type: {type(results)}")
    print(f"   Result length: {len(results)}")
    
    if len(results) > 0:
        result = results[0]
        print(f"   Result[0] type: {type(result)}")
        print(f"   Has 'boxes' attribute: {hasattr(result, 'boxes')}")
        
        if hasattr(result, 'boxes'):
            boxes = result.boxes
            print(f"   Boxes type: {type(boxes)}")
            print(f"   Number of detections: {len(boxes)}")
            
            # Test plot method
            if hasattr(result, 'plot'):
                print(f"   Has 'plot' method: True")
                try:
                    plotted = result.plot()
                    print(f"   ✅ Plot method works, output shape: {plotted.shape}")
                except Exception as e:
                    print(f"   ❌ Plot method failed: {e}")
            else:
                print(f"   ⚠️ No 'plot' method found")
        else:
            print(f"   ⚠️ Result has no 'boxes' attribute")
            print(f"   Available attributes: {dir(result)}")
    
except Exception as e:
    print(f"   ❌ Inference failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 4: Performance test
print("\n4. Performance Test (10 inferences)...")
import time
times = []
for i in range(10):
    start = time.time()
    results = engine.infer(test_image, conf_threshold=0.1)
    elapsed = time.time() - start
    times.append(elapsed)
    if i == 0:
        print(f"   First inference: {elapsed*1000:.1f}ms")

avg_time = np.mean(times[1:])  # Exclude first (warmup)
print(f"   Average inference time: {avg_time*1000:.1f}ms")
print(f"   Estimated FPS: {1.0/avg_time:.1f}")

# Test 5: Stats
print("\n5. Engine Statistics:")
stats = engine.get_stats()
for key, value in stats.items():
    print(f"   {key}: {value}")

print("\n" + "="*60)
print("TEST COMPLETED SUCCESSFULLY")
print("="*60)
print("\nNext steps:")
print("1. If all tests passed, ONNX engine is working correctly")
print("2. If detections are still not working in main app:")
print("   - Check camera lighting (image too dark?)")
print("   - Check if pieces are in frame and visible")
print("   - Try CLAHE mode (press 'M' in OpenCV window)")
print("   - Lower confidence threshold further (currently 0.1)")
