"""
Test SPEED baseline - Compare laptop webcam vs DroidCam
Goal: Measure actual FPS improvement vs original
"""
import cv2
import time
import numpy as np

print("\n" + "="*70)
print("SPEED BASELINE TEST - ONNX vs Original")
print("="*70)

# Test with laptop webcam first (more reliable)
CAMERA_INDEX = 1  # Laptop webcam

print(f"\n🎥 Testing Camera {CAMERA_INDEX} (Laptop Webcam)...")
print("   This will measure:")
print("   - Camera frame rate (valid frames per second)")
print("   - ONNX inference speed")
print("   - Total processing FPS")
print("\nPress 'q' to quit\n")

# Open camera
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
if not cap.isOpened():
    print(f"❌ Failed to open camera {CAMERA_INDEX}")
    exit(1)

# Get camera info
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_cam = cap.get(cv2.CAP_PROP_FPS)
print(f"✅ Camera opened: {width}x{height} @ {fps_cam:.1f} FPS\n")

# Load ONNX model with DirectML
import onnxruntime as ort

onnx_path = "app/model/best.onnx"
print(f"📦 Loading ONNX model: {onnx_path}")

# Use DirectML provider for GPU acceleration
available_providers = ort.get_available_providers()
print(f"   Available providers: {available_providers}")

if 'DmlExecutionProvider' in available_providers:
    providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
    print(f"   🚀 Using DirectML (GPU)")
else:
    providers = ['CPUExecutionProvider']
    print(f"   ⚠️ Using CPU only")

session = ort.InferenceSession(onnx_path, providers=providers)
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
print(f"✅ ONNX loaded: {input_name} {input_shape}\n")

# Tracking
frame_count = 0
inference_count = 0
start_time = time.time()
valid_frames = 0
failed_frames = 0

inference_times = []
total_times = []

print("🚀 Starting test (press 'q' to stop)...\n")
print("-" * 70)

while True:
    frame_start = time.time()
    
    # Read frame
    ret, frame = cap.read()
    
    if not ret or frame is None:
        failed_frames += 1
        continue
    
    valid_frames += 1
    frame_count += 1
    
    # Preprocess for ONNX (resize to 736x736, normalize)
    input_img = cv2.resize(frame, (736, 736), interpolation=cv2.INTER_LINEAR)
    input_img = input_img.transpose(2, 0, 1).astype(np.float32) / 255.0
    input_img = np.expand_dims(input_img, axis=0)
    
    # Run inference
    inference_start = time.time()
    outputs = session.run(None, {input_name: input_img})
    inference_time = (time.time() - inference_start) * 1000
    inference_times.append(inference_time)
    inference_count += 1
    
    # Total frame processing time
    total_time = (time.time() - frame_start) * 1000
    total_times.append(total_time)
    
    # Print stats every 30 frames
    if frame_count % 30 == 0:
        elapsed = time.time() - start_time
        fps = valid_frames / elapsed
        avg_inference = np.mean(inference_times[-30:])
        avg_total = np.mean(total_times[-30:])
        
        print(f"Frame {frame_count:4d} | "
              f"FPS: {fps:5.1f} | "
              f"Inference: {avg_inference:5.1f}ms | "
              f"Total: {avg_total:5.1f}ms | "
              f"Failed: {failed_frames}")
    
    # Display
    cv2.putText(frame, f"FPS: {valid_frames/(time.time()-start_time):.1f}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Speed Test', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Final stats
elapsed = time.time() - start_time
cap.release()
cv2.destroyAllWindows()

print("\n" + "="*70)
print("FINAL RESULTS")
print("="*70)
print(f"\n📊 Camera Performance:")
print(f"   Valid frames:    {valid_frames}")
print(f"   Failed frames:   {failed_frames}")
print(f"   Success rate:    {100*valid_frames/(valid_frames+failed_frames):.1f}%")
print(f"   Total duration:  {elapsed:.1f}s")
print(f"\n⚡ Processing Speed:")
print(f"   Average FPS:     {valid_frames/elapsed:.1f} frames/sec")
print(f"   Avg inference:   {np.mean(inference_times):.1f}ms")
print(f"   Avg total time:  {np.mean(total_times):.1f}ms")
print(f"   Min inference:   {np.min(inference_times):.1f}ms")
print(f"   Max inference:   {np.max(inference_times):.1f}ms")

print(f"\n🎯 Expected vs Actual:")
print(f"   Camera native:   ~{fps_cam:.0f} FPS" if fps_cam > 0 else "   Camera native:   Unknown")
print(f"   Achieved:        {valid_frames/elapsed:.1f} FPS")
if fps_cam > 0:
    print(f"   Efficiency:      {100*(valid_frames/elapsed)/fps_cam:.0f}%")

print(f"\n💡 Comparison with Original (PyTorch ~65ms):")
print(f"   ONNX inference:  {np.mean(inference_times):.1f}ms")
print(f"   PyTorch (ref):   ~65ms")
print(f"   Speedup:         {65/np.mean(inference_times):.1f}x faster")

print("\n" + "="*70)
