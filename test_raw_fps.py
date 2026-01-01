"""
ULTRA SIMPLE TEST - Minimal overhead untuk cek FPS maksimal
"""
import cv2
import time
import numpy as np

print("="*70)
print("ULTRA SIMPLE CAMERA TEST - NO PROCESSING")
print("="*70)

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("❌ Cannot open camera")
    exit(1)

# Exposure settings
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
cap.set(cv2.CAP_PROP_EXPOSURE, -1)
cap.set(cv2.CAP_PROP_BRIGHTNESS, 255)
cap.set(cv2.CAP_PROP_GAIN, 100)
time.sleep(1.5)
cap.read()  # Discard

print("✅ Camera opened")
print("\nRunning for 5 seconds with MINIMAL processing...")
print("Press 'q' to quit early\n")

fps_list = []
frame_count = 0
start_time = time.time()

while time.time() - start_time < 5:
    loop_start = time.time()
    
    ret, frame = cap.read()
    if not ret:
        continue
    
    frame_count += 1
    
    # MINIMAL processing - just show frame
    cv2.imshow('Simple Test', frame)
    
    # NON-BLOCKING waitKey
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    loop_time = time.time() - loop_start
    fps_list.append(1.0 / loop_time if loop_time > 0 else 0)

elapsed = time.time() - start_time
avg_fps = frame_count / elapsed if elapsed > 0 else 0

cap.release()
cv2.destroyAllWindows()

print("="*70)
print("RESULTS")
print("="*70)
print(f"Total frames: {frame_count}")
print(f"Time elapsed: {elapsed:.2f}s")
print(f"Average FPS: {avg_fps:.1f}")
print(f"Max FPS: {max(fps_list):.1f}")
print(f"Min FPS: {min(fps_list):.1f}")

if avg_fps > 25:
    print(f"\n✅✅ EXCELLENT! Camera capable of {avg_fps:.0f} FPS")
elif avg_fps > 20:
    print(f"\n✅ GOOD! Camera capable of {avg_fps:.0f} FPS")
else:
    print(f"\n⚠️ Camera FPS is low: {avg_fps:.0f}")

print("\nThis is BASELINE FPS without any detection.")
print("With detection, expect 50-70% of this FPS.")
print("="*70)
