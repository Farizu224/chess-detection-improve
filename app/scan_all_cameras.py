import cv2

print("🔍 Scanning all available cameras...\n")
print("="*60)

working_cameras = []

for i in range(10):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret and frame is not None:
            h, w = frame.shape[:2]
            print(f"✅ Camera {i}: WORKS - Resolution: {w}x{h}")
            working_cameras.append(i)
        else:
            print(f"⚠️  Camera {i}: Opens but cannot read frames")
        cap.release()
    else:
        # Stop checking after 3 consecutive failures
        if i >= 3 and len(working_cameras) > 0:
            break

print("\n" + "="*60)
print(f"📊 Summary: {len(working_cameras)} working camera(s) found")
print(f"   Indexes: {working_cameras}")
print("\n💡 Recommendations:")
print(f"   Camera 0: Usually laptop camera")
print(f"   Camera 2+: Usually external USB webcam")
print("\nTest your webcam with: python -c \"import cv2; cv2.imshow('Test', cv2.VideoCapture(X).read()[1]); cv2.waitKey(0)\"")
print("(Replace X with camera index)")
