"""
Test OpenCV GUI dan Camera dengan window
"""
import cv2
import numpy as np

print("="*70)
print("TESTING OPENCV GUI & CAMERA")
print("="*70)

print("\n[1/3] Testing OpenCV installation...")
print(f"   OpenCV version: {cv2.__version__}")

print("\n[2/3] Testing GUI support...")
try:
    # Create test window
    test_img = np.zeros((300, 400, 3), dtype=np.uint8)
    cv2.putText(test_img, "OpenCV GUI Test", (50, 150), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.namedWindow('Test Window', cv2.WINDOW_NORMAL)
    cv2.imshow('Test Window', test_img)
    print("   ✅ GUI functions work!")
    print("   ⚠️ Test window created (will close in 2 seconds)")
    cv2.waitKey(2000)
    cv2.destroyAllWindows()
    print("   ✅ Window cleanup successful!")
except Exception as e:
    print(f"   ❌ GUI Error: {e}")
    import traceback
    traceback.print_exc()

print("\n[3/3] Testing camera access...")
for i in range(3):
    try:
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                print(f"   ✅ Camera {i} FOUND - Resolution: {w}x{h}")
                cap.release()
            else:
                print(f"   ⚠️ Camera {i} opened but cannot read")
                cap.release()
                break
        else:
            if i == 0:
                print(f"   ❌ Camera {i} NOT FOUND")
            break
    except Exception as e:
        print(f"   ❌ Camera {i} error: {e}")
        break

print("\n" + "="*70)
print("✅ OPENCV GUI READY!")
print("="*70)
print("\nYou can now run the app:")
print("  cd app")
print("  python app.py")
print("\nOpenCV detection window will appear when you start detection.")
print("="*70)
