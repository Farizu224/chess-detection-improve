"""
Fix OpenCV dan Test Camera Access
"""
import subprocess
import sys

print("="*70)
print("FIXING OPENCV AND CAMERA ISSUES")
print("="*70)

print("\n[1/4] Checking current OpenCV installation...")
try:
    import cv2
    print(f"   Current: opencv-python {cv2.__version__}")
    print(f"   Build info: {cv2.getBuildInformation()[:200]}...")
except Exception as e:
    print(f"   Error: {e}")

print("\n[2/4] Uninstalling opencv-python-headless (no GUI support)...")
result = subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "opencv-python-headless"], 
                       capture_output=True, text=True)
if result.returncode == 0:
    print("   ✅ Removed opencv-python-headless")
else:
    print("   ⚠️ opencv-python-headless not found or already removed")

print("\n[3/4] Installing full opencv-python (with GUI support)...")
result = subprocess.run([sys.executable, "-m", "pip", "install", "--force-reinstall", "opencv-python"], 
                       capture_output=True, text=True)
if result.returncode == 0:
    print("   ✅ Installed full opencv-python")
else:
    print(f"   ❌ Error: {result.stderr}")

print("\n[4/4] Testing camera access...")
# Reimport after reinstall
try:
    import importlib
    import cv2
    importlib.reload(cv2)
    
    print(f"   OpenCV version: {cv2.__version__}")
    
    # Test camera
    print("\n   Testing cameras...")
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                h, w = frame.shape[:2]
                print(f"   ✅ Camera {i} FOUND - Resolution: {w}x{h}")
                cap.release()
            else:
                print(f"   ⚠️ Camera {i} opened but cannot read frame")
                cap.release()
        else:
            if i == 0:
                print(f"   ❌ Camera {i} NOT FOUND")
            break
    
except Exception as e:
    print(f"   ❌ Error testing camera: {e}")

print("\n" + "="*70)
print("NEXT STEPS:")
print("="*70)
print("1. Restart your terminal/IDE")
print("2. Run: python app.py")
print("3. If still no camera:")
print("   - Check Windows Camera permissions (Settings > Privacy > Camera)")
print("   - Test with: python find_cameras.py")
print("   - Try external USB webcam if integrated camera fails")
print("="*70)
