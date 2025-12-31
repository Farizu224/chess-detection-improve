"""
Cek semua kamera yang tersedia dan test USB camera
"""
import cv2
import numpy as np

print("="*70)
print("SCANNING ALL AVAILABLE CAMERAS")
print("="*70)

cameras_found = []

print("\n[1/2] Scanning camera indices 0-9...")
for i in range(10):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret and frame is not None:
            h, w = frame.shape[:2]
            
            # Get camera info
            backend = cap.getBackendName()
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            cameras_found.append({
                'index': i,
                'resolution': f"{w}x{h}",
                'backend': backend,
                'fps': fps
            })
            
            print(f"\n   ✅ Camera {i} FOUND:")
            print(f"      Resolution: {w}x{h}")
            print(f"      Backend: {backend}")
            print(f"      FPS: {fps}")
            
            # Determine if likely USB or built-in
            if i > 0:
                print(f"      Type: Likely USB/External camera")
            else:
                print(f"      Type: Likely Built-in camera")
            
            cap.release()
        else:
            cap.release()
    else:
        if i > len(cameras_found) + 2:
            break  # Stop if we've gone 2 indices past the last camera

print(f"\n[2/2] Summary:")
print(f"   Total cameras found: {len(cameras_found)}")

if len(cameras_found) == 0:
    print("\n   ❌ NO CAMERAS FOUND!")
    print("   Troubleshooting:")
    print("   - Check Windows Camera privacy settings")
    print("   - Ensure USB camera is plugged in")
    print("   - Try different USB port")
elif len(cameras_found) == 1:
    print(f"\n   Only built-in camera detected (index 0)")
    print(f"   ⚠️ USB camera not detected!")
    print(f"\n   Troubleshooting USB camera:")
    print(f"   1. Unplug and replug USB camera")
    print(f"   2. Check if USB camera shows in Device Manager")
    print(f"   3. Try different USB port")
    print(f"   4. Update USB camera drivers")
else:
    print(f"\n   ✅ Multiple cameras detected!")
    print(f"\n   Recommendations:")
    for cam in cameras_found:
        if cam['index'] == 0:
            print(f"   - Camera {cam['index']}: Built-in laptop camera")
        else:
            print(f"   - Camera {cam['index']}: USB/External camera ⭐ USE THIS")

print("\n" + "="*70)
print("NEXT STEPS:")
print("="*70)

if len(cameras_found) > 1:
    usb_cam = cameras_found[1]
    print(f"\nTo use USB camera (index {usb_cam['index']}):")
    print(f"1. Di web interface, pilih Camera Index: {usb_cam['index']}")
    print(f"2. Atau edit di templates untuk default USB camera")
else:
    print(f"\nTo use built-in camera (index 0):")
    print(f"1. Camera Index: 0 (default)")

print("="*70)
