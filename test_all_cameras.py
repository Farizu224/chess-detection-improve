"""
Test ALL camera indices to find which one actually captures frames
"""
import cv2
import numpy as np

print("=" * 70)
print("TESTING ALL CAMERA INDICES")
print("=" * 70)

working_cameras = []

# Test indices 0-5
for idx in range(6):
    print(f"\n{'='*70}")
    print(f"Testing camera index {idx}...")
    print(f"{'='*70}")
    
    # Try DirectShow first
    for backend_name, backend in [("DirectShow", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF), ("Any", cv2.CAP_ANY)]:
        print(f"\n  Backend: {backend_name}")
        cap = cv2.VideoCapture(idx, backend)
        
        if not cap.isOpened():
            print(f"  ❌ Cannot open")
            continue
            
        print(f"  ✅ Camera opened")
        
        # Get camera properties
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        brightness = cap.get(cv2.CAP_PROP_BRIGHTNESS)
        
        print(f"  Resolution: {int(width)}x{int(height)}")
        print(f"  FPS: {fps}")
        print(f"  Brightness setting: {brightness}")
        
        # Try to capture frame
        ret, frame = cap.read()
        
        if not ret or frame is None:
            print(f"  ❌ Cannot read frame")
            cap.release()
            continue
            
        # Calculate actual brightness
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray)
        
        print(f"  ✅ Frame captured!")
        print(f"  📊 Actual brightness: {avg_brightness:.1f}")
        
        # Save frame
        filename = f"camera_idx{idx}_{backend_name.lower()}.jpg"
        cv2.imwrite(filename, frame)
        print(f"  💾 Saved: {filename}")
        
        # Check if frame is usable
        if avg_brightness > 20:
            print(f"  ✅✅ USABLE FRAME! (brightness > 20)")
            working_cameras.append({
                'index': idx,
                'backend': backend_name,
                'backend_code': backend,
                'brightness': avg_brightness,
                'resolution': f"{int(width)}x{int(height)}"
            })
        else:
            print(f"  ⚠️ Frame too dark (brightness {avg_brightness:.1f})")
        
        cap.release()
        break  # Stop after first successful backend

print(f"\n{'='*70}")
print(f"SUMMARY")
print(f"{'='*70}")

if working_cameras:
    print(f"\n✅ Found {len(working_cameras)} working camera(s):\n")
    for cam in working_cameras:
        print(f"  Index {cam['index']}: {cam['backend']} backend")
        print(f"    Brightness: {cam['brightness']:.1f}")
        print(f"    Resolution: {cam['resolution']}")
        print(f"    Usage: cv2.VideoCapture({cam['index']}, cv2.CAP_{cam['backend'].upper()})")
        print()
else:
    print("\n❌ No working cameras found with adequate brightness!")
    print("\nTROUBLESHOOTING:")
    print("  1. Check if camera lens is covered")
    print("  2. Try in a different USB port")
    print("  3. Restart the camera/computer")
    print("  4. Check camera permissions in Windows Settings")

print("="*70)
