"""
Script to find all available cameras and test them
Helps identify the correct index for OBS Virtual Camera
"""
import cv2
import sys

print("=" * 60)
print("🔍 CAMERA DETECTOR - Finding all available cameras")
print("=" * 60)

cameras_found = []

# Test camera indices 0-5 only (faster)
for index in range(6):
    print(f"\n📹 Testing camera index {index}...", end=" ")
    
    # Try CAP_DSHOW only (fastest on Windows)
    try:
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        
        if cap.isOpened():
            # Try to read a frame with timeout
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            ret, frame = cap.read()
            
            if ret and frame is not None:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                cameras_found.append({
                    'index': index,
                    'resolution': f"{width}x{height}",
                })
                
                print(f"✅ FOUND ({width}x{height})")
                cap.release()
            else:
                print("❌ Cannot read frame")
                cap.release()
        else:
            print("❌ Cannot open")
            if cap:
                cap.release()
                
    except Exception as e:
        print(f"❌ Error: {e}")
        continue

print("\n" + "=" * 60)
print("📊 SUMMARY")
print("=" * 60)

print("\n" + "=" * 60)
print("📊 SUMMARY")
print("=" * 60)

if cameras_found:
    for cam in cameras_found:
        print(f"\n✅ Camera {cam['index']}: {cam['resolution']}")
        
    print("\n💡 RECOMMENDATIONS:")
    if len(cameras_found) > 1:
        print(f"   📷 Laptop Camera: likely index {cameras_found[0]['index']}")
        print(f"   🎥 OBS Virtual Camera: likely index {cameras_found[-1]['index']}")
        print(f"\n   Use index {cameras_found[-1]['index']} for OBS in Flask app")
    else:
        print(f"   Only 1 camera found at index {cameras_found[0]['index']}")
else:
    print("\n❌ No cameras found!")
    print("   Make sure OBS Virtual Camera is started")

print("\n" + "=" * 60)
