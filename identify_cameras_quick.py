"""
Quick Camera Identifier - Identify which is laptop webcam, which is DroidCam
RUN THIS FIRST before using the web app!
"""
import cv2
import time
import platform

def identify_all_cameras():
    """Identify all cameras and tell which is which"""
    print("=" * 80)
    print("🔍 CAMERA IDENTIFIER - Finding Laptop Webcam vs DroidCam")
    print("=" * 80)
    print()
    print("Testing cameras 0-5... Please wait...")
    print()
    
    cameras_found = []
    
    for i in range(6):
        print(f"[Camera {i}]", end=" ", flush=True)
        
        # Try with DirectShow (Windows default)
        cap = None
        try:
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            
            if not cap.isOpened():
                print("❌ Not available")
                continue
            
            # Try to read frame
            start_time = time.time()
            ret = False
            frame = None
            
            # Try up to 10 times with delay (for DroidCam warm-up)
            for attempt in range(10):
                ret, frame = cap.read()
                if ret and frame is not None:
                    break
                time.sleep(0.1)
            
            read_time = time.time() - start_time
            
            if not ret or frame is None:
                print("❌ Cannot read frames")
                cap.release()
                continue
            
            # Get camera info
            height, width = frame.shape[:2]
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            backend = cap.getBackendName()
            
            # Analyze characteristics
            is_likely_droidcam = False
            reasons = []
            
            # 1. Resolution check (DroidCam usually higher)
            if width >= 1280 or height >= 720:
                is_likely_droidcam = True
                reasons.append("High resolution")
            
            # 2. Init time check (DroidCam slower)
            if read_time > 0.5:
                is_likely_droidcam = True
                reasons.append("Slow initialization")
            
            # 3. Quick brightness check (laptop webcam often darker)
            avg_brightness = frame.mean()
            if avg_brightness > 100:
                reasons.append("Good brightness")
            
            camera_info = {
                'index': i,
                'width': width,
                'height': height,
                'fps': fps,
                'backend': backend,
                'read_time': read_time,
                'avg_brightness': avg_brightness,
                'is_likely_droidcam': is_likely_droidcam,
                'reasons': reasons
            }
            
            cameras_found.append(camera_info)
            
            if is_likely_droidcam:
                print(f"✅ {width}x{height} - 🎯 LIKELY DROIDCAM ({', '.join(reasons)})")
            else:
                print(f"✅ {width}x{height} - 💻 Likely laptop webcam (low res)")
            
            cap.release()
            
        except Exception as e:
            print(f"❌ Error: {e}")
            if cap:
                cap.release()
        
        # Small delay between cameras
        time.sleep(0.2)
    
    print()
    print("=" * 80)
    print(f"📊 SUMMARY: Found {len(cameras_found)} camera(s)")
    print("=" * 80)
    print()
    
    if not cameras_found:
        print("❌ No cameras found!")
        print()
        print("Troubleshooting:")
        print("  • Check if DroidCam Client is running")
        print("  • Try restart DroidCam Client")
        print("  • Check Windows Device Manager")
        return None
    
    # Print detailed summary
    laptop_webcam = None
    droidcam = None
    
    for cam in cameras_found:
        print(f"Camera {cam['index']}:")
        print(f"  Resolution:    {cam['width']}x{cam['height']}")
        print(f"  FPS:           {cam['fps']:.1f}")
        print(f"  Init Time:     {cam['read_time']:.2f}s")
        print(f"  Brightness:    {cam['avg_brightness']:.1f}")
        print(f"  Backend:       {cam['backend']}")
        
        if cam['is_likely_droidcam']:
            print(f"  Type:          🎯 DROIDCAM (recommended)")
            droidcam = cam['index']
        else:
            print(f"  Type:          💻 Laptop Webcam")
            laptop_webcam = cam['index']
        
        print()
    
    # Recommendation
    print("=" * 80)
    print("🎯 RECOMMENDATION:")
    print("=" * 80)
    
    if droidcam is not None:
        print(f"✅ Use Camera {droidcam} (DroidCam - better quality)")
        print()
        print(f"Update your config or select Camera {droidcam} in web app!")
        return droidcam
    elif laptop_webcam is not None:
        print(f"⚠️ Only laptop webcam found (Camera {laptop_webcam})")
        print()
        print("DroidCam not detected. Make sure:")
        print("  1. DroidCam Client is running")
        print("  2. Phone is connected")
        print("  3. Video preview visible in DroidCam Client")
        return laptop_webcam
    else:
        print("❌ No usable cameras found")
        return None


def test_specific_camera_live(camera_index):
    """Test specific camera with live preview to verify it's the right one"""
    print()
    print("=" * 80)
    print(f"🎥 LIVE TEST: Camera {camera_index}")
    print("=" * 80)
    print()
    
    # Try multiple backends
    if platform.system() == 'Windows':
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    else:
        backends = [cv2.CAP_ANY]
    
    cap = None
    working_backend = None
    
    for backend in backends:
        backend_name = {
            cv2.CAP_DSHOW: 'DirectShow',
            cv2.CAP_MSMF: 'Media Foundation',
            cv2.CAP_ANY: 'Auto'
        }.get(backend, str(backend))
        
        print(f"Trying {backend_name}...", end=" ", flush=True)
        
        try:
            cap = cv2.VideoCapture(camera_index, backend)
            
            if cap.isOpened():
                # Warm up
                ret = False
                frame = None
                for _ in range(10):
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        break
                    time.sleep(0.1)
                
                if ret and frame is not None:
                    print("✅ Working!")
                    working_backend = backend_name
                    break
                else:
                    print("❌ No frames")
                    cap.release()
            else:
                print("❌ Cannot open")
                
        except Exception as e:
            print(f"❌ Error: {e}")
            if cap:
                cap.release()
    
    if not cap or not cap.isOpened():
        print()
        print("❌ Cannot open this camera")
        return False
    
    print()
    print(f"✅ Camera {camera_index} is working with {working_backend}")
    print()
    print("Opening live preview...")
    print("📹 Check if this is the RIGHT camera (laptop or DroidCam)")
    print("Press 'q' to close, 'd' to mark as DroidCam, 'l' for laptop webcam")
    print()
    
    cv2.namedWindow('Camera Test', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Camera Test', 800, 600)
    
    camera_type = None
    
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("⚠️ Lost frame")
            break
        
        # Draw instructions
        h, w = frame.shape[:2]
        
        cv2.putText(frame, f"Camera {camera_index}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Resolution: {w}x{h}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, "Is this DroidCam or Laptop webcam?", (10, h-80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(frame, "Press: 'd' = DroidCam | 'l' = Laptop | 'q' = Quit", (10, h-40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        cv2.imshow('Camera Test', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            camera_type = 'droidcam'
            print(f"\n✅ Marked Camera {camera_index} as DROIDCAM")
            break
        elif key == ord('l'):
            camera_type = 'laptop'
            print(f"\n✅ Marked Camera {camera_index} as LAPTOP WEBCAM")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    return camera_type


if __name__ == "__main__":
    import sys
    
    print()
    print("╔══════════════════════════════════════════════════════════════════════════╗")
    print("║              CAMERA IDENTIFIER - Find DroidCam vs Laptop                ║")
    print("╚══════════════════════════════════════════════════════════════════════════╝")
    print()
    
    if len(sys.argv) > 1:
        # Test specific camera
        camera_idx = int(sys.argv[1])
        camera_type = test_specific_camera_live(camera_idx)
        
        if camera_type:
            print()
            print("=" * 80)
            if camera_type == 'droidcam':
                print(f"✅ Camera {camera_idx} = DROIDCAM")
                print()
                print(f"Use this in web app: Select Camera {camera_idx}")
            else:
                print(f"✅ Camera {camera_idx} = LAPTOP WEBCAM")
                print()
                print(f"⚠️ For better quality, use DroidCam instead")
            print("=" * 80)
    else:
        # Auto-identify all cameras
        recommended = identify_all_cameras()
        
        if recommended is not None:
            print()
            response = input(f"Want to verify Camera {recommended} with live preview? (Y/n): ").strip().lower()
            
            if response != 'n':
                camera_type = test_specific_camera_live(recommended)
                
                if camera_type == 'droidcam':
                    print()
                    print("=" * 80)
                    print("✅ SETUP COMPLETE!")
                    print("=" * 80)
                    print()
                    print(f"Camera {recommended} is confirmed as DroidCam")
                    print()
                    print("Next steps:")
                    print(f"  1. Start web app: START_APP.bat")
                    print(f"  2. Select Camera {recommended} from dropdown")
                    print(f"  3. Start detection")
                    print()
