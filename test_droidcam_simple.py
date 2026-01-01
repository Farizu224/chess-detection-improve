"""
Test DroidCam dengan approach yang working (simple, no timeout complexity)
"""
import cv2
import time
import platform

def test_droidcam_simple(camera_index=1):
    """Test camera dengan cara yang sama seperti versi yang working"""
    print("=" * 70)
    print(f"🎥 TESTING DROIDCAM - Camera Index {camera_index}")
    print("=" * 70)
    print()
    
    # Try multiple backends like working version
    if platform.system() == 'Windows':
        backend_candidates = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    else:
        backend_candidates = [cv2.CAP_ANY]
    
    cap = None
    test_frame = None
    successful_backend = None
    
    for backend in backend_candidates:
        backend_name = {
            cv2.CAP_DSHOW: 'CAP_DSHOW',
            cv2.CAP_MSMF: 'CAP_MSMF',
            cv2.CAP_ANY: 'CAP_ANY',
        }.get(backend, str(backend))
        
        print(f"[{backend_name}] Opening camera {camera_index}...", end=" ", flush=True)
        
        try:
            cap = cv2.VideoCapture(camera_index, backend)
            
            if not cap or not cap.isOpened():
                print("❌ Cannot open")
                if cap:
                    cap.release()
                continue
            
            print("✅ Opened!", end=" ")
            
            # WARM UP: Read multiple frames (KEY to DroidCam success!)
            print("Warming up...", end=" ", flush=True)
            test_ret = False
            for attempt in range(10):
                test_ret, test_frame = cap.read()
                if test_ret and test_frame is not None:
                    print(f"✅ Got frame (attempt {attempt + 1})")
                    successful_backend = backend_name
                    break
                time.sleep(0.1)
            
            if test_ret and test_frame is not None:
                break
            else:
                print("❌ No frames")
                cap.release()
        
        except Exception as e:
            print(f"❌ Error: {e}")
            if cap:
                cap.release()
    
    print()
    print("=" * 70)
    
    if cap and cap.isOpened() and test_frame is not None:
        # Success!
        height, width = test_frame.shape[:2]
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        
        print("✅ DROIDCAM WORKING!")
        print()
        print(f"  Backend:    {successful_backend}")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS:        {actual_fps}")
        print(f"  Frame type: {test_frame.dtype}")
        print()
        
        # Show preview
        print("Opening preview window (press 'q' to close)...")
        cv2.namedWindow('DroidCam Test', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('DroidCam Test', 640, 480)
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("⚠️ Lost frame")
                break
            
            frame_count += 1
            elapsed = time.time() - start_time
            if elapsed > 0:
                actual_fps = frame_count / elapsed
            
            # Draw info
            info_text = f"FPS: {actual_fps:.1f} | Frame: {frame_count}"
            cv2.putText(frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('DroidCam Test', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        print()
        print("=" * 70)
        print("✅ TEST COMPLETE - DroidCam is working properly!")
        print()
        print(f"Use this in your app:")
        print(f"  • Camera Index: {camera_index}")
        print(f"  • Backend: {successful_backend}")
        print("=" * 70)
        
        return True
    else:
        print("❌ FAILED - DroidCam not working")
        print()
        print("Troubleshooting:")
        print("  1. Make sure DroidCam Client is running")
        print("  2. DroidCam app running on phone")
        print("  3. Phone connected (USB or WiFi)")
        print("  4. Try different camera index (0, 1, 2)")
        print()
        print("=" * 70)
        return False


if __name__ == "__main__":
    import sys
    
    # Default to camera 1 (common for DroidCam)
    camera_idx = 1
    if len(sys.argv) > 1:
        camera_idx = int(sys.argv[1])
    
    print()
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║              DROIDCAM SIMPLE TEST (Working Method)                ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    print()
    print("This uses the SAME approach as the working version.")
    print("No complex timeout logic - just simple, robust backend testing.")
    print()
    
    if not test_droidcam_simple(camera_idx):
        print()
        print("Want to try a different camera index?")
        print("Run: python test_droidcam_simple.py <index>")
        print("Example: python test_droidcam_simple.py 0")
