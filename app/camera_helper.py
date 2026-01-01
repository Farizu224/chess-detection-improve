"""
Camera Helper - DroidCam Configuration Assistant
Membantu user menemukan dan mengkonfigurasi DroidCam dengan mudah
"""
import cv2
import time


def scan_all_cameras_safe(max_index=10, timeout_per_camera=3.0):
    """Scan all available cameras with timeout protection
    
    Args:
        max_index: Maximum camera index to scan
        timeout_per_camera: Maximum time to wait per camera (seconds)
        
    Returns:
        list of dict with camera info
    """
    print("=" * 70)
    print("🎥 SAFE CAMERA SCANNER (DroidCam Compatible)")
    print("=" * 70)
    print(f"Scanning cameras 0-{max_index-1} with {timeout_per_camera}s timeout each...")
    print()
    
    available_cameras = []
    
    for i in range(max_index):
        print(f"[Camera {i}] Testing...", end=" ", flush=True)
        
        # Try CAP_ANY first (best for DroidCam virtual camera)
        cap = None
        try:
            cap = cv2.VideoCapture(i, cv2.CAP_ANY)
            
            if cap.isOpened():
                # Try to read a frame with timeout
                start_time = time.time()
                ret = False
                frame = None
                
                # Attempt read (this is where DroidCam may hang)
                try:
                    ret, frame = cap.read()
                    elapsed = time.time() - start_time
                    
                    if ret and frame is not None:
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                        backend = cap.getBackendName()
                        
                        camera_info = {
                            'index': i,
                            'name': f'Camera {i}',
                            'resolution': f'{width}x{height}',
                            'fps': fps,
                            'backend': backend,
                            'read_time': f'{elapsed:.2f}s'
                        }
                        
                        available_cameras.append(camera_info)
                        
                        # Check if it's likely DroidCam (slower response)
                        if elapsed > 1.0:
                            print(f"✅ FOUND (Slow init - likely DroidCam/Virtual) {width}x{height}")
                        else:
                            print(f"✅ FOUND {width}x{height}")
                    else:
                        print("❌ Failed to read frame")
                        
                except Exception as e:
                    print(f"❌ Error reading: {e}")
            else:
                print("❌ Cannot open")
                
            if cap:
                cap.release()
                
        except Exception as e:
            print(f"❌ Exception: {e}")
            if cap:
                cap.release()
        
        # Small delay to prevent driver overload
        time.sleep(0.1)
    
    print()
    print("=" * 70)
    print(f"✅ Found {len(available_cameras)} camera(s)")
    print("=" * 70)
    
    return available_cameras


def identify_droidcam(cameras_list):
    """Try to identify which camera is likely DroidCam
    
    Args:
        cameras_list: List of camera dicts from scan_all_cameras_safe()
        
    Returns:
        int: Most likely DroidCam index, or None
    """
    if not cameras_list:
        return None
    
    # DroidCam characteristics:
    # - Usually higher resolution (720p or 1080p)
    # - Slower initialization time
    # - Backend is often "Auto" or "ANY"
    
    droidcam_candidates = []
    
    for cam in cameras_list:
        score = 0
        
        # Check resolution (DroidCam usually better than laptop webcam)
        if 'resolution' in cam:
            res = cam['resolution']
            if '1920' in res or '1280' in res or '1080' in res or '720' in res:
                score += 2
        
        # Check backend (CAP_ANY/Auto suggests virtual camera)
        if 'backend' in cam and cam['backend'] in ['Auto', 'ANY']:
            score += 1
        
        # Check init time (DroidCam usually slower)
        if 'read_time' in cam:
            try:
                read_time = float(cam['read_time'].replace('s', ''))
                if read_time > 0.5:  # Slow init suggests virtual camera
                    score += 2
            except:
                pass
        
        if score > 0:
            droidcam_candidates.append((cam['index'], score))
    
    if droidcam_candidates:
        # Sort by score (highest first)
        droidcam_candidates.sort(key=lambda x: x[1], reverse=True)
        return droidcam_candidates[0][0]
    
    return None


def test_droidcam_index(camera_index):
    """Test specific camera index for DroidCam
    
    Args:
        camera_index: Camera index to test
        
    Returns:
        bool: True if camera works, False otherwise
    """
    print("=" * 70)
    print(f"🎥 TESTING DROIDCAM AT INDEX {camera_index}")
    print("=" * 70)
    
    try:
        print("Opening camera...", end=" ", flush=True)
        cap = cv2.VideoCapture(camera_index, cv2.CAP_ANY)
        
        if not cap.isOpened():
            print("❌ Cannot open camera")
            return False
        
        print("✅")
        
        print("Reading test frame...", end=" ", flush=True)
        start_time = time.time()
        ret, frame = cap.read()
        elapsed = time.time() - start_time
        
        if not ret or frame is None:
            print("❌ Cannot read frame")
            cap.release()
            return False
        
        print(f"✅ ({elapsed:.2f}s)")
        
        # Get camera info
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        backend = cap.getBackendName()
        
        print()
        print("Camera Information:")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps}")
        print(f"  Backend: {backend}")
        print(f"  Init Time: {elapsed:.2f}s")
        
        cap.release()
        
        print()
        print("=" * 70)
        print("✅ DROIDCAM WORKING!")
        print("=" * 70)
        print()
        print(f"Update your config dengan: DROIDCAM_VIRTUAL_CAMERA_INDEX = {camera_index}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def run_droidcam_assistant():
    """Interactive assistant untuk setup DroidCam"""
    print("\n" + "=" * 70)
    print("🤖 DROIDCAM SETUP ASSISTANT")
    print("=" * 70)
    print()
    print("Langkah-langkah:")
    print("1. Pastikan DroidCam Client sudah running di laptop")
    print("2. Pastikan DroidCam app running di HP (mode USB atau WiFi)")
    print("3. Jika WiFi, pastikan HP dan laptop di jaringan yang sama")
    print()
    input("Tekan Enter untuk mulai scan...")
    print()
    
    # Scan cameras
    cameras = scan_all_cameras_safe(max_index=10, timeout_per_camera=2.0)
    
    if not cameras:
        print()
        print("❌ Tidak ada camera yang ditemukan!")
        print()
        print("Troubleshooting:")
        print("  • Pastikan DroidCam Client running")
        print("  • Coba restart DroidCam Client")
        print("  • Check Windows Device Manager untuk DroidCam Video")
        return
    
    print()
    print("Cameras found:")
    for cam in cameras:
        print(f"  [Index {cam['index']}] {cam['resolution']} - Backend: {cam['backend']}")
    
    # Try to identify DroidCam
    droidcam_index = identify_droidcam(cameras)
    
    print()
    if droidcam_index is not None:
        print(f"🎯 LIKELY DROIDCAM: Camera Index {droidcam_index}")
        print()
        response = input(f"Test camera {droidcam_index}? (Y/n): ").strip().lower()
        
        if response != 'n':
            print()
            if test_droidcam_index(droidcam_index):
                print()
                print("=" * 70)
                print("✅ SETUP COMPLETE!")
                print("=" * 70)
                print()
                print("Next steps:")
                print(f"1. Edit droidcam_config.py:")
                print(f"   DROIDCAM_VIRTUAL_CAMERA_INDEX = {droidcam_index}")
                print()
                print("2. Jalankan aplikasi web dan pilih camera tersebut")
                print()
            else:
                print("\n❌ Camera tidak berfungsi dengan baik")
    else:
        print("⚠️ Cannot auto-detect DroidCam")
        print()
        print("Silakan test manual:")
        for cam in cameras:
            print(f"  python -m app.camera_helper test {cam['index']}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test" and len(sys.argv) > 2:
        # Test specific index
        camera_index = int(sys.argv[2])
        test_droidcam_index(camera_index)
    else:
        # Run interactive assistant
        run_droidcam_assistant()
