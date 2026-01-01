"""
DroidCam Configuration Helper
Untuk menggunakan DroidCam sebagai pengganti webcam
"""
import os
import cv2

# ========== DROIDCAM CONFIGURATION ==========

# 🎯 QUICK CONFIG: Set ini setelah run identify_cameras_quick.py
# Jika tahu DroidCam ada di index berapa, set DROIDCAM_CAMERA_INDEX
# Ini akan SKIP laptop webcam dan langsung pakai DroidCam!
DROIDCAM_CAMERA_INDEX = 0  # ✅ IDENTIFIED: Camera 0 = DroidCam (HIGH RES - needs Media Foundation)
LAPTOP_WEBCAM_INDEX = 1    # ✅ IDENTIFIED: Camera 1 = Laptop Webcam

# 🔧 SKIP LAPTOP WEBCAM: Set True untuk skip webcam laptop
# Web app akan hide laptop webcam dari dropdown dan hanya show DroidCam
SKIP_LAPTOP_WEBCAM = True  # ✅ ENABLED: Skip Camera 1 (laptop)

# NOTE: Camera 2 mungkin adalah output dari virtual camera lain (resolusi rendah)
#       Camera 0 adalah DroidCam asli dengan resolusi tinggi (butuh Media Foundation backend)

# ========== LEGACY OPTIONS (for URL mode) ==========

# Option 1: DroidCam via Virtual Camera (RECOMMENDED - paling mudah)
# Setelah install DroidCam Client, biasanya muncul sebagai virtual camera
# Coba index: 1, 2, atau 3
DROIDCAM_VIRTUAL_CAMERA_INDEX = 1  # Adjust sesuai system Anda

# Option 2: DroidCam via WiFi/IP Camera
# Buka DroidCam app di HP, lihat IP address di bagian WiFi IP
# Format: http://IP_ADDRESS:4747/video
DROIDCAM_URL = "http://192.168.100.27:4747/video"  # IP HP Anda

# Option 3: DroidCam via USB (muncul sebagai virtual camera index)
# Biasanya sama dengan virtual camera

# ========== WHICH MODE TO USE? ==========
USE_DROIDCAM_URL = False  # Set True untuk pakai WiFi, False untuk virtual camera
USE_DROIDCAM_VIRTUAL = True  # Set True untuk pakai virtual camera index

# ========== DETECTION SETTINGS ==========
# Karena DroidCam biasanya lebih bagus dari webcam laptop,
# threshold bisa diturunkan untuk deteksi lebih sensitive

# Untuk kamera bagus (DroidCam):
CONFIDENCE_THRESHOLD_HIGH = 0.4  # Turun dari 0.5
IOU_THRESHOLD_HIGH = 0.4

# Untuk kamera jelek (webcam laptop):
CONFIDENCE_THRESHOLD_LOW = 0.6  # Naik dari 0.5
IOU_THRESHOLD_LOW = 0.5


def get_camera_source():
    """Get camera source berdasarkan konfigurasi"""
    if USE_DROIDCAM_URL:
        return DROIDCAM_URL
    elif USE_DROIDCAM_VIRTUAL:
        return DROIDCAM_VIRTUAL_CAMERA_INDEX
    else:
        return 0  # Default webcam


def get_detection_thresholds(use_droidcam=True):
    """Get detection thresholds berdasarkan kualitas camera"""
    if use_droidcam:
        return CONFIDENCE_THRESHOLD_HIGH, IOU_THRESHOLD_HIGH
    else:
        return CONFIDENCE_THRESHOLD_LOW, IOU_THRESHOLD_LOW


def test_droidcam_connection():
    """Test apakah DroidCam bisa connect"""
    print("=" * 60)
    print("🎥 TESTING DROIDCAM CONNECTION")
    print("=" * 60)
    
    # Test virtual camera
    if USE_DROIDCAM_VIRTUAL:
        print(f"\n[Test 1] Testing DroidCam Virtual Camera (index {DROIDCAM_VIRTUAL_CAMERA_INDEX})...")
        cap = cv2.VideoCapture(DROIDCAM_VIRTUAL_CAMERA_INDEX)
        
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                print(f"✅ DroidCam Virtual Camera WORKING!")
                print(f"   Resolution: {w}x{h}")
                cap.release()
            else:
                print(f"❌ Camera opened but cannot read frames")
                cap.release()
        else:
            print(f"❌ Cannot open camera index {DROIDCAM_VIRTUAL_CAMERA_INDEX}")
            print(f"💡 Coba index lain (0, 1, 2, 3) atau pastikan DroidCam Client running")
    
    # Test URL
    if USE_DROIDCAM_URL:
        print(f"\n[Test 2] Testing DroidCam via URL: {DROIDCAM_URL}")
        cap = cv2.VideoCapture(DROIDCAM_URL, cv2.CAP_FFMPEG)
        
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                print(f"✅ DroidCam URL WORKING!")
                print(f"   Resolution: {w}x{h}")
                cap.release()
            else:
                print(f"❌ URL opened but cannot read frames")
                cap.release()
        else:
            print(f"❌ Cannot open DroidCam URL")
            print(f"💡 Troubleshooting:")
            print(f"   1. Pastikan DroidCam app running di HP")
            print(f"   2. Check IP address di DroidCam app (WiFi IP)")
            print(f"   3. HP & laptop harus di WiFi yang sama")
            print(f"   4. Try ping IP dari laptop: ping {DROIDCAM_URL.split('//')[1].split(':')[0]}")
    
    print("\n" + "=" * 60)
    print("📊 TEST COMPLETE")
    print("=" * 60)


def scan_all_cameras():
    """Scan semua camera index untuk cari DroidCam virtual camera"""
    print("🔍 Scanning for cameras (index 0-10)...\n")
    
    found_cameras = []
    
    for i in range(11):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                print(f"✅ Camera {i}: {w}x{h}")
                found_cameras.append(i)
            cap.release()
        else:
            print(f"❌ Camera {i}: Not available")
    
    print(f"\n📊 Found {len(found_cameras)} camera(s): {found_cameras}")
    
    if len(found_cameras) > 1:
        print(f"\n💡 Tip: Camera selain index 0 biasanya adalah DroidCam virtual camera")
        print(f"   Coba set DROIDCAM_VIRTUAL_CAMERA_INDEX = {found_cameras[-1]}")
    
    return found_cameras


# ========== QUICK SETUP WIZARD ==========

def setup_wizard():
    """Interactive setup untuk DroidCam"""
    print("\n" + "=" * 60)
    print("🎯 DROIDCAM SETUP WIZARD")
    print("=" * 60)
    
    print("\n[1] Pilih mode DroidCam:")
    print("   1. Virtual Camera (DroidCam Client installed di laptop)")
    print("   2. WiFi/IP Camera (connect via URL)")
    
    try:
        choice = input("\nPilih mode (1/2): ").strip()
        
        if choice == "1":
            print("\n[2] Scanning cameras...")
            cameras = scan_all_cameras()
            
            if len(cameras) > 1:
                cam_choice = input(f"\nPilih camera index ({cameras}): ").strip()
                try:
                    cam_index = int(cam_choice)
                    if cam_index in cameras:
                        print(f"\n✅ Setting: USE_DROIDCAM_VIRTUAL = True, INDEX = {cam_index}")
                        print(f"\n📝 Update file droidcam_config.py:")
                        print(f"   DROIDCAM_VIRTUAL_CAMERA_INDEX = {cam_index}")
                        print(f"   USE_DROIDCAM_VIRTUAL = True")
                        print(f"   USE_DROIDCAM_URL = False")
                    else:
                        print(f"❌ Invalid index")
                except:
                    print(f"❌ Invalid input")
            else:
                print(f"\n⚠️ Hanya 1 camera ditemukan. Install DroidCam Client dulu.")
        
        elif choice == "2":
            print(f"\n[2] Setup WiFi/IP Camera")
            print(f"   1. Buka DroidCam app di HP")
            print(f"   2. Lihat WiFi IP (biasanya 192.168.x.x:4747)")
            
            ip = input(f"\nMasukkan IP address (contoh: 192.168.1.100): ").strip()
            
            if ip:
                url = f"http://{ip}:4747/video"
                print(f"\n✅ Testing URL: {url}")
                
                cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        print(f"✅ DroidCam URL WORKING!")
                        print(f"\n📝 Update file droidcam_config.py:")
                        print(f"   DROIDCAM_URL = '{url}'")
                        print(f"   USE_DROIDCAM_URL = True")
                        print(f"   USE_DROIDCAM_VIRTUAL = False")
                    else:
                        print(f"❌ Cannot read from URL")
                    cap.release()
                else:
                    print(f"❌ Cannot open URL. Check:")
                    print(f"   - DroidCam app running?")
                    print(f"   - Same WiFi network?")
                    print(f"   - IP address benar?")
        
        else:
            print("❌ Invalid choice")
    
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            test_droidcam_connection()
        elif sys.argv[1] == "scan":
            scan_all_cameras()
        elif sys.argv[1] == "setup":
            setup_wizard()
        else:
            print("Usage: python droidcam_config.py [test|scan|setup]")
    else:
        # Default: run setup wizard
        setup_wizard()
