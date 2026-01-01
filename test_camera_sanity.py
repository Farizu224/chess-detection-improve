"""
🔧 CAMERA SANITY CHECK
Test kamera dengan kode paling sederhana (NO Flask, NO Threading, NO AI)
Untuk diagnosa: Hardware Problem vs Software Regression
"""
import cv2
import sys

def test_camera(index, backend_name, backend_flag):
    """Test single camera with specific backend"""
    print(f"\n{'='*60}")
    print(f"🧪 Testing Camera Index {index} with {backend_name}")
    print(f"{'='*60}")
    
    try:
        cap = cv2.VideoCapture(index, backend_flag)
        
        if not cap.isOpened():
            print(f"❌ FAILED: Could not open camera {index} with {backend_name}")
            return False
        
        print(f"✅ SUCCESS: Camera {index} opened with {backend_name}")
        
        # Try to read a frame
        print("   📸 Attempting to read frame...")
        ret, frame = cap.read()
        
        if not ret or frame is None:
            print(f"⚠️  WARNING: Camera opened but frame is empty/black")
            cap.release()
            return False
        
        print(f"✅ Frame captured successfully!")
        print(f"   Resolution: {frame.shape[1]}x{frame.shape[0]}")
        print(f"   Channels: {frame.shape[2]}")
        
        # Show frame briefly
        cv2.imshow(f"Camera {index} - {backend_name}", frame)
        print("   👁️  Displaying frame for 2 seconds...")
        cv2.waitKey(2000)
        
        cap.release()
        cv2.destroyAllWindows()
        return True
        
    except Exception as e:
        print(f"❌ EXCEPTION: {e}")
        return False

def main():
    print("\n" + "="*60)
    print("🔍 CAMERA HARDWARE SANITY CHECK")
    print("="*60)
    print("Goal: Verify if camera hardware works with SIMPLEST code")
    print("If this fails → Hardware/Driver problem")
    print("If this works → Regression in app code")
    print()
    
    # Test configurations
    tests = [
        # (index, name, backend)
        (2, "DirectShow", cv2.CAP_DSHOW),  # Your usual camera
        (1, "DirectShow", cv2.CAP_DSHOW),
        (0, "DirectShow", cv2.CAP_DSHOW),
        (2, "Default/Auto", cv2.CAP_ANY),   # Fallback
    ]
    
    results = []
    for idx, name, backend in tests:
        success = test_camera(idx, name, backend)
        results.append((idx, name, success))
    
    # Summary
    print("\n" + "="*60)
    print("📊 RESULTS SUMMARY")
    print("="*60)
    
    any_success = False
    for idx, name, success in results:
        status = "✅ WORKS" if success else "❌ FAILED"
        print(f"   Camera {idx} ({name}): {status}")
        if success:
            any_success = True
    
    print()
    if any_success:
        print("🎉 CONCLUSION: Hardware is FINE!")
        print("   → Problem is in app/chess_detection.py code (REGRESSION)")
        print("   → Need to simplify camera initialization logic")
    else:
        print("⚠️  CONCLUSION: ALL tests failed!")
        print("   → Check USB cable connection")
        print("   → Close other apps using camera (Zoom, Teams, Browser)")
        print("   → Try different USB port")
        print("   → Restart computer")
    
    print("="*60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        cv2.destroyAllWindows()
        sys.exit(0)
