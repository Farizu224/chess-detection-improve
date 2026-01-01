"""
Quick Test - DroidCam with Chess Detection
Test DroidCam camera dan detection threshold
"""
import cv2
import sys

def test_camera_and_display(camera_source):
    """Test camera source (index atau URL) dan tampilkan preview"""
    
    print("=" * 60)
    print(f"🎥 Testing Camera: {camera_source}")
    print("=" * 60)
    
    # Buka camera
    if isinstance(camera_source, str):
        print("📱 Opening DroidCam URL...")
        cap = cv2.VideoCapture(camera_source, cv2.CAP_FFMPEG)
    else:
        print(f"📷 Opening Camera Index {camera_source}...")
        cap = cv2.VideoCapture(camera_source)
    
    if not cap.isOpened():
        print("❌ ERROR: Cannot open camera!")
        print("\n💡 Troubleshooting:")
        if isinstance(camera_source, str):
            print("   - Pastikan DroidCam app running di HP")
            print("   - Check IP address di DroidCam app")
            print("   - HP & laptop di WiFi yang sama")
        else:
            print("   - Coba index lain (0, 1, 2)")
            print("   - Pastikan DroidCam Client running (jika pakai virtual camera)")
        return False
    
    # Get camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"✅ Camera opened successfully!")
    print(f"   Resolution: {width}x{height}")
    print(f"   FPS: {fps}")
    print(f"\n📹 Displaying preview... Press 'Q' to quit")
    print(f"   Press 'S' to save screenshot")
    
    # Display preview
    frame_count = 0
    cv2.namedWindow("DroidCam Preview", cv2.WINDOW_NORMAL)
    
    while True:
        ret, frame = cap.read()
        
        if not ret or frame is None:
            print("⚠️ Warning: Cannot read frame")
            break
        
        frame_count += 1
        
        # Add info overlay
        info_text = f"Frame: {frame_count} | Resolution: {width}x{height} | Press Q to quit, S to save"
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, (0, 255, 0), 2)
        
        # Display
        cv2.imshow("DroidCam Preview", frame)
        
        # Handle keypress
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == ord('Q'):
            print("\n🛑 Quit by user")
            break
        elif key == ord('s') or key == ord('S'):
            filename = f"droidcam_test_{frame_count}.jpg"
            cv2.imwrite(filename, frame)
            print(f"💾 Screenshot saved: {filename}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n✅ Test complete! Captured {frame_count} frames")
    return True


def main():
    """Main function"""
    print("\n" + "=" * 60)
    print("📱 DROIDCAM QUICK TEST")
    print("=" * 60)
    
    # Try to load config
    try:
        from droidcam_config import get_camera_source
        camera_source = get_camera_source()
        print(f"✅ Loaded from droidcam_config.py: {camera_source}")
    except:
        print("⚠️ droidcam_config.py not found or error")
        
        # Manual input
        print("\nOptions:")
        print("  1. Camera index (contoh: 0, 1, 2)")
        print("  2. DroidCam URL (contoh: http://192.168.1.100:4747/video)")
        
        user_input = input("\nEnter camera source: ").strip()
        
        # Parse input
        try:
            camera_source = int(user_input)
        except ValueError:
            camera_source = user_input
    
    # Run test
    success = test_camera_and_display(camera_source)
    
    if success:
        print("\n" + "=" * 60)
        print("✅ DroidCam is working!")
        print("=" * 60)
        print("\n💡 Next steps:")
        print("   1. Update droidcam_config.py dengan camera source yang benar")
        print("   2. Adjust threshold di app/config.py (turunkan ke 0.4 untuk kamera bagus)")
        print("   3. Run: python app\\app.py")
    else:
        print("\n" + "=" * 60)
        print("❌ DroidCam test failed")
        print("=" * 60)
        print("\n💡 Run scan untuk cari camera:")
        print("   python droidcam_config.py scan")


if __name__ == "__main__":
    main()
