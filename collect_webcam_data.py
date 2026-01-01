"""
SOLUTION: Collect Real-World Data from YOUR Webcam
Capture images from your actual setup untuk improve model
"""

import cv2
import os
import time
from datetime import datetime

def capture_training_data(camera_index=1, save_dir="webcam_captures"):
    """Capture images from webcam for additional training data"""
    
    print("\n" + "="*70)
    print("WEBCAM DATA COLLECTION FOR RE-TRAINING")
    print("="*70)
    print("\nInstructions:")
    print("  1. Set up your chessboard in normal position")
    print("  2. Press SPACEBAR to capture image")
    print("  3. Move pieces around, change lighting, angle")
    print("  4. Capture ~50-100 images in different configurations")
    print("  5. Press 'Q' to quit")
    print("\n💡 Goal: Capture YOUR real-world setup variations")
    print("   - Different piece positions")
    print("   - Different lighting conditions")
    print("   - Small camera angle changes")
    print("   - Both white and black pieces visible")
    print("="*70 + "\n")
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Open camera
    cap = cv2.VideoCapture(camera_index, cv2.CAP_ANY)
    if not cap.isOpened():
        print(f"❌ Cannot open camera {camera_index}")
        return
    
    # Read test frame
    ret, test_frame = cap.read()
    if not ret:
        print("❌ Cannot read frame")
        cap.release()
        return
    
    print(f"✅ Camera opened: {test_frame.shape[1]}x{test_frame.shape[0]}")
    print(f"📁 Saving to: {save_dir}/\n")
    
    # Create window
    cv2.namedWindow('Data Collection - Press SPACE to capture', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Data Collection - Press SPACE to capture', 1280, 720)
    
    capture_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Warning: Frame read failed")
            continue
        
        # Draw instructions
        display_frame = frame.copy()
        cv2.putText(display_frame, f"Captures: {capture_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display_frame, "SPACE: Capture | Q: Quit", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show frame
        cv2.imshow('Data Collection - Press SPACE to capture', display_frame)
        
        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):  # Spacebar
            # Save image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"capture_{timestamp}_{capture_count:04d}.jpg"
            filepath = os.path.join(save_dir, filename)
            
            cv2.imwrite(filepath, frame)
            capture_count += 1
            
            print(f"✅ Captured: {filename}")
            
            # Visual feedback
            cv2.putText(display_frame, "CAPTURED!", (frame.shape[1]//2 - 100, frame.shape[0]//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
            cv2.imshow('Data Collection - Press SPACE to capture', display_frame)
            cv2.waitKey(200)  # Show feedback for 200ms
        
        elif key == ord('q') or key == ord('Q') or key == 27:
            print(f"\n🏁 Collection finished!")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Summary
    print("\n" + "="*70)
    print("COLLECTION SUMMARY")
    print("="*70)
    print(f"Total captures: {capture_count}")
    print(f"Saved to: {save_dir}/")
    
    if capture_count > 0:
        print("\n📋 NEXT STEPS:")
        print("  1. Annotate images using Roboflow/LabelImg")
        print("  2. Upload to Roboflow as NEW VERSION")
        print("  3. Re-train model with this real-world data")
        print("  4. New model will perform MUCH better on your setup!")
        print("\n💡 Alternative (Faster):")
        print("  - Use existing model from chess-detection-original")
        print("  - That model might have better real-world training data")
    else:
        print("\n⚠️ No images captured!")
    
    print("="*70)


if __name__ == "__main__":
    capture_training_data(camera_index=1, save_dir="webcam_training_data")
