"""Show camera feed with detection zone overlay"""
import cv2
import numpy as np

print("=" * 70)
print("CAMERA POSITIONING HELPER")
print("=" * 70)
print("\nOpening camera with detection zone overlay...")
print("Position chess piece in the RED ZONE at bottom of frame")
print("Press 'Q' to quit\n")

cap = cv2.VideoCapture(1, cv2.CAP_ANY)
if not cap.isOpened():
    print("❌ Cannot open camera")
    exit(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocess (same as detection)
    h, w = frame.shape[:2]
    min_dim = min(h, w)
    start_x = (w - min_dim) // 2
    start_y = (h - min_dim) // 2
    cropped = frame[start_y:start_y+min_dim, start_x:start_x+min_dim]
    resized = cv2.resize(cropped, (736, 736))
    
    # Draw detection zone overlay (where model looks for pieces)
    overlay = resized.copy()
    # Detection zone: x=217-324, y=518-628
    cv2.rectangle(overlay, (217, 518), (324, 628), (0, 0, 255), 3)  # Red box
    cv2.rectangle(overlay, (217, 518), (324, 628), (0, 0, 255), -1)  # Red fill
    
    # Blend overlay
    alpha = 0.3
    output = cv2.addWeighted(overlay, alpha, resized, 1 - alpha, 0)
    
    # Add text instructions
    brightness = np.mean(resized)
    cv2.putText(output, "RED ZONE = Detection Area", (20, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(output, "Place chess piece HERE", (220, 500), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(output, f"Brightness: {brightness:.1f} (target: 80-150)", (20, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if brightness > 60 else (0, 0, 255), 2)
    cv2.putText(output, "Press 'Q' to quit", (20, 710), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Show
    cv2.imshow('Camera Positioning Helper', output)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("\n✅ Done!")
