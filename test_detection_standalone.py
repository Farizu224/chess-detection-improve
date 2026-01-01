"""
STANDALONE CHESS DETECTION TEST
Test camera + ONNX detection TANPA web framework
Jika ini berhasil, baru implementasi ke web app
"""

import cv2
import numpy as np
import time
import os

# Suppress ONNX warnings
os.environ['ORT_LOGGING_LEVEL'] = '3'

import onnxruntime as ort

class SimpleChessDetector:
    def __init__(self, model_path, camera_index=1):
        self.camera_index = camera_index
        self.model_path = model_path
        
        # Load ONNX model
        print(f"Loading ONNX model: {model_path}")
        self.session = ort.InferenceSession(
            model_path,
            providers=['CPUExecutionProvider']
        )
        
        # Get input details
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        print(f"✅ Model loaded! Input shape: {self.input_shape}")
        
        # Class names
        self.class_names = {
            0: 'black_bishop', 1: 'black_king', 2: 'black_knight',
            3: 'black_pawn', 4: 'black_queen', 5: 'black_rook',
            6: 'white_bishop', 7: 'white_king', 8: 'white_knight',
            9: 'white_pawn', 10: 'white_queen', 11: 'white_rook'
        }
        
        # Colors for visualization
        self.colors = {
            'white': (255, 255, 255),
            'black': (0, 0, 0),
            'box_white': (0, 255, 0),
            'box_black': (255, 0, 0)
        }
    
    def preprocess(self, frame):
        """Preprocess frame for ONNX model"""
        # Resize to model input size
        img = cv2.resize(frame, (self.input_shape[2], self.input_shape[3]))
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # Transpose to CHW format
        img = np.transpose(img, (2, 0, 1))
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img
    
    def postprocess(self, outputs, orig_shape, conf_threshold=0.25, debug=False):
        """Postprocess ONNX outputs to get bounding boxes"""
        # Get output tensor
        output = outputs[0]
        
        # YOLOv8 output format: [1, 17, N] where N is number of detections
        # 17 = 4 (bbox) + 1 (obj_conf) + 12 (class_probs)
        predictions = output[0].T  # Transpose to [N, 17]
        
        detections = []
        all_detections = []  # Track ALL detections for debugging
        h_orig, w_orig = orig_shape[:2]
        h_model, w_model = self.input_shape[2], self.input_shape[3]
        
        for pred in predictions:
            # Extract bbox coordinates (xywh format)
            x_center, y_center, width, height = pred[:4]
            
            # Extract class probabilities
            class_probs = pred[4:]
            class_id = int(np.argmax(class_probs))
            confidence = float(class_probs[class_id])
            
            # Track all detections for debugging
            if debug and confidence > 0.1:
                all_detections.append({
                    'class_name': self.class_names[class_id],
                    'confidence': confidence
                })
            
            if confidence < conf_threshold:
                continue
            
            # Convert xywh to xyxy
            x1 = (x_center - width / 2) / w_model * w_orig
            y1 = (y_center - height / 2) / h_model * h_orig
            x2 = (x_center + width / 2) / w_model * w_orig
            y2 = (y_center + height / 2) / h_model * h_orig
            
            detections.append({
                'class_id': class_id,
                'class_name': self.class_names[class_id],
                'confidence': confidence,
                'bbox': [int(x1), int(y1), int(x2), int(y2)]
            })
        
        if debug and all_detections:
            print(f"\n📊 ALL DETECTIONS (conf > 0.1): {len(all_detections)}")
            # Count by class
            class_counts = {}
            for det in all_detections:
                class_name = det['class_name']
                if class_name not in class_counts:
                    class_counts[class_name] = 0
                class_counts[class_name] += 1
            for class_name, count in sorted(class_counts.items()):
                print(f"  {class_name}: {count}")
            print(f"✅ PASSED threshold (conf > {conf_threshold}): {len(detections)}\n")
        
        return detections
    
    def draw_detections(self, frame, detections):
        """Draw bounding boxes on frame"""
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            class_name = det['class_name']
            confidence = det['confidence']
            
            # Choose color based on piece color
            color = self.colors['box_white'] if 'white' in class_name else self.colors['box_black']
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label with background
            label = f"{class_name.replace('_', ' ')}: {confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            # Draw label background
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            # Draw label text
            cv2.putText(frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def run(self):
        """Main detection loop"""
        print(f"\n{'='*60}")
        print("STARTING STANDALONE CHESS DETECTION")
        print(f"{'='*60}\n")
        
        # Open camera - MINIMAL SETTINGS
        print(f"Opening camera {self.camera_index}...")
        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_ANY)
        
        if not cap.isOpened():
            print("❌ Error: Could not open camera")
            return
        
        # Read test frame FIRST
        print("Reading test frame...")
        ret, test_frame = cap.read()
        
        if not ret or test_frame is None:
            print("❌ Error: Could not read test frame")
            cap.release()
            return
        
        print(f"✅ Camera working! Native resolution: {test_frame.shape[1]}x{test_frame.shape[0]}")
        
        # DON'T change resolution - use native!
        print("Using native camera resolution (no property changes)")
        
        # Create window
        cv2.namedWindow('Chess Detection - Standalone', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Chess Detection - Standalone', 1280, 720)
        
        # FPS tracking
        fps = 0
        frame_count = 0
        start_time = time.time()
        debug_mode = True  # Show debug info first 60 frames
        
        print("\n" + "="*60)
        print("DETECTION RUNNING - Press 'Q' to quit")
        print("Confidence threshold: 0.25 (lowered to catch more pieces)")
        print("="*60 + "\n")
        
        while True:
            # Read frame
            ret, frame = cap.read()
            
            if not ret or frame is None:
                print("Warning: Could not read frame, skipping...")
                time.sleep(0.01)
                continue
            
            frame_count += 1
            
            # Disable debug mode after 60 frames
            if frame_count == 60:
                debug_mode = False
                print("\n⚡ Debug mode OFF (less console spam)\n")
            
            # Calculate FPS every 30 frames
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                if not debug_mode:
                    print(f"FPS: {fps:.1f} | Frames processed: {frame_count}")
            
            try:
                # Preprocess
                input_tensor = self.preprocess(frame)
                
                # Run inference
                outputs = self.session.run(None, {self.input_name: input_tensor})
                
                # Postprocess (with debug for first 60 frames)
                detections = self.postprocess(outputs, frame.shape, debug=debug_mode)
                
                # Draw detections
                frame = self.draw_detections(frame, detections)
                
                # Draw info
                info_text = f"FPS: {fps:.1f} | Pieces: {len(detections)}"
                cv2.putText(frame, info_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
            except Exception as e:
                # Silent error handling - don't spam console
                if frame_count % 30 == 0:
                    print(f"Detection error: {e}")
            
            # Show frame
            cv2.imshow('Chess Detection - Standalone', frame)
            
            # Check for quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q') or key == 27:  # Q or ESC
                print("\nQuitting...")
                break
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Final stats
        elapsed = time.time() - start_time
        avg_fps = frame_count / elapsed
        print(f"\n{'='*60}")
        print(f"DETECTION STOPPED")
        print(f"Total frames: {frame_count}")
        print(f"Average FPS: {avg_fps:.1f}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    # Configuration
    MODEL_PATH = "app/model/best.onnx"
    CAMERA_INDEX = 1  # Change this if needed
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model not found at {MODEL_PATH}")
        exit(1)
    
    # Create detector and run
    detector = SimpleChessDetector(MODEL_PATH, CAMERA_INDEX)
    detector.run()
