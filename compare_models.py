"""
MODEL COMPARISON TEST
Test all ONNX models to find which one works best
"""

import cv2
import numpy as np
import os
import time

os.environ['ORT_LOGGING_LEVEL'] = '3'
import onnxruntime as ort

class ModelTester:
    def __init__(self, camera_index=1):
        self.camera_index = camera_index
        self.class_names = {
            0: 'black_bishop', 1: 'black_king', 2: 'black_knight',
            3: 'black_pawn', 4: 'black_queen', 5: 'black_rook',
            6: 'white_bishop', 7: 'white_king', 8: 'white_knight',
            9: 'white_pawn', 10: 'white_queen', 11: 'white_rook'
        }
        
    def test_model(self, model_path, num_frames=30):
        """Test a single model"""
        print(f"\n{'='*70}")
        print(f"TESTING: {model_path}")
        print(f"{'='*70}")
        
        # Load model
        try:
            session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
            input_name = session.get_inputs()[0].name
            input_shape = session.get_inputs()[0].shape
            print(f"✅ Model loaded! Input shape: {input_shape}")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return None
        
        # Open camera
        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_ANY)
        if not cap.isOpened():
            print("❌ Cannot open camera")
            return None
        
        # Read test frame
        ret, test_frame = cap.read()
        if not ret:
            print("❌ Cannot read frame")
            cap.release()
            return None
        
        print(f"Camera: {test_frame.shape[1]}x{test_frame.shape[0]}")
        
        # Statistics
        stats = {
            'total_detections': 0,
            'high_conf_detections': 0,  # conf > 0.5
            'med_conf_detections': 0,   # conf 0.25-0.5
            'low_conf_detections': 0,   # conf 0.1-0.25
            'black_pieces': 0,
            'white_pieces': 0,
            'class_distribution': {},
            'avg_confidence': 0,
            'inference_times': []
        }
        
        print(f"\nProcessing {num_frames} frames...")
        
        for i in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Preprocess
            img = cv2.resize(frame, (input_shape[2], input_shape[3]))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))
            img = np.expand_dims(img, axis=0)
            
            # Inference
            start_time = time.time()
            outputs = session.run(None, {input_name: img})
            inference_time = time.time() - start_time
            stats['inference_times'].append(inference_time)
            
            # Parse output
            output = outputs[0]
            predictions = output[0].T
            
            for pred in predictions:
                class_probs = pred[4:]
                class_id = int(np.argmax(class_probs))
                confidence = float(class_probs[class_id])
                
                if confidence < 0.1:
                    continue
                
                stats['total_detections'] += 1
                
                # Confidence buckets
                if confidence > 0.5:
                    stats['high_conf_detections'] += 1
                elif confidence > 0.25:
                    stats['med_conf_detections'] += 1
                else:
                    stats['low_conf_detections'] += 1
                
                # Class distribution
                class_name = self.class_names[class_id]
                if class_name not in stats['class_distribution']:
                    stats['class_distribution'][class_name] = 0
                stats['class_distribution'][class_name] += 1
                
                # Color distribution
                if 'black' in class_name:
                    stats['black_pieces'] += 1
                else:
                    stats['white_pieces'] += 1
                
                stats['avg_confidence'] += confidence
        
        cap.release()
        
        # Calculate averages
        if stats['total_detections'] > 0:
            stats['avg_confidence'] /= stats['total_detections']
        
        stats['avg_inference_time'] = np.mean(stats['inference_times']) * 1000  # ms
        stats['avg_detections_per_frame'] = stats['total_detections'] / num_frames
        
        # Print results
        print(f"\n📊 RESULTS:")
        print(f"  Total detections (conf > 0.1): {stats['total_detections']}")
        print(f"  Avg detections per frame: {stats['avg_detections_per_frame']:.1f}")
        print(f"  Avg confidence: {stats['avg_confidence']:.2f}")
        print(f"  Avg inference time: {stats['avg_inference_time']:.1f} ms")
        print(f"\n  Confidence distribution:")
        print(f"    High (>0.5): {stats['high_conf_detections']}")
        print(f"    Med (0.25-0.5): {stats['med_conf_detections']}")
        print(f"    Low (0.1-0.25): {stats['low_conf_detections']}")
        print(f"\n  Color distribution:")
        print(f"    Black pieces: {stats['black_pieces']}")
        print(f"    White pieces: {stats['white_pieces']}")
        
        if stats['white_pieces'] + stats['black_pieces'] > 0:
            white_pct = stats['white_pieces'] / (stats['white_pieces'] + stats['black_pieces']) * 100
            print(f"    White %: {white_pct:.1f}%")
        
        print(f"\n  Class distribution (top 5):")
        sorted_classes = sorted(stats['class_distribution'].items(), key=lambda x: x[1], reverse=True)
        for class_name, count in sorted_classes[:5]:
            pct = count / stats['total_detections'] * 100
            print(f"    {class_name}: {count} ({pct:.1f}%)")
        
        return stats


def compare_all_models():
    """Compare all available models"""
    models = [
        'app/model/best.onnx',
        'app/model/best(v1).onnx',
        'app/model/best(v2).onnx',
        'app/model/best(v3).onnx'
    ]
    
    print("\n" + "="*70)
    print("CHESS DETECTION MODEL COMPARISON")
    print("Testing all models to find the best one")
    print("="*70)
    
    tester = ModelTester(camera_index=1)
    results = {}
    
    for model_path in models:
        if not os.path.exists(model_path):
            print(f"\n⚠️  Skipping {model_path} (not found)")
            continue
        
        stats = tester.test_model(model_path, num_frames=30)
        if stats:
            results[model_path] = stats
    
    # Print comparison
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    
    for model_path, stats in results.items():
        model_name = os.path.basename(model_path)
        white_pct = 0
        if stats['white_pieces'] + stats['black_pieces'] > 0:
            white_pct = stats['white_pieces'] / (stats['white_pieces'] + stats['black_pieces']) * 100
        
        print(f"\n{model_name}:")
        print(f"  Avg detections/frame: {stats['avg_detections_per_frame']:.1f}")
        print(f"  Avg confidence: {stats['avg_confidence']:.2f}")
        print(f"  White %: {white_pct:.1f}%")
        print(f"  Inference: {stats['avg_inference_time']:.1f} ms")
    
    # Recommend best model
    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)
    
    # Score models (lower is better)
    scores = {}
    for model_path, stats in results.items():
        white_pct = 0
        if stats['white_pieces'] + stats['black_pieces'] > 0:
            white_pct = stats['white_pieces'] / (stats['white_pieces'] + stats['black_pieces']) * 100
        
        # Ideal: ~5 pieces per frame, 50% white, high confidence
        score = 0
        score += abs(stats['avg_detections_per_frame'] - 5) * 5  # Penalty for wrong count
        score += abs(white_pct - 50) * 0.5  # Penalty for color bias
        score += (1 - stats['avg_confidence']) * 10  # Penalty for low confidence
        
        scores[model_path] = score
    
    best_model = min(scores, key=scores.get)
    print(f"\n🏆 BEST MODEL: {os.path.basename(best_model)}")
    print(f"   Score: {scores[best_model]:.2f} (lower is better)")
    print(f"\n   Use this model for best results!")


if __name__ == "__main__":
    compare_all_models()
