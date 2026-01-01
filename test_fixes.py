"""
Quick test to verify critical fixes are working
"""
import sys
sys.path.insert(0, 'chess-detection-improve')

print("=" * 60)
print("TESTING CRITICAL FIXES")
print("=" * 60)

# Test 1: ONNX Engine class_names initialization
print("\n1. Testing ONNX Engine class_names...")
try:
    from app.onnx_engine import ONNXInferenceEngine
    
    # Test with string path
    engine = ONNXInferenceEngine('chess-detection-improve/app/model/best.onnx', 
                                 'chess-detection-improve/app/model/best.pt')
    
    if engine.class_names is not None:
        print(f"   ✅ Class names loaded: {len(engine.class_names)} classes")
        print(f"   Sample classes: {list(engine.class_names.values())[:3]}")
    else:
        print("   ❌ Class names is None!")
        
except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Check if ChessDetection can initialize
print("\n2. Testing ChessDetection initialization...")
try:
    from app.chess_detection import ChessDetectionService
    
    service = ChessDetectionService()
    
    if service.inference_engine and hasattr(service.inference_engine, 'class_names'):
        if service.inference_engine.class_names is not None:
            print(f"   ✅ ChessDetection has class_names: {len(service.inference_engine.class_names)} classes")
        else:
            print("   ❌ ChessDetection class_names is None!")
    else:
        print("   ⚠️ Using PyTorch model (no inference_engine)")
        
except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)
