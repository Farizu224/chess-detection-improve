import sys
sys.path.insert(0, 'app')
from chess_detection import ChessDetectionService

print("="*70)
print("CHECKING CURRENT CONFIGURATION")
print("="*70)

s = ChessDetectionService()

print(f"\nuse_onnx setting: {s.use_onnx}")
print(f"Model loaded: {type(s.model).__name__ if s.model else 'None'}")
print(f"ONNX Engine: {type(s.inference_engine).__name__ if s.inference_engine else 'None'}")

if s.use_onnx and s.inference_engine:
    print("\n✅ ONNX is configured and loaded")
elif s.use_onnx and s.model:
    print("\n⚠️ ONNX configured but fell back to PyTorch!")
    print("   This is why FPS is still slow!")
elif not s.use_onnx and s.model:
    print("\n⚠️ Using PyTorch mode (slow!)")
else:
    print("\n❌ No model loaded!")

print("="*70)
