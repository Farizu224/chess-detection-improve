"""
Test ORIGINAL model from chess-detection-original
Convert to ONNX and compare with current models
"""

import os
import sys

# Suppress warnings
os.environ['ORT_LOGGING_LEVEL'] = '3'

from ultralytics import YOLO
import torch

print("\n" + "="*70)
print("TESTING ORIGINAL MODEL")
print("="*70)

# Check if original model exists
original_model_path = "../chess-detection-original/app/model/best.pt"

if not os.path.exists(original_model_path):
    print(f"\n❌ Original model not found at: {original_model_path}")
    print("   Expected location: chess-detection-original/app/model/best.pt")
    sys.exit(1)

print(f"\n✅ Found original model: {original_model_path}")

# Load model
print("\nLoading PyTorch model...")
model = YOLO(original_model_path)

# Get model info
print(f"✅ Model loaded")
print(f"   Model type: {model.model.__class__.__name__}")

# Export to ONNX
print("\nExporting to ONNX...")
try:
    onnx_path = model.export(
        format='onnx',
        dynamic=False,
        simplify=True,
        opset=12,
    )
    print(f"✅ ONNX export successful: {onnx_path}")
    
    # Copy to test location
    import shutil
    test_onnx_path = "app/model/best_original.onnx"
    shutil.copy(onnx_path, test_onnx_path)
    print(f"✅ Copied to: {test_onnx_path}")
    
    print("\n" + "="*70)
    print("READY TO TEST!")
    print("="*70)
    print(f"\nNow test with:")
    print(f"  python compare_models.py")
    print(f"\nOr use in standalone test:")
    print(f"  Modify test_detection_standalone.py to use '{test_onnx_path}'")
    
except Exception as e:
    print(f"❌ Export failed: {e}")
    sys.exit(1)
