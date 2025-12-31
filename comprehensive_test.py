"""
Script untuk testing apakah semua dependencies dan model berfungsi
"""
import sys
import os

print("="*70)
print("COMPREHENSIVE SYSTEM CHECK - CHESS DETECTION")
print("="*70)

print("\n1. Checking Python Version:")
print("-"*70)
print(f"Python: {sys.version}")

print("\n2. Checking Required Packages:")
print("-"*70)

packages_to_check = [
    'cv2',
    'numpy',
    'ultralytics',
    'sklearn',
    'scipy',
    'onnxruntime',
    'chess',
    'filterpy',
    'albumentations'
]

missing_packages = []
for package in packages_to_check:
    try:
        if package == 'cv2':
            import cv2
            print(f"  ✅ opencv-python ({cv2.__version__})")
        elif package == 'ultralytics':
            import ultralytics
            print(f"  ✅ ultralytics ({ultralytics.__version__})")
        elif package == 'sklearn':
            import sklearn
            print(f"  ✅ scikit-learn ({sklearn.__version__})")
        elif package == 'onnxruntime':
            import onnxruntime
            print(f"  ✅ onnxruntime ({onnxruntime.__version__})")
        elif package == 'chess':
            import chess
            print(f"  ✅ python-chess ({chess.__version__})")
        else:
            exec(f"import {package}")
            print(f"  ✅ {package}")
    except ImportError as e:
        print(f"  ❌ {package} - NOT INSTALLED")
        missing_packages.append(package)

print("\n3. Checking Model Files:")
print("-"*70)

model_files = [
    'app/model/best.pt',
    'app/model/best.onnx'
]

for model_file in model_files:
    if os.path.exists(model_file):
        size = os.path.getsize(model_file) / (1024 * 1024)  # MB
        print(f"  ✅ {model_file} ({size:.2f} MB)")
    else:
        print(f"  ❌ {model_file} - NOT FOUND")

print("\n4. Testing Model Load:")
print("-"*70)

try:
    from ultralytics import YOLO
    model = YOLO('app/model/best.pt')
    print(f"  ✅ Model loaded successfully")
    print(f"  ✅ Classes: {len(model.names)} classes")
    
    # Test inference on dummy image
    import numpy as np
    import cv2
    dummy_img = np.zeros((720, 720, 3), dtype=np.uint8)
    results = model(dummy_img, verbose=False)
    print(f"  ✅ Inference test passed")
    
except Exception as e:
    print(f"  ❌ Model load/inference failed: {e}")

print("\n5. Testing Chess Detection Service:")
print("-"*70)

try:
    sys.path.insert(0, 'app')
    from chess_detection import ChessDetectionService
    
    service = ChessDetectionService(model_path='app/model/best.pt', use_onnx=False)
    print(f"  ✅ ChessDetectionService initialized")
    
except Exception as e:
    print(f"  ❌ ChessDetectionService failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("SUMMARY")
print("="*70)

if missing_packages:
    print(f"❌ MISSING PACKAGES: {', '.join(missing_packages)}")
    print(f"\nInstall with:")
    print(f"  pip install {' '.join(missing_packages)}")
else:
    print("✅ ALL DEPENDENCIES ARE INSTALLED")

print("="*70)
