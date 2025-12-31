"""
Script untuk memeriksa class names dari model YOLO LAMA
"""
from ultralytics import YOLO

print("="*60)
print("CHECKING OLD MODEL CLASSES")
print("="*60)

try:
    # Load model lama
    model = YOLO('../chess-detection/app/model/best.pt')

    print("\nOLD Model Classes:")
    print("-"*60)
    for k, v in model.names.items():
        print(f"  {k}: {v}")

    print("\n" + "="*60)
    print(f"Total Classes: {len(model.names)}")
    print("="*60)
except Exception as e:
    print(f"Error loading old model: {e}")
