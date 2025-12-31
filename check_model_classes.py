"""
Script untuk memeriksa class names dari model YOLO
"""
from ultralytics import YOLO

print("="*60)
print("CHECKING MODEL CLASSES")
print("="*60)

# Load model
model = YOLO('app/model/best.pt')

print("\nModel Classes:")
print("-"*60)
for k, v in model.names.items():
    print(f"  {k}: {v}")

print("\n" + "="*60)
print(f"Total Classes: {len(model.names)}")
print("="*60)
