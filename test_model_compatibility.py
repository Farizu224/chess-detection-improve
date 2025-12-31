"""
Script untuk test deteksi dengan model dan melihat output class names
"""
from ultralytics import YOLO
import cv2
import numpy as np

print("="*70)
print("TESTING MODEL DETECTION AND CLASS NAMES")
print("="*70)

# Load model
model = YOLO('app/model/best.pt')

print("\n1. Model Classes (from model.names):")
print("-"*70)
for k, v in model.names.items():
    print(f"  Class {k}: {v}")

print("\n2. Expected Classes in Code (from chess_detection.py):")
print("-"*70)
expected_classes = {
    'white_king': 'K',
    'white_queen': 'Q', 
    'white_rook': 'R',
    'white_bishop': 'B',
    'white_knight': 'N',
    'white_pawn': 'P',
    'black_king': 'k',
    'black_queen': 'q',
    'black_rook': 'r', 
    'black_bishop': 'b',
    'black_knight': 'n',
    'black_pawn': 'p'
}

for class_name, fen_piece in expected_classes.items():
    is_in_model = class_name in model.names.values()
    status = "✅" if is_in_model else "❌"
    print(f"  {status} {class_name} -> {fen_piece}")

print("\n3. Checking for mismatches:")
print("-"*70)
model_classes = set(model.names.values())
expected_class_names = set(expected_classes.keys())

missing_in_model = expected_class_names - model_classes
extra_in_model = model_classes - expected_class_names

if missing_in_model:
    print(f"  ❌ Classes in code but MISSING in model: {missing_in_model}")
else:
    print(f"  ✅ All expected classes are present in model")

if extra_in_model:
    print(f"  ⚠️  Extra classes in model not in code: {extra_in_model}")
else:
    print(f"  ✅ No extra classes in model")

print("\n4. Class Name Format Check:")
print("-"*70)
for class_name in model.names.values():
    # Check if format is color_piece (e.g., white_knight)
    parts = class_name.split('_')
    if len(parts) == 2:
        color, piece = parts
        if color in ['white', 'black'] and piece in ['king', 'queen', 'rook', 'bishop', 'knight', 'pawn']:
            print(f"  ✅ {class_name} - Format is correct (color_piece)")
        else:
            print(f"  ⚠️  {class_name} - Format unusual")
    else:
        print(f"  ❌ {class_name} - Format WRONG (expected: color_piece)")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)

all_match = (not missing_in_model) and (not extra_in_model)
if all_match:
    print("✅ MODEL AND CODE ARE COMPATIBLE!")
    print("   All class names match perfectly.")
else:
    print("❌ MISMATCH DETECTED!")
    print("   Please update either your model or code.")

print("="*70)
