"""
DATASET ANALYSIS - Check for Class Imbalance
Analyze training dataset to find why model is biased to black pieces
"""

import os
import yaml
from collections import Counter

def analyze_dataset(dataset_path):
    """Analyze dataset for class distribution"""
    
    # Read data.yaml to get class names
    with open(f"{dataset_path}/data.yaml", 'r') as f:
        data_config = yaml.safe_load(f)
    
    class_names = data_config['names']
    print(f"\n{'='*70}")
    print(f"DATASET ANALYSIS: {dataset_path}")
    print(f"{'='*70}\n")
    print(f"Classes ({len(class_names)}):")
    for i, name in enumerate(class_names):
        print(f"  {i}: {name}")
    
    # Analyze train and valid splits
    for split in ['train', 'valid']:
        labels_dir = f"{dataset_path}/{split}/labels"
        
        if not os.path.exists(labels_dir):
            print(f"\n⚠️ {split} labels not found")
            continue
        
        print(f"\n{'='*70}")
        print(f"{split.upper()} SPLIT ANALYSIS")
        print(f"{'='*70}")
        
        # Count instances per class
        class_counts = Counter()
        total_instances = 0
        file_count = 0
        
        # Count images with each color
        files_with_white = 0
        files_with_black = 0
        
        for label_file in os.listdir(labels_dir):
            if not label_file.endswith('.txt'):
                continue
            
            file_count += 1
            file_path = os.path.join(labels_dir, label_file)
            
            has_white = False
            has_black = False
            
            with open(file_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    
                    class_id = int(parts[0])
                    if 0 <= class_id < len(class_names):
                        class_counts[class_id] += 1
                        total_instances += 1
                        
                        # Check color
                        if 'white' in class_names[class_id]:
                            has_white = True
                        elif 'black' in class_names[class_id]:
                            has_black = True
            
            if has_white:
                files_with_white += 1
            if has_black:
                files_with_black += 1
        
        # Print statistics
        print(f"\nTotal files: {file_count}")
        print(f"Total instances: {total_instances}")
        print(f"Avg instances per file: {total_instances / file_count if file_count > 0 else 0:.2f}")
        
        print(f"\nFiles with pieces:")
        print(f"  White pieces: {files_with_white} ({files_with_white/file_count*100:.1f}%)")
        print(f"  Black pieces: {files_with_black} ({files_with_black/file_count*100:.1f}%)")
        
        # Calculate white/black ratio
        white_count = sum(class_counts[i] for i in range(len(class_names)) 
                         if 'white' in class_names[i])
        black_count = sum(class_counts[i] for i in range(len(class_names)) 
                         if 'black' in class_names[i])
        
        print(f"\nPiece distribution:")
        print(f"  White pieces: {white_count} ({white_count/(white_count+black_count)*100:.1f}%)")
        print(f"  Black pieces: {black_count} ({black_count/(white_count+black_count)*100:.1f}%)")
        
        if white_count < black_count * 0.8:
            print(f"  ⚠️ WARNING: White pieces underrepresented!")
            print(f"     Ratio: {white_count/black_count:.2f}:1 (should be ~1:1)")
        elif black_count < white_count * 0.8:
            print(f"  ⚠️ WARNING: Black pieces underrepresented!")
            print(f"     Ratio: {black_count/white_count:.2f}:1 (should be ~1:1)")
        else:
            print(f"  ✅ Color balance OK")
        
        # Per-class distribution
        print(f"\nPer-class distribution:")
        
        # Sort by count (descending)
        sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
        
        for class_id, count in sorted_classes:
            pct = count / total_instances * 100
            class_name = class_names[class_id]
            bar = '█' * int(pct / 2)  # Visual bar
            print(f"  {class_name:20s}: {count:5d} ({pct:5.1f}%) {bar}")
        
        # Check for severely underrepresented classes
        avg_count = total_instances / len(class_names)
        print(f"\nAverage instances per class: {avg_count:.1f}")
        
        underrepresented = []
        for class_id, count in class_counts.items():
            if count < avg_count * 0.5:  # Less than 50% of average
                underrepresented.append((class_names[class_id], count, avg_count))
        
        if underrepresented:
            print(f"\n⚠️ UNDERREPRESENTED CLASSES (< 50% average):")
            for class_name, count, avg in underrepresented:
                print(f"  {class_name}: {count} (avg: {avg:.1f})")
        else:
            print(f"\n✅ All classes reasonably represented")


if __name__ == "__main__":
    # Try both possible dataset locations
    possible_paths = [
        "Chess-Detection-76mfe-3",  # Typical Roboflow download name
        "../Chess-Detection-76mfe-3",
        "/content/Chess-Detection-76mfe-3",  # Colab path
    ]
    
    found = False
    for path in possible_paths:
        if os.path.exists(path):
            analyze_dataset(path)
            found = True
            break
    
    if not found:
        print("\n❌ Dataset not found!")
        print("   Please provide dataset path manually")
        print("\n   Usage:")
        print("   python dataset_analysis.py <path_to_dataset>")
