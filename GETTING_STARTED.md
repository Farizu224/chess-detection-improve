# 🚀 GETTING STARTED - Langkah Pertama

## 📌 Ringkasan Scope Improvement

Berdasarkan keputusan Anda, improvement difokuskan pada:

### ✅ 1. Speed Optimization (PRIORITAS 1)
- ONNX Export (30-50% faster inference)
- Frame Skipping (proses tiap 3-5 frame)
- ❌ SKIP: TensorRT/Quantization (terlalu risky)

### ✅ 2. Logic Enhancement (PRIORITAS 2)
- FEN Validation (chess rules)
- Temporal Smoothing (stabil 5 frame)

### ✅ 3. Data Augmentation (PRIORITAS 3)
- HSV, rotation, mosaic, mixup
- ❌ SKIP: Ensemble prediction
- ❌ SKIP: ArUco markers

---

## 🎯 LANGKAH PERTAMA: MODEL TRAINING

### Prerequisites
Sebelum mulai, pastikan Anda punya:
- [ ] Python 3.8+ installed
- [ ] CUDA + cuDNN (jika punya GPU NVIDIA)
- [ ] Minimal 8GB RAM (16GB recommended)
- [ ] 10GB free disk space

### Step-by-Step Guide

#### 1️⃣ Open Training Notebook
```bash
# Buka file yang sudah dibuat
cd d:\Documents\Project\Tugas\Tugas_Semester_5\Computer_Vision\Tugas_Besar\Chess_Detection\model
jupyter notebook chess_detection_improve.ipynb

# ATAU gunakan VS Code (recommended)
code chess_detection_improve.ipynb
```

#### 2️⃣ Install Dependencies
Jalankan cell pertama untuk install semua library:
```python
!pip install roboflow ultralytics albumentations onnx onnxruntime
```

**⚠️ PENTING:**
- Jika punya GPU: Install `onnxruntime-gpu` bukan `onnxruntime`
- Check CUDA compatibility dengan PyTorch Anda

#### 3️⃣ Download Datasets (2 Sources)
Anda akan menggabungkan 2 dataset untuk mendapatkan data yang lebih banyak dan diverse:

**Dataset 1: Original (kosan-hendra)**
```python
from roboflow imprf2 = Roboflow(api_key="isi_dengan_api_key_anda")

# Download dataset 1
rf1 = Roboflow(api_key="jDh0EVC94eG10ly0jiAY")
project1 = rf1.workspace("kosan-hendra").project("chess_detection-uzejh")
version1 = project1.version(2)
dataset1 = version1.download("yolov8")
print(f"Dataset 1 downloaded to: {dataset1.location}")

# Download dataset 2 (PUNYA ANDA)
rf2 = Roboflow(api_key="YOUR_API_KEY_HERE")  # 🔑 Ganti dengan API key Anda
project2 = rf2.workspace("detection-chess").project("chess-detection-76mfe")
version2 = project2.version(1)  # Sesuaikan version number
dataset2 = version2.download("yolov8")
print(f"Dataset 2 downloaded to: {dataset2.location}")
```

**⚠️ PENTING:** Ganti `YOUR_API_KEY_HERE` dengan API key Anda dari Roboflow.

**Expected output:**
```
✅ Dataset 1 downloaded to: /path/to/chess_detection-2
✅ Dataset 2 downloaded to: /path/to/chess-detection-76mfe-1
```

#### 4️⃣ Merge Datasets (PENTING!)
Gabungkan kedua dataset menjadi satu:

```python
import os
import shutil
from pathlib import Path

# Create merged dataset folder
merged_dataset_path = "chess_detection_merged"
os.makedirs(merged_dataset_path, exist_ok=True)

for split in ['train', 'valid', 'test']:
    os.makedirs(f"{merged_dataset_path}/{split}/images", exist_ok=True)
    os.makedirs(f"{merged_dataset_path}/{split}/labels", exist_ok=True)

# Function to copy files with prefix to avoid conflicts
def copy_dataset(source_path, dest_path, prefix):
    for split in ['train', 'valid']:
        # Copy images
        img_src = f"{source_path}/{split}/images"
        img_dst = f"{dest_path}/{split}/images"
        
        if os.path.exists(img_src):
            for img_file in os.listdir(img_src):
                src_file = os.path.join(img_src, img_file)
                # Add prefix to avoid duplicate filenames
                new_filename = f"{prefix}_{img_file}"
                dst_file = os.path.join(img_dst, new_filename)
                shutil.copy2(src_file, dst_file)
        
        # Copy labels
        lbl_src = f"{source_path}/{split}/labels"
        lbl_dst = f"{dest_path}/{split}/labels"
        
        if os.path.exists(lbl_src):
            for lbl_file in os.listdir(lbl_src):
                src_file = os.path.join(lbl_src, lbl_file)
                # Same prefix as images
                new_filename = f"{prefix}_{lbl_file}"
                dst_file = os.path.join(lbl_dst, new_filename)
                shutil.copy2(src_file, dst_file)
    
    print(f"✅ Copied {prefix} dataset")

# Copy both datasets
copy_dataset(dataset1.location, merged_dataset_path, "ds1")
copy_dataset(dataset2.location, merged_dataset_path, "ds2")

print("\n📊 Merged Dataset Statistics:")
for split in ['train', 'valid']:
    img_count = len(os.listdir(f"{merged_dataset_path}/{split}/images"))
    lbl_count = len(os.listdir(f"{merged_dataset_path}/{split}/labels"))
    print(f"  {split.capitalize()}: {img_count} images, {lbl_count} labels")
```

#### 5️⃣ Create data.yaml for Merged Dataset
```python
import yaml

# Read original data.yaml to get class names
with open(f"{dataset1.location}/data.yaml", 'r') as f:
    original_config = yaml.safe_load(f)

# Create new data.yaml for merged dataset
merged_config = {
    'path': os.path.abspath(merged_dataset_path),
    'train': 'train/images',
    'val': 'valid/images',
    'nc': original_config['nc'],
    'names': original_config['names']
}

# Save merged data.yaml
merged_yaml_path = f"{merged_dataset_path}/data.yaml"
with open(merged_yaml_path, 'w') as f:
    yaml.dump(merged_config, f, default_flow_style=False)

print(f"\n✅ data.yaml created at: {merged_yaml_path}")
print(f"\n📋 Dataset Configuration:")
print(f"   - Train images: {len(os.listdir(f'{merged_dataset_path}/train/images'))}")
print(f"   - Val images: {len(os.listdir(f'{merged_dataset_path}/valid/images'))}")
print(f"   - Classes ({merged_config['nc']}): {merged_config['names']}")
```

**Decision Point:**
- Jika **total < 800 images**: Jalankan augmentation (next cell)
- Jika **total > 1500 images**: SKIP augmentation, langsung training

#### 6️⃣ Start Training
Jalankan training cell dengan merged dataset:
```python
from ultralytics import YOLO
model = YOLO("yolov8s.pt")

results = model.train(
    data=merged_yaml_path,  # 🔥 Gunakan merged dataset
    epochs=150,
    patience=30,
    batch=16,  # Sesuaikan dengan GPU memory
    imgsz=720,
    device=0,  # 0 = GPU, 'cpu' = CPU
    project="chess_detection_improved",
    name="yolov8s_v1",
)
```

**⏱️ Expected Duration:**
- GPU (RTX 3060): ~6-8 jam
- GPU (RTX 4090): ~3-4 jam
- CPU: ~24-48 jam (TIDAK RECOMMENDED)

**💡 TIP:** Training bisa di-run overnight. Monitor progress di TensorBoard:
```bash
tensorboard --logdir chess_detection_improved/yolov8s_v1
```

#### 6️⃣ Monitor Training
Selama training, check metrics:
- **Loss harus turun** (train & val)
- **mAP harus naik**
- **Patience counter** (jika stuck di 30, akan auto-stop)

**Red Flags:**
- ❌ Val loss naik terus (overfitting)
- ❌ mAP stuck di <50% (dataset problem)
- ❌ Training crash (memory issue, reduce batch size)

#### 7️⃣ Export to ONNX
Setelah training selesai, jalankan export cell (cell 7):
```python
onnx_path = model.export(format='onnx', simplify=True)
print(f"✅ ONNX model saved: {onnx_path}")
```

#### 8️⃣ Test Inference Speed
Jalankan benchmark cell (cell 8) untuk bandingkan PyTorch vs ONNX:
```python
# Expected output:
# PyTorch: 80ms/frame (12 FPS)
# ONNX: 40ms/frame (25 FPS)
# 🚀 ONNX is 2.0x faster!
```

---

## 📊 Expected Results (After Training)

### Minimum Acceptable
- mAP@50: **>88%** (jika <88%, ada problem di dataset/training)
- Inference Time (ONNX): **<50ms** per frame
- Model Size: ~22MB (yolov8s)

### Target Results
- mAP@50: **90-92%**
- mAP@50-95: **70-75%**
- Inference Time (ONNX): **30-40ms**
- Per-class AP: **>85%** untuk semua piece types

### Troubleshooting

#### Problem 1: Training sangat lambat
**Solution:**
- Reduce `batch=16` → `batch=8` atau `batch=4`
- Reduce `imgsz=720` → `imgsz=640`
- Reduce `epochs=150` → `epochs=100`

#### Problem 2: mAP stuck di <70%
**Solution:**
- Check dataset labels (mungkin ada yang salah)
- Reduce learning rate: `lr0=0.0005`
- Increase augmentation: `hsv_v=0.6`

#### Problem 3: Out of Memory (OOM)
**Solution:**
- Reduce batch size: `batch=8` → `batch=4`
- Close other applications
- Use `device='cpu'` (last resort, sangat lambat)

#### Problem 4: ONNX export gagal
**Solution:**
- Update ultralytics: `pip install --upgrade ultralytics`
- Try different opset: `opset=11` instead of `opset=12`
- SKIP ONNX for now, use PyTorch (still get gains from other improvements)

---

## 🗂️ File Locations (After Training)

Training akan generate struktur berikut:
```
chess_detection_improved/
└── yolov8s_v1/
    ├── weights/
    │   ├── best.pt          # ✅ MODEL TERBAIK (copy ini)
    │   ├── best.onnx        # ✅ ONNX version (copy ini)
    │   └── last.pt          # Model terakhir (backup)
    ├── results.png          # Training curves
    ├── confusion_matrix.png # Per-class accuracy
    └── F1_curve.png         # F1 scores
```

---

## 📋 Next Steps (Setelah Training Selesai)

### ✅ Checklist
- [ ] Training completed (mAP >88%)
- [ ] ONNX export successful
- [ ] Speed test passed (ONNX >2x faster)
- [ ] Copy `best.pt` dan `best.onnx` ke folder project

### 🎯 Hari 1 Selesai!
Setelah checklist di atas complete, lanjut ke **Day 2**:
1. Copy model ke `chess-detection-improve/app/model/`
2. Implement ONNX inference di `chess_detection.py`
3. Implement frame skipping
4. Implement FEN validation

---

## 🆘 Need Help?

### Common Issues & Solutions

**Q: Training stuck (tidak progress)**
A: Restart kernel, clear GPU memory: `torch.cuda.empty_cache()`

**Q: Hasil mAP sangat rendah (<60%)**
A: Problem di dataset, check labels di Roboflow

**Q: ONNX tidak bisa load**
A: Check ONNX Runtime version compatibility, try CPU provider first

**Q: Kagak punya GPU, gimana?**
A: 
- Option 1: Pakai Google Colab (free GPU)
- Option 2: Training di CPU (lama tapi bisa)
- Option 3: Pakai model original (skip training, fokus ke improvements lain)

---

## 📝 Summary

### Yang HARUS Dikerjakan Hari Ini:
1. ✅ Buka `chess_detection_improve.ipynb`
2. ✅ Install dependencies
3. ✅ Download dataset
4. ✅ Start training (biarkan running)
5. ✅ Monitor progress

### Yang BISA Dikerjakan Sambil Training:
1. Copy files dari `chess-detection/` ke `chess-detection-improve/`
2. Baca dan pahami original code
3. Plan implementation untuk Day 2

### Expected Timeline:
- **Setup & start training**: 1-2 jam
- **Training duration**: 6-8 jam (overnight)
- **Export & testing**: 30 menit

**Total Day 1**: ~2-3 jam active work + 6-8 jam passive (training)

---

## 🎯 Key Differences vs Original

| Aspect | Original | Improved | Why? |
|--------|----------|----------|------|
| **Base Model** | YOLOv8n | YOLOv8s | +7% accuracy, still fast |
| **Epochs** | 100 | 150 | More learning |
| **Patience** | 20 | 30 | Better early stopping |
| **Optimizer** | SGD | AdamW | Better for small datasets |
| **Augmentation** | Default | Enhanced (HSV, mosaic, mixup) | Robustness |
| **Export** | PyTorch only | PyTorch + ONNX | 30-50% faster |
| **Image Size** | 640 | 720 | Match real-world use case |

---

**🚀 Ready to Start? Open the notebook and let's go! ♟️**

---

## 📞 Contact

Jika ada yang bingung atau stuck, dokumentasikan:
1. Error message lengkap
2. Cell yang dijalankan
3. Dataset size & GPU yang dipakai

Good luck! 💪
