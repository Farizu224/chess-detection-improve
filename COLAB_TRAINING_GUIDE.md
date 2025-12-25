# 🚀 GOOGLE COLAB TRAINING GUIDE

## 📌 Overview
Panduan lengkap training model di Google Colab (FREE GPU!) untuk proyek Chess Detection Anda.

---

## ✅ Keuntungan Training di Colab

### 🎁 FREE!
- ✅ GPU Tesla T4 gratis (15GB VRAM)
- ✅ Tidak perlu setup CUDA/cuDNN
- ✅ Tidak perlu install dependencies di laptop
- ✅ Save langsung ke Google Drive

### ⚡ Faster
- GPU T4: Training ~4-6 jam (vs CPU 24-48 jam)
- GPU V100: Training ~2-3 jam (jika dapat)
- Batch size lebih besar = convergence lebih cepat

### 🛡️ Safe
- Laptop tidak panas
- Model auto-save ke Google Drive
- Bisa disconnect tanpa khawatir (training tetap jalan)

---

## 🎯 STEP-BY-STEP GUIDE

### Step 1: Upload Notebook ke Colab

**Option A: Upload File** (RECOMMENDED)
1. Buka https://colab.research.google.com/
2. File → Upload notebook
3. Pilih: `chess_detection_improve.ipynb`

**Option B: Google Drive**
1. Upload notebook ke Google Drive Anda
2. Double click → Open with → Google Colaboratory

---

### Step 2: Enable GPU (CRITICAL!)

```
1. Runtime → Change runtime type
2. Hardware accelerator → GPU (T4 atau V100)
3. Save
```

**⚠️ WAJIB! Tanpa GPU, training akan 10x lebih lambat!**

---

### Step 3: Run Cells Berurutan

#### Cell 1: Google Colab Setup
```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')
```
- Click link yang muncul
- Login dengan Google account
- Copy authorization code
- Paste ke Colab

**Expected output:**
```
Mounted at /content/drive
🖥️ GPU Available: True
   GPU: Tesla T4
```

✅ **Verify GPU aktif!** Jika False, ulangi Step 2.

---

#### Cell 2: Install Dependencies
```python
!pip install -q roboflow ultralytics albumentations onnx onnxruntime tqdm
```

**Duration:** 1-2 menit

**Expected output:**
```
✅ All dependencies installed!
```

---

#### Cell 3: Download 2 Datasets

```python
# Dataset 1: kosan-hendra (original)
# Dataset 2: detection-chess (ANDA)
```

**Duration:** 5-10 menit (tergantung internet Colab)

**Expected output:**
```
✅ Dataset 1 downloaded to: /content/chess_detection-2
✅ Dataset 2 downloaded to: /content/chess-detection-76mfe-1
```

⚠️ **Jika error "API key invalid"**: Check API key di cell sudah benar (`9ip0woBWDJ4vucWT4GfN`)

---

#### Cell 4: Merge Datasets

**Duration:** 30 detik - 2 menit

**Expected output:**
```
📊 MERGED DATASET STATISTICS
Dataset 1 contribution: XXX images
Dataset 2 contribution: XXX images
TOTAL: XXX images
```

✅ **Verify total > 800 images untuk hasil optimal**

---

#### Cell 5: Create data.yaml

**Expected output:**
```
✅ data.yaml created
📋 Final Dataset Configuration:
   Training: XXX images
   Validation: XXX images
   TOTAL: XXX images

💡 RECOMMENDATION:
   [Recommendation based on size]
```

**Decision Point:**
- Total < 800: RUN Cell 6 (augmentation)
- Total > 1500: SKIP Cell 6, go to Cell 7 (training)

---

#### Cell 6: Augmentation (OPTIONAL)

**Skip jika total > 1500 images!**

Uncomment baris terakhir jika ingin run:
```python
augmented_count = apply_augmentation_to_dataset(merged_dataset_path, target_multiplier=2)
```

**Duration:** 10-20 menit (tergantung dataset size)

---

#### Cell 7: Load Model

```python
model = YOLO("yolov8s.pt")
```

**Duration:** 30 detik (download pretrained weights)

---

#### Cell 8: 🔥 TRAINING (MAIN EVENT!)

**Duration:** 
- GPU T4: 4-6 jam
- GPU V100: 2-3 jam

**What to expect:**
```
Epoch 1/150: 100%|██████████| 
  train: Loss: 0.05, mAP@50: 0.85
  val: Loss: 0.07, mAP@50: 0.83

Epoch 2/150: 100%|██████████|
  ...

Early stopping triggered at epoch XX (patience=30)
```

**💡 PRO TIPS:**
1. **Don't close browser!** Colab akan disconnect setelah ~90 menit idle
2. **Keep tab active** (play video/music di tab lain)
3. **Check progress tiap 30 menit**
4. **Jika disconnect**: Reconnect → Semua hasil AMAN di Google Drive

**Red Flags:**
- ❌ Loss tidak turun setelah 20 epochs → ada masalah
- ❌ mAP stuck di <50% → dataset problem
- ❌ "Out of memory" → reduce batch size di cell training

---

#### Cell 9: Validate Model

**Expected output:**
```
📊 Model Performance Metrics:
   mAP@50: 0.90-0.93 ✅ (target: >0.88)
   mAP@50-95: 0.70-0.75
   Precision: 0.88-0.92
   Recall: 0.85-0.90
```

---

#### Cell 10: 🚀 Export to ONNX

**Duration:** 1-2 menit

**Expected output:**
```
✅ Model exported to ONNX
📦 Files ready:
   1. PyTorch: best.pt
   2. ONNX: best.onnx
```

---

#### Cell 11: Speed Test

**Expected output:**
```
⏱️ PyTorch: 45ms/frame (22 FPS)
⏱️ ONNX: 28ms/frame (35 FPS)
🚀 ONNX is 1.6x faster!
```

---

### Step 4: Download Models dari Google Drive

#### Via Google Drive Web:
1. Buka Google Drive (drive.google.com)
2. Navigate: `chess_detection_improved/yolov8s_merged/weights/`
3. Download:
   - ✅ `best.pt` (model utama)
   - ✅ `best.onnx` (untuk speed)
   - 📊 `results.png` (training curves)
   - 📊 `confusion_matrix.png`

#### Via Colab (Download langsung):
```python
from google.colab import files

# Download best.pt
files.download('/content/drive/MyDrive/chess_detection_improved/yolov8s_merged/weights/best.pt')

# Download best.onnx
files.download('/content/drive/MyDrive/chess_detection_improved/yolov8s_merged/weights/best.onnx')
```

---

## 📊 Expected File Structure (Google Drive)

After training:
```
Google Drive/
└── chess_detection_improved/
    └── yolov8s_merged/
        ├── weights/
        │   ├── best.pt          # ✅ DOWNLOAD INI
        │   ├── best.onnx        # ✅ DOWNLOAD INI
        │   └── last.pt          # Backup
        ├── results.png          # Training curves
        ├── confusion_matrix.png
        ├── F1_curve.png
        ├── P_curve.png
        ├── R_curve.png
        ├── PR_curve.png
        └── model_card.json      # Metadata
```

---

## ⚠️ TROUBLESHOOTING

### Issue 1: "Runtime disconnected"
**Cause:** Colab disconnect setelah idle terlalu lama

**Solution:**
1. Reconnect ke runtime
2. Check: File masih ada di `/content/drive/MyDrive/chess_detection_improved/`
3. Jika training belum selesai:
   ```python
   # Resume training dari checkpoint
   model = YOLO('/content/drive/MyDrive/chess_detection_improved/yolov8s_merged/weights/last.pt')
   model.train(resume=True)
   ```

**Prevention:**
- Keep browser tab active
- Play video/music di background
- Use Colab Pro (no disconnect)

---

### Issue 2: "Out of Memory"
**Symptoms:**
```
CUDA out of memory. Tried to allocate XX MB
```

**Solution:**
```python
# Reduce batch size di cell training
batch=16  →  batch=8  atau  batch=4
```

---

### Issue 3: GPU Not Detected
**Check:**
```python
import torch
print(torch.cuda.is_available())  # Harus True!
```

**Solution:**
1. Runtime → Change runtime type → GPU
2. Runtime → Restart runtime
3. Re-run Cell 1

---

### Issue 4: Dataset Download Stuck
**Solution:**
- Check internet Colab (test: `!ping google.com -c 4`)
- Re-run download cell
- Check Roboflow workspace accessible

---

### Issue 5: Training Sangat Lambat
**Check:**
- ✅ GPU enabled? (`torch.cuda.is_available()` = True?)
- ✅ Batch size terlalu kecil? (min 8)
- ✅ Workers = 2-4 (jangan 0)

---

## 💡 PRO TIPS

### Tip 1: Monitor Training Progress
```python
# Install tensorboard
%load_ext tensorboard
%tensorboard --logdir /content/drive/MyDrive/chess_detection_improved/yolov8s_merged
```

### Tip 2: Save Checkpoint More Frequently
```python
# Di cell training, ubah:
save_period=10  →  save_period=5  # Save tiap 5 epochs
```

### Tip 3: Get Colab Pro (Optional)
Jika sering training:
- ✅ No disconnects
- ✅ Longer runtime (24 jam vs 12 jam)
- ✅ Better GPU (V100, A100)
- 💰 ~$10/month

---

## 📈 Expected Timeline

| Step | Duration | Cumulative |
|------|----------|------------|
| Setup & mount Drive | 2 min | 2 min |
| Install dependencies | 2 min | 4 min |
| Download datasets | 10 min | 14 min |
| Merge datasets | 2 min | 16 min |
| Augmentation (optional) | 15 min | 31 min |
| **TRAINING** 🔥 | **4-6 hours** | **~6 hours** |
| Export ONNX | 2 min | 6h 2min |
| Speed test | 1 min | 6h 3min |
| Download models | 5 min | 6h 8min |

**Total:** ~6-8 jam (mostly training time)

---

## ✅ Success Checklist

Setelah training selesai:
- [ ] mAP@50 > 88% (target: 90-93%)
- [ ] Training converged (early stopping triggered)
- [ ] ONNX export successful
- [ ] Speed test shows ONNX faster
- [ ] Models saved to Google Drive
- [ ] best.pt & best.onnx downloaded
- [ ] Training plots look good (loss turun, mAP naik)

---

## 🚀 Next Steps (Setelah Download Model)

1. Copy models ke laptop:
   ```
   best.pt → chess-detection-improve/app/model/
   best.onnx → chess-detection-improve/app/model/
   ```

2. Follow ROADMAP.md Day 2:
   - Implement ONNX inference
   - Implement frame skipping
   - Implement FEN validation
   - Implement temporal smoothing

---

## 📞 Need Help?

### Colab Resources:
- Official Docs: https://colab.research.google.com/
- FAQ: https://research.google.com/colaboratory/faq.html

### Common Questions:

**Q: Berapa lama GPU gratis bisa dipakai?**
A: ~12 jam per session, reset tiap hari.

**Q: Bisa pause training?**
A: Tidak langsung pause, tapi bisa stop → resume dari checkpoint.

**Q: Apakah data aman?**
A: Ya, selama save ke Google Drive (bukan `/content/` saja).

**Q: Bisa training multiple models sekaligus?**
A: Tidak, 1 session = 1 runtime GPU.

---

**🎉 Happy Training di Google Colab! Nikmati GPU gratis! 🚀**

---

## 📝 Quick Reference

### Important Paths (Colab):
```python
# Google Drive
/content/drive/MyDrive/

# Models output
/content/drive/MyDrive/chess_detection_improved/yolov8s_merged/

# Datasets (temporary)
/content/chess_detection-2/
/content/chess-detection-76mfe-1/
```

### Important Commands:
```python
# Check GPU
!nvidia-smi

# Check disk space
!df -h

# Monitor training (real-time)
!watch -n 5 cat /content/drive/MyDrive/chess_detection_improved/yolov8s_merged/results.txt

# Kill training (if needed)
# Runtime → Interrupt execution
```

---

**✨ Good luck with your training! 💪**
