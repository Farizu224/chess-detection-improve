# 🔗 DATASET MERGING GUIDE

## 📊 Overview
Proyek ini menggabungkan **2 dataset** dari Roboflow untuk mendapatkan data training yang lebih banyak dan diverse:

### Dataset Sources
1. **Dataset Original** (kosan-hendra)
   - Workspace: `kosan-hendra`
   - Project: `chess_detection-uzejh`
   - Version: 2

2. **Dataset Anda** (detection-chess)
   - Workspace: `detection-chess`
   - Project: `chess-detection-76mfe`
   - Version: 1 (sesuaikan dengan version di Roboflow Anda)

---

## 🎯 Mengapa Merge Dataset?

### Keuntungan:
✅ **Lebih Banyak Data** = Model lebih generalized  
✅ **Lebih Diverse** = Robust terhadap variasi kondisi  
✅ **Better Performance** = Accuracy meningkat (expected +3-5%)  
✅ **Avoid Overfitting** = Model tidak terlalu fit ke satu sumber data  

### Trade-offs:
⚠️ Training time lebih lama (proportional dengan jumlah data)  
⚠️ Perlu verifikasi class compatibility  

---

## 📝 Step-by-Step: Cara Merge

### Step 1: Dapatkan API Key Anda
1. Login ke https://app.roboflow.com/
2. Klik profil (kanan atas) → Settings
3. Copy API key Anda

### Step 2: Update Notebook
Buka `chess_detection_improve.ipynb` dan:

**Cell 2 (Download Datasets):**
```python
# Ganti YOUR_API_KEY_HERE dengan API key Anda
rf2 = Roboflow(api_key="YOUR_ACTUAL_API_KEY")
```

**Sesuaikan version number:**
```python
version2 = project2.version(1)  # Ubah sesuai version dataset Anda
```

### Step 3: Run Cells Berurutan
1. **Cell 1**: Install dependencies
2. **Cell 2**: Download kedua dataset (butuh waktu 5-10 menit)
3. **Cell 3**: Merge datasets (automatic prefix untuk avoid conflicts)
4. **Cell 4**: Create data.yaml (verify class compatibility)
5. **Cell 5**: (Optional) Augmentation jika total <800 images
6. **Cell 6+**: Training dengan merged dataset

---

## 🔍 Cara Kerja Merging

### Prefix System
File dari kedua dataset diberi prefix untuk menghindari duplikasi:
```
Dataset 1: image1.jpg → ds1_image1.jpg
Dataset 2: image1.jpg → ds2_image1.jpg
```

### Class Compatibility Check
Notebook akan otomatis:
- ✅ Verify kedua dataset punya classes yang sama
- ⚠️ Warn jika ada perbedaan
- ✅ Gunakan Dataset 1 classes sebagai reference

### Expected Structure
```
chess_detection_merged/
├── train/
│   ├── images/
│   │   ├── ds1_img001.jpg
│   │   ├── ds1_img002.jpg
│   │   ├── ds2_img001.jpg
│   │   └── ds2_img002.jpg
│   └── labels/
│       ├── ds1_img001.txt
│       ├── ds1_img002.txt
│       ├── ds2_img001.txt
│       └── ds2_img002.txt
├── valid/
│   ├── images/
│   └── labels/
└── data.yaml
```

---

## 📊 Expected Results

### Dataset Size Scenarios

#### Scenario 1: Both Small (<500 each)
- **Total**: ~800-1000 images
- **Action**: RUN augmentation (Cell 5)
- **Expected mAP**: 88-90%

#### Scenario 2: One Large (>800)
- **Total**: ~1200-1500 images
- **Action**: OPTIONAL augmentation
- **Expected mAP**: 90-92%

#### Scenario 3: Both Large (>800 each)
- **Total**: >1600 images
- **Action**: SKIP augmentation
- **Expected mAP**: 92-94% ✨

---

## ⚠️ Common Issues & Solutions

### Issue 1: API Key Error
```
Error: Invalid API key
```
**Solution:** 
- Verify API key dari Roboflow settings
- Pastikan tidak ada spasi atau typo
- Try re-generate API key

### Issue 2: Version Not Found
```
Error: Version 1 not found
```
**Solution:**
- Check di Roboflow, version berapa yang available
- Update: `version2 = project2.version(2)` (atau version yang benar)

### Issue 3: Class Mismatch Warning
```
⚠️ WARNING: Classes don't match exactly!
```
**Solution:**
- Review kedua dataset di Roboflow
- Pastikan semua pieces punya label yang sama
- Jika beda sedikit (e.g., "king" vs "King"), rename di Roboflow

### Issue 4: Out of Disk Space
```
Error: No space left on device
```
**Solution:**
- Clear unused files
- Dataset merged ~2-5GB tergantung size
- Ensure minimal 10GB free space

---

## 🎯 Verification Checklist

Setelah merge, verify:
- [ ] Kedua dataset ter-download
- [ ] Merged folder created
- [ ] Image count = Dataset1 + Dataset2
- [ ] Labels count = Images count
- [ ] data.yaml contains correct paths
- [ ] Class names match antara kedua dataset
- [ ] No file conflicts (prefix system working)

---

## 📈 Performance Impact

### Expected Improvements from Merging:

| Metric | Single Dataset | Merged Dataset | Gain |
|--------|----------------|----------------|------|
| Training Data | 400-600 | 800-1200 | +100% |
| Model Robustness | Medium | High | ✅ |
| mAP@50 | 88-90% | 90-92% | +2-3% |
| Overfitting Risk | Higher | Lower | ✅ |
| Generalization | Limited | Better | ✅ |

---

## 💡 Pro Tips

### 1. Verify Before Training
Jalankan quick check:
```python
# Verify merged dataset
train_imgs = os.listdir(f"{merged_dataset_path}/train/images")
train_lbls = os.listdir(f"{merged_dataset_path}/train/labels")

print(f"Images: {len(train_imgs)}")
print(f"Labels: {len(train_lbls)}")
print(f"Match: {len(train_imgs) == len(train_lbls)}")  # Should be True
```

### 2. Backup Original Datasets
Sebelum training, backup:
```bash
# Jangan hapus dataset1.location dan dataset2.location
# Sampai training selesai dan verified
```

### 3. Monitor Class Distribution
Check apakah semua classes terwakili cukup:
```python
# Count labels per class
# (code provided in notebook)
```

---

## 🚀 Next Steps After Merge

1. ✅ Verify merged dataset (checklist di atas)
2. ✅ Decide: augmentation or not?
3. ✅ Start training (Cell 6)
4. ✅ Monitor training metrics
5. ✅ Export to ONNX (Cell 7)

---

## 📞 Need Help?

### If Merge Fails:
1. Check API keys are correct
2. Verify internet connection
3. Try downloading datasets separately first
4. Check Roboflow project permissions

### If Training on Merged Dataset Fails:
1. Verify data.yaml path is correct
2. Check all images have corresponding labels
3. Ensure no corrupted files
4. Try smaller batch size

---

**✨ Happy Merging! Dengan 2 dataset, model Anda akan lebih powerful! ♟️**
