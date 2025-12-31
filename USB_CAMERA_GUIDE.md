# 🎯 CARA MENGGUNAKAN USB CAMERA

## Status Saat Ini

✅ **USB Camera Terdeteksi:** Camera Index 1
✅ **Laptop Camera:** Camera Index 0  
❌ **Masalah:** Window tidak muncul setelah camera dibuka

---

## 🚀 SOLUSI 1: Gunakan USB Camera (INDEX 1)

### Di Web Interface:

1. **Buka match** di browser
2. **Scroll ke "Detection Settings"**
3. **Pilih USB camera** dari dropdown "Select Camera"
   - Dropdown ini list semua camera yang terdeteksi
   - Pilih yang BUKAN "Built-in Camera" (biasanya index kedua)
4. **Klik "Start Camera"**

### Atau Edit Default Camera:

Edit file: `app/templates/match_detail_admin.html`

Cari baris ini (sekitar line 704):
```javascript
const cameraIndex = Array.from(cameraSelect.options).findIndex(option => option.value === deviceId) - 1;
```

Ganti dengan (untuk force USB camera):
```javascript
const cameraIndex = 1; // Force USB camera (index 1)
```

---

## 🐛 SOLUSI 2: Debug Window Issue

### Masalah:

Camera berhasil dibuka tapi window tidak muncul:
```
✅ Successfully opened camera 0 with DirectShow
(no output after this)
```

### Kemungkinan Penyebab:

1. **Thread crash** sebelum window creation
2. **Exception tidak tertangkap**
3. **OpenCV GUI issue**

### Testing:

Jalankan test sederhana:
```bash
cd chess-detection-improve
test_camera.bat
```

Atau manual:
```powershell
cd chess-detection-improve
python -c "import cv2; cap = cv2.VideoCapture(1); print('Camera 1:', cap.isOpened()); ret, frame = cap.read(); print('Read OK:', ret); cap.release(); import numpy as np; cv2.imshow('Test', np.zeros((480,640,3))); cv2.waitKey(2000); cv2.destroyAllWindows()"
```

Jika test berhasil tapi app gagal → masalah di detection loop.

---

## 🔧 QUICK FIX: Test Detection Loop

Saya sudah menambahkan logging lebih detail. Run app lagi:

```bash
cd app
python app.py
```

Sekarang Anda akan lihat:
```
▶️ DETECTION LOOP STARTED
   Thread ID: ...
   Camera Index: 1
   Mode: raw

🎥 Opening camera index: 1
   ...

📹 Camera Configuration:
   Camera Index: 1
   Resolution: 640x480
   ...

🖼️ Creating OpenCV window...
   ✅ Window created successfully!

▶️ Starting detection loop...
   Loop iteration 1...
   Loop iteration 2...
   ...
```

**Jika tidak muncul output setelah "Successfully opened camera":**
→ Ada exception yang crash thread sebelum logging. Check console untuk error.

---

## 📝 NEXT STEPS

1. **Run app** dengan logging baru
2. **Copy paste SEMUA output console** dari saat click "Start Camera" sampai selesai
3. **Check apakah ada error** setelah "Successfully opened camera"
4. **Berikan output ke saya** untuk debug lebih lanjut

---

## 💡 Tips

### Untuk Pakai USB Camera Setiap Kali:

Edit `app/app.py` line 863 area (di `on_start_opencv_detection`):

```python
camera_index = int(data.get('camera_index', 1))  # Default ke 1 (USB)
```

### Test USB Camera Langsung:

```bash
cd chess-detection-improve
python -c "import cv2, numpy as np; cap = cv2.VideoCapture(1); cv2.namedWindow('USB Camera'); while True: ret, f = cap.read(); cv2.imshow('USB Camera', f if ret else np.zeros((480,640,3), dtype=np.uint8)); if cv2.waitKey(1) & 0xFF == ord('q'): break; cap.release(); cv2.destroyAllWindows()"
```

Press Q to quit.

---

**PENTING:** Setelah run app lagi, tunjukkan **FULL console output** terutama bagian setelah "Successfully opened camera". Dengan logging baru, saya bisa tau persis di mana crash terjadi!
