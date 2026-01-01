# 📱 CARA PAKAI DROIDCAM

## 🎯 Kenapa Pakai DroidCam?
- ✅ Kualitas camera HP jauh lebih bagus dari webcam laptop
- ✅ Resolusi lebih tinggi (720p, 1080p, bahkan 4K)
- ✅ Gratis dan mudah disetup
- ✅ Bisa via WiFi atau USB

---

## 📥 SETUP DROIDCAM

### Option 1: DroidCam via Virtual Camera (RECOMMENDED)

#### Step 1: Install DroidCam
1. **Di HP (Android/iOS):**
   - Download **DroidCam** dari Play Store / App Store
   - Buka app, allow camera permission

2. **Di Laptop (Windows):**
   - Download **DroidCam Client** dari: https://www.dev47apps.com/
   - Install client di laptop
   - Ini akan membuat "virtual camera" yang bisa dipakai

#### Step 2: Connect
1. Pastikan HP & Laptop di **WiFi yang sama**
2. Buka DroidCam di HP, lihat **IP address**
3. Buka DroidCam Client di laptop
4. Masukkan IP address dari HP
5. Klik **Start**
6. DroidCam sekarang muncul sebagai virtual camera!

#### Step 3: Cari Camera Index
```bash
cd d:\chess-detection-improve\chess-detection-improve
python droidcam_config.py scan
```

Output contoh:
```
✅ Camera 0: 640x480     <- Webcam laptop
✅ Camera 1: 1920x1080   <- DroidCam! (resolusi tinggi)
```

#### Step 4: Update Config
Edit file `droidcam_config.py`:
```python
USE_DROIDCAM_VIRTUAL = True
USE_DROIDCAM_URL = False
DROIDCAM_VIRTUAL_CAMERA_INDEX = 1  # Ganti sesuai hasil scan
```

---

### Option 2: DroidCam via WiFi URL (Tanpa Client)

#### Step 1: Get IP Address
1. Buka DroidCam app di HP
2. Lihat **WiFi IP** (contoh: 192.168.1.100:4747)

#### Step 2: Update Config
Edit file `droidcam_config.py`:
```python
USE_DROIDCAM_URL = True
USE_DROIDCAM_VIRTUAL = False
DROIDCAM_URL = "http://192.168.1.100:4747/video"  # Ganti IP Anda
```

#### Step 3: Test Connection
```bash
python droidcam_config.py test
```

---

## 🚀 CARA PAKAI DI APLIKASI

### Method 1: Via Config File (RECOMMENDED)

Update `app/config.py`:
```python
from droidcam_config import get_camera_source

CAMERA_SOURCE = get_camera_source()  # Auto detect dari droidcam_config
```

### Method 2: Langsung Set di Code

Edit `app/app.py`, cari baris:
```python
detection_service = ChessDetectionService()
```

Ganti dengan:
```python
detection_service = ChessDetectionService()

# Option A: DroidCam Virtual Camera
detection_service.camera_source = 1  # Index dari scan

# Option B: DroidCam URL
detection_service.camera_source = "http://192.168.1.100:4747/video"
```

### Method 3: Via Environment Variable

Buat file `.env`:
```bash
CAMERA_SOURCE=1
# atau
CAMERA_SOURCE=http://192.168.1.100:4747/video
```

---

## 🎯 ADJUST DETECTION THRESHOLD

Karena DroidCam lebih bagus, threshold perlu disesuaikan:

Edit `app/config.py`:
```python
# Untuk DroidCam (kamera bagus):
DETECTION_CONFIDENCE = 0.4  # Turun dari 0.5 → lebih sensitive
NMS_IOU_THRESHOLD = 0.4

# Untuk webcam laptop (kamera jelek):
DETECTION_CONFIDENCE = 0.6  # Naik dari 0.5 → lebih strict
NMS_IOU_THRESHOLD = 0.5
```

Atau edit di `chess_detection.py`, cari method yang ada `conf=` dan `iou=`:
```python
# Contoh di method inference
results = self.model.predict(
    image,
    conf=0.4,  # Turun untuk kamera bagus
    iou=0.4,   # Turun untuk kamera bagus
    ...
)
```

---

## 🧪 TESTING

### Test 1: Scan Cameras
```bash
python droidcam_config.py scan
```

### Test 2: Test DroidCam Connection
```bash
python droidcam_config.py test
```

### Test 3: Setup Wizard (Interactive)
```bash
python droidcam_config.py setup
```

### Test 4: Run Quick Camera Test
```bash
python quick_usb_test.py
# Saat diminta camera index, masukkan index DroidCam (dari scan)
```

---

## 🔧 TROUBLESHOOTING

### Problem: DroidCam tidak muncul di scan
**Solution:**
- Pastikan DroidCam Client installed & running
- Restart DroidCam Client
- Restart laptop
- Check di Device Manager → Cameras → lihat apakah ada "DroidCam"

### Problem: URL tidak bisa connect
**Solution:**
- HP & Laptop harus di WiFi yang sama (TIDAK bisa beda WiFi)
- Check IP address benar (lihat di DroidCam app)
- Ping IP dari laptop: `ping 192.168.1.100`
- Disable firewall sementara
- Coba port lain: 4747, 4748

### Problem: Frame lag/choppy
**Solution:**
- Gunakan WiFi 5GHz (lebih cepat dari 2.4GHz)
- Pindah HP lebih dekat ke router
- Turunkan resolusi di DroidCam app
- Gunakan USB mode jika ada kabel

### Problem: Detection kurang akurat
**Solution:**
- Adjust threshold di config.py
- Pastikan pencahayaan bagus
- Posisi HP tegak lurus ke chessboard
- Jarak ideal: 30-50cm dari board

---

## 💡 TIPS UNTUK HASIL TERBAIK

### 1. Pencahayaan
- Gunakan lampu putih (bukan kuning)
- Cahaya dari samping, bukan dari atas (avoid shadow)
- Avoid backlight (jangan ada cahaya terang di belakang board)

### 2. Posisi Camera
- Tegak lurus ke chessboard (90 derajat)
- Jarak 30-50cm
- Frame hanya board, jangan terlalu banyak background

### 3. Stabilitas
- Gunakan tripod atau holder HP
- Jangan pegang HP dengan tangan (goyang)
- Bisa pakai stack buku untuk naikkan HP

### 4. Settings
- Resolusi: 720p cukup (1080p bisa lebih lambat)
- FPS: 30fps optimal
- Focus: Auto focus ON
- Exposure: Auto

---

## 📊 EXPECTED IMPROVEMENTS

### Before (Webcam Laptop):
- Resolution: 640x480
- FPS: 15-20
- Detection accuracy: 70-80%
- Pencahayaan: Kurang bagus

### After (DroidCam HP):
- Resolution: 1280x720 atau 1920x1080
- FPS: 30
- Detection accuracy: 85-95%
- Pencahayaan: Lebih bagus (camera HP modern)

---

## ✅ CHECKLIST SETUP

Sebelum run app dengan DroidCam:

- [ ] DroidCam app installed di HP
- [ ] DroidCam Client installed di laptop (jika pakai virtual camera)
- [ ] HP & laptop di WiFi yang sama
- [ ] Scan cameras: `python droidcam_config.py scan`
- [ ] Update config: `droidcam_config.py` dengan index/URL yang benar
- [ ] Test connection: `python droidcam_config.py test`
- [ ] Adjust threshold di `config.py` (turunkan ke 0.4)
- [ ] Run app: `python app\app.py`

---

## 🎮 QUICK START

Cara paling cepat:

```bash
# 1. Setup DroidCam (interactive)
python droidcam_config.py setup

# 2. Run app dengan DroidCam
python app\app.py

# 3. Di browser, saat create match:
#    - Pilih camera index sesuai hasil scan
#    - Start detection
```

---

Selamat mencoba! Camera HP Anda jauh lebih bagus dari webcam laptop! 🚀📱
