# CARA TEST DENGAN BENAR

## Masalah Yang Ditemukan:
1. ✅ Model TIDAK ada false positive di kamera kosong
2. ❌ Kamera TERLALU GELAP (brightness 5.5 - hampir hitam!)
3. ✅ Confidence threshold diubah dari 0.05 → 0.15 (mengurangi noise)

## LANGKAH TEST YANG BENAR:

### 1. Fix Camera Brightness (WAJIB!)
Kamera Anda terlalu gelap. Brightness harus minimal 80-150.

**Cara:**
- Nyalakan lampu ruangan / lampu meja
- Arahkan kamera ke tempat terang
- Atau buka DroidCam settings dan naikkan brightness/exposure

### 2. Test Model Quality (Opsional - sudah dilakukan)
```bash
python test_model_quality.py
```
Hasil: Model OK, hanya masalah pencahayaan.

### 3. Test PyTorch Detection (Opsional)
```bash
python test_pytorch_detection.py
```
Arahkan kamera ke bidak catur dengan cahaya cukup.

### 4. Jalankan Aplikasi
```bash
cd app
python app.py
```

### 5. Test Detection:
1. Login ke web (http://127.0.0.1:5000)
2. Pilih match / create new match
3. **SEBELUM START DETECTION:**
   - ✅ Nyalakan lampu
   - ✅ Arahkan kamera ke bidak catur
   - ✅ Pastikan bidak catur terlihat jelas
4. Start detection
5. Lihat hasilnya

## SETTINGS TERBARU:
- Confidence: 0.15 (lebih ketat, kurangi false positive)
- Camera backend: DirectShow + CAP_ANY fallback
- Thread prevention: 1.5s delay
- Cache cleared: ✅

## EKSPEKTASI:
- **Jika TIDAK ada bidak:** 0 detections ✅
- **Jika ADA bidak + cahaya cukup:** Akan terdeteksi ✅
- **Jika ADA bidak tapi gelap:** Mungkin 0-2 detections (naikkan cahaya!)

## TROUBLESHOOTING:
❓ "Masih 0 detections meski ada bidak?"
→ Naikkan cahaya! Target brightness: 80-150

❓ "Camera terlalu lama loading?"
→ Sudah diperbaiki (3 camera max + DirectShow)

❓ "Multiple threads error?"
→ Sudah diperbaiki (thread prevention)

❓ "Model masih detect padahal kosong?"
→ Tidak akan terjadi lagi (conf 0.15 + test confirmed)
