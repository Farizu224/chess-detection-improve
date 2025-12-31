# 🚀 CARA MENJALANKAN APLIKASI

## Opsi 1: Menggunakan Batch File (TERMUDAH) ✅

Cukup double-click file ini:
```
START_APP.bat
```

File ini akan otomatis:
- ✅ Check dependencies
- ✅ Install yang missing (jika ada)
- ✅ Start Flask server
- ✅ Open http://localhost:5000

---

## Opsi 2: Manual via Terminal

### Windows PowerShell:
```powershell
cd "chess-detection-improve"
cd app
python app.py
```

### Command Prompt:
```cmd
cd chess-detection-improve
cd app
python app.py
```

---

## 🌐 Akses Aplikasi

Setelah server jalan, buka browser:
```
http://localhost:5000
```

### Login Default:
- **Username:** `admin`
- **Password:** `admin123`

---

## 🔧 Jika Ada Error

### Error: ModuleNotFoundError

Install dependencies yang missing:
```bash
pip install flask flask-socketio flask-login flask-bcrypt ultralytics opencv-python scikit-learn python-chess filterpy albumentations
```

### Error: pygame/onnxruntime not found

**ABAIKAN!** Sudah dihandle otomatis. Aplikasi tetap jalan tanpa pygame/onnxruntime.
- ❌ Pygame: Untuk UI visualization (optional)
- ❌ ONNX: Untuk faster inference (optional, pakai PyTorch mode)

### Error: Port already in use

Port 5000 sudah dipakai. Ubah port di app.py:
```python
# app.py line 882
socketio.run(app, debug=True, port=5001)  # Ganti ke 5001
```

---

## 📱 Fitur Aplikasi

1. **Dashboard Admin** - Monitor semua match
2. **Dashboard Player** - View personal games
3. **Live Detection** - Real-time chess piece detection
4. **Match Recording** - Record moves & analysis
5. **Board Detection** - Auto-detect chessboard

---

## 🎮 Cara Menggunakan

### 1. Login
- Buka http://localhost:5000
- Login dengan admin/admin123

### 2. Start Match
- Create new match
- Select players
- Set time control

### 3. Start Detection
- Click "Start Camera"
- Choose camera index (biasanya 0)
- Detection mode: raw/annotated
- Point camera ke papan catur

### 4. Play & Monitor
- Move pieces on board
- System detects automatically
- View real-time analysis
- Timer runs automatically

---

## 🛑 Stop Server

Press `Ctrl + C` di terminal untuk stop server.

---

## 📊 Status Dependencies

✅ **Installed:**
- opencv-python
- ultralytics  
- scikit-learn
- python-chess
- filterpy
- albumentations
- flask & extensions

❌ **Optional (Not Required):**
- pygame (UI visualization)
- onnxruntime (faster inference)

**App berjalan normal tanpa 2 dependency optional di atas!**

---

## 💡 Tips

1. **Camera tidak terdeteksi?**
   - Coba ganti camera_index (0, 1, 2, dst)
   - Check camera permissions
   - Test dengan `python find_cameras.py`

2. **Detection lambat?**
   - Model otomatis pakai PyTorch mode (karena ONNX tidak available)
   - Normal untuk inference time ~100-200ms per frame

3. **Board tidak terdeteksi?**
   - Pastikan pencahayaan bagus
   - Board dalam frame lengkap
   - Coba adjust camera angle

---

## 🐛 Debug Mode

Untuk debug detailed:
```python
# app.py line 882
socketio.run(app, debug=True, log_level='DEBUG')
```

---

Selamat mencoba! 🎉
