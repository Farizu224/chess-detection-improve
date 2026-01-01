# 📷 CAMERA SETUP GUIDE - Top-Down View

## ⚠️ CRITICAL: Camera Position Matters!

Model trained dengan **top-down view** (dari atas tegak lurus).
Webcam setup sekarang **angled view** (dari samping) → Model tidak bisa detect!

---

## ✅ CORRECT SETUP (Match Training Data):

```
        📱 Camera (looking straight down)
        |
        |
        ↓ 90° perpendicular
    ┌─────────────┐
    │ Chess Board │  ← Top-down view
    └─────────────┘
```

### Requirements:
1. **Camera angle**: 90° perpendicular (directly above board)
2. **Height**: ~30-50 cm above board (adjust untuk full board visible)
3. **Lighting**: Even lighting, no shadows
4. **Background**: Minimal background, focus on board

---

## ❌ CURRENT SETUP (Tidak Cocok):

```
                    📱 Camera
                   /
                  /  ~30-45° angle
                 ↙
    ┌─────────────┐
    │ Chess Board │  ← Angled view (model tidak trained untuk ini!)
    └─────────────┘
```

---

## 🛠️ How to Fix:

### Method 1: Overhead Mount
- Mount phone/webcam di atas board menggunakan tripod/stand
- Arahkan camera langsung ke bawah (90°)
- Test dengan test_detection_standalone.py

### Method 2: Laptop + Book Stack
- Letakkan laptop di atas tumpukan buku
- Buka laptop hingga webcam menghadap ke bawah
- Adjust tinggi dengan menambah/kurangi buku

### Method 3: Phone Holder
- Gunakan flexible phone holder
- Posisikan phone di atas board
- Use phone as webcam (via DroidCam/IP Webcam)

---

## 🧪 Test After Setup:

```bash
cd d:\chess-detection-improve\chess-detection-improve
python test_detection_standalone.py
```

**Expected result:**
- FPS: ~2-3 (sama)
- Detections: 5-8 per frame (bukan 20+!)
- White pieces detected: ~50% of detections
- Confidence: >0.5 average

---

## 🔄 If Top-Down Setup Not Possible:

Need to **retrain model** with angled view data:

1. Collect 50-100 images from YOUR current webcam setup:
   ```bash
   python collect_webcam_data.py
   ```

2. Annotate images using Roboflow

3. Merge with existing dataset

4. Retrain model (in Colab)

---

## 📊 Why This Matters:

**Training data = Top-down view**
→ Model learned features: circular pieces from above, full board visibility

**Your webcam = Angled view**
→ Pieces look elongated, occluded, different aspect ratio
→ Model: "I don't recognize these shapes!"

**This is called "Domain Shift" problem in ML**

---

## ✅ Quick Check:

Test if your current setup matches training data:
- [ ] Camera looking straight down?
- [ ] Full board visible (no edges cut off)?
- [ ] Pieces clearly visible from above?
- [ ] Minimal background/distraction?

If NOT all checked → Repositioning required!
