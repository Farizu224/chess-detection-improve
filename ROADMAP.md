# 🗓️ ROADMAP PENGERJAAN - 3 HARI

## 📌 Overview
Roadmap realistis untuk menyelesaikan improvement dalam 3 hari dengan fokus pada **Low Effort, High Impact**.

---

## 📅 DAY 1: MODEL TRAINING & PREPARATION (8-10 jam)

### 🎯 Target Hari 1
✅ Model baru dengan data augmentation  
✅ Export ke ONNX untuk speed boost  
✅ Validasi performa model  

### 📋 Task Breakdown

#### Morning (4 jam)
**1. Setup Environment & Dataset** (1 jam)
- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Download dataset dari Roboflow (chess_detection_improve.ipynb)
- [ ] Verify dataset structure (train/val/test)

**2. Data Augmentation Setup** (1 jam)
- [ ] Review augmentation pipeline di notebook
- [ ] *OPTIONAL*: Run augmentation script (jika dataset < 1000 images)
- [ ] Jika dataset sudah besar (>1000), skip augmentation tambahan

**3. Model Training - Start** (2 jam)
- [ ] Run training cell di `chess_detection_improve.ipynb`
- [ ] Monitor GPU usage & memory
- [ ] Training akan berjalan 6-8 jam (biarkan overnight jika perlu)

#### Afternoon (4 jam) - *Sambil training berjalan*
**4. Prepare Project Structure** (2 jam)
- [ ] Copy file-file dari `chess-detection/` ke `chess-detection-improve/`
- [ ] Update `requirements.txt` dengan library baru:
  ```
  onnxruntime-gpu  # untuk ONNX inference
  filterpy         # untuk Kalman filter (temporal smoothing)
  ```
- [ ] Setup folder structure (sudah dibuat)

**5. Study Original Code** (2 jam)
- [ ] Baca dan pahami `chess_detection.py` (fokus method `detect()`)
- [ ] Baca dan pahami FEN generation logic
- [ ] Identifikasi bottleneck (preprocessing, inference, postprocessing)

#### Evening - Check Training
**6. Monitor Training Progress**
- [ ] Check training curves (loss, mAP)
- [ ] Jika ada masalah, adjust hyperparameters
- [ ] Biarkan training continue overnight

---

## 📅 DAY 2: CORE IMPROVEMENTS (8-10 jam)

### 🎯 Target Hari 2
✅ ONNX model siap pakai  
✅ Frame skipping implemented  
✅ FEN validation logic complete  
✅ Temporal smoothing working  

### 📋 Task Breakdown

#### Morning (4 jam)
**1. Finalize Model Training** (1 jam)
- [ ] Check training completed (150 epochs or early stopped)
- [ ] Validate model metrics (mAP@50 should be >90%)
- [ ] Export to ONNX (run export cell)
- [ ] Test ONNX inference speed

**2. Copy Model Files** (15 menit)
- [ ] Copy `best.pt` ke `chess-detection-improve/app/model/`
- [ ] Copy `best.onnx` ke `chess-detection-improve/app/model/`
- [ ] Backup original model

**3. Implement ONNX Inference** (2 jam)
- [ ] Modify `chess_detection.py`:
  - Add ONNX Runtime support
  - Create `load_onnx_model()` method
  - Replace PyTorch inference dengan ONNX
- [ ] Test inference speed (should be 30-50% faster)
- [ ] Fallback ke PyTorch jika ONNX gagal

**4. Implement Frame Skipping** (1 jam)
- [ ] Create `frame_tracker.py`:
  ```python
  class FrameTracker:
      def __init__(self, skip_frames=4):
          self.skip_frames = skip_frames
          self.frame_count = 0
          self.last_detections = None
      
      def should_detect(self):
          self.frame_count += 1
          return (self.frame_count % self.skip_frames) == 0
  ```
- [ ] Integrate ke `chess_detection.py`
- [ ] Test FPS improvement

#### Afternoon (4 jam)
**5. Implement FEN Validator** (2 jam)
- [ ] Create `fen_validator.py`:
  ```python
  class FENValidator:
      def validate_piece_count(self, fen):
          # Max 16 pieces per color
      
      def validate_king_count(self, fen):
          # Exactly 1 king per color
      
      def validate_pawn_position(self, fen):
          # Pawns tidak di rank 1/8
      
      def is_valid(self, fen):
          # Combined validation
  ```
- [ ] Test dengan valid & invalid FEN

**6. Implement Temporal Smoothing** (2 jam)
- [ ] Add FEN buffer ke `chess_detection.py`:
  ```python
  from collections import deque
  
  class ChessDetectionService:
      def __init__(self):
          self.fen_buffer = deque(maxlen=5)
      
      def smooth_fen(self, current_fen):
          self.fen_buffer.append(current_fen)
          # Return most common FEN
          return most_common(self.fen_buffer)
  ```
- [ ] Implement voting mechanism
- [ ] Test stability

#### Evening (2 jam)
**7. Integration Testing**
- [ ] Test full pipeline: Camera → Detection → FEN → Validation → Smoothing
- [ ] Check for bugs & edge cases
- [ ] Fix critical issues

---

## 📅 DAY 3: POLISHING & DOCUMENTATION (6-8 jam)

### 🎯 Target Hari 3
✅ All features working  
✅ Benchmarks complete  
✅ Documentation ready  
✅ Demo prepared  

### 📋 Task Breakdown

#### Morning (3 jam)
**1. Bug Fixes & Refinement** (2 jam)
- [ ] Fix any remaining bugs from Day 2
- [ ] Optimize code (remove debug prints, etc.)
- [ ] Add error handling & logging

**2. Performance Benchmarking** (1 jam)
- [ ] Create `tests/benchmark_speed.py`:
  - Compare original vs improved FPS
  - Measure inference time (PyTorch vs ONNX)
  - Test FEN validity rate
- [ ] Run benchmarks & document results
- [ ] Generate comparison table

#### Afternoon (3 jam)
**3. Documentation** (2 jam)
- [ ] Update `README.md` dengan hasil benchmark
- [ ] Create `IMPROVEMENTS.md` dengan detail teknis:
  - What was changed?
  - Why?
  - How much improvement?
- [ ] Add code comments di file-file penting
- [ ] Screenshot results untuk demo

**4. Testing & Validation** (1 jam)
- [ ] Test dengan berbagai kondisi:
  - Lighting berbeda (terang, gelap)
  - Kamera angle berbeda
  - Mid-game positions
- [ ] Verify FEN validation bekerja
- [ ] Check temporal smoothing reduces jitter

#### Evening (2 jam)
**5. Demo Preparation**
- [ ] Prepare demo script:
  1. Show original system (slow, jittery FEN)
  2. Show improved system (fast, stable FEN)
  3. Show benchmark comparison
- [ ] Record video demo (optional)
- [ ] Prepare presentation slides (jika perlu)

**6. Final Check**
- [ ] Semua kode di-commit ke git
- [ ] README complete
- [ ] Model files tersedia
- [ ] Application berjalan tanpa error

---

## 📊 Success Criteria

### Minimum Viable Product (MUST HAVE)
- [x] Model trained dengan augmentation
- [ ] ONNX export working
- [ ] Frame skipping implemented
- [ ] FEN validation working
- [ ] Temporal smoothing implemented
- [ ] FPS meningkat minimal 2x
- [ ] FEN validity >90%

### Nice to Have (OPTIONAL)
- [ ] Kalman filter untuk smoother corner detection
- [ ] Confidence threshold tuning
- [ ] Per-piece accuracy analysis
- [ ] Video demo

---

## ⚠️ Risk Mitigation

### Potential Issues & Solutions

**Issue 1: Training terlalu lama**
- **Solution**: Gunakan pretrained `yolov8s.pt`, reduce epochs ke 100
- **Backup**: Gunakan model original jika training gagal

**Issue 2: ONNX export error**
- **Solution**: Skip ONNX, fokus ke frame skipping saja
- **Impact**: Masih dapat 2x speedup dari frame skipping

**Issue 3: FEN validation terlalu strict**
- **Solution**: Relax validation rules, fokus ke basic checks saja
- **Backup**: Skip auto-correction, hanya validation

**Issue 4: Temporal smoothing delay terlalu besar**
- **Solution**: Reduce buffer size dari 5 ke 3 frames
- **Alternative**: Gunakan weighted voting (recent frames lebih penting)

---

## 📝 Daily Checklist

### Setiap Akhir Hari:
- [ ] Commit code ke git
- [ ] Backup model files
- [ ] Update progress di ROADMAP.md
- [ ] List blockers untuk besok

### Setiap Pagi:
- [ ] Review kemarin punya progress
- [ ] Check training status (if running)
- [ ] Plan today's priorities

---

## 🎯 Expected Outcome

Setelah 3 hari, Anda akan punya:
1. ✅ Model yang lebih akurat (+5-10% mAP)
2. ✅ Inference 2-3x lebih cepat
3. ✅ FEN validation yang robust (95%+ validity)
4. ✅ System yang lebih stable (temporal smoothing)
5. ✅ Dokumentasi lengkap untuk demo

---

**🚀 Good luck! You got this! ♟️**

---

## 📞 Notes

Jika ada yang stuck atau butuh bantuan, prioritaskan:
1. **Core features** (ONNX, frame skipping, FEN validation) DULU
2. **Polishing** (UI, documentation) KEMUDIAN
3. **Nice-to-have** (Kalman filter, advanced features) SKIP jika waktu mepet

**Remember**: Done is better than perfect! 💪
