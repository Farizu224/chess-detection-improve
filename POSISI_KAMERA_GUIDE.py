"""
CARA POSISI KAMERA YANG BENAR

Berdasarkan hasil diagnostic:
- Model mendeteksi di area: x=217-324, y=518-628
- Ini adalah BAGIAN BAWAH-TENGAH frame (736x736)

POSISI YANG SALAH (sekarang):
┌─────────────────┐
│                 │
│                 │  ← Bidak catur di sini (TIDAK TERDETEKSI)
│                 │
│                 │
│      [🔲]       │  ← Detection area di sini (kosong)
└─────────────────┘

POSISI YANG BENAR:
┌─────────────────┐
│                 │
│                 │
│                 │
│                 │
│      [♟]       │  ← Taruh bidak catur di SINI
└─────────────────┘  ← Detection area = bagian bawah frame!

SOLUSI:
1. Arahkan kamera sehingga bidak catur ada di BAGIAN BAWAH frame
2. Atau letakkan bidak catur lebih dekat ke kamera
3. Atau posisikan kamera lebih tinggi (top-down view)

COORDINATE REFERENCE:
- Detection area: x=217-324 (center ~270), y=518-628 (bottom area)
- Frame size: 736x736
- Y=0 adalah TOP, Y=736 adalah BOTTOM
- Model expects pieces in BOTTOM HALF of frame!

BRIGHTNESS:
- Current: 8-64 (TOO DARK!)
- Target: 80-150
- Action: Nyalakan lampu atau naikkan camera exposure
"""
print(__doc__)
