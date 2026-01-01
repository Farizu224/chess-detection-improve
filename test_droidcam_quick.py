"""
Quick DroidCam Test - Test apakah DroidCam bisa dideteksi dan digunakan
Run this BEFORE starting the web application
"""
import sys
sys.path.insert(0, 'app')

from camera_helper import run_droidcam_assistant

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                  DROIDCAM QUICK TEST & SETUP                         ║
╚══════════════════════════════════════════════════════════════════════╝

Gunakan tool ini untuk:
✓ Scan semua cameras yang available
✓ Auto-detect camera mana yang DroidCam
✓ Test apakah DroidCam berfungsi dengan baik
✓ Dapatkan index yang benar untuk config

PREREQUISITE:
• DroidCam Client running di laptop
• DroidCam app running di HP
• Connect via USB atau WiFi (sama network)
    """)
    
    run_droidcam_assistant()
