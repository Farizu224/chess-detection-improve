"""
Quick test untuk verify config sudah benar
"""
import sys
sys.path.insert(0, 'app')

# Load config
try:
    from droidcam_config import DROIDCAM_CAMERA_INDEX, LAPTOP_WEBCAM_INDEX, SKIP_LAPTOP_WEBCAM, HIDE_CAMERA_2
except:
    HIDE_CAMERA_2 = False

print("=" * 70)
print("📋 CURRENT CONFIG")
print("=" * 70)
print()
print(f"DroidCam Camera Index:  {DROIDCAM_CAMERA_INDEX}")
print(f"Laptop Webcam Index:    {LAPTOP_WEBCAM_INDEX}")
print(f"Skip Laptop Webcam:     {SKIP_LAPTOP_WEBCAM}")
print(f"Hide Camera 2 (Low Res): {HIDE_CAMERA_2}")
print()
print("=" * 70)
print("✅ CONFIG STATUS")
print("=" * 70)
print()

cameras_shown = []
cameras_hidden = []

if SKIP_LAPTOP_WEBCAM:
    cameras_hidden.append(f"Camera {LAPTOP_WEBCAM_INDEX} (Laptop Webcam)")
else:
    cameras_shown.append(f"Camera {LAPTOP_WEBCAM_INDEX} (Laptop Webcam)")

if HIDE_CAMERA_2:
    cameras_hidden.append("Camera 2 (Low Quality Virtual Camera)")
else:
    cameras_shown.append("Camera 2")

cameras_shown.append(f"Camera {DROIDCAM_CAMERA_INDEX} (📱 DroidCam - HIGH RES)")

print("Cameras SHOWN in web app:")
for cam in cameras_shown:
    print(f"  ✅ {cam}")

print()
print("Cameras HIDDEN:")
for cam in cameras_hidden:
    print(f"  ❌ {cam}")

print()
print("=" * 70)
print("⚠️ IMPORTANT")
print("=" * 70)
print()
print(f"✅ Select Camera {DROIDCAM_CAMERA_INDEX} for HIGH RESOLUTION (sharp, not blur)")
print(f"❌ DO NOT select Camera 2 (640x480 = VERY BLUR!)")
print()
print("=" * 70)
print("🚀 NEXT STEPS")
print("=" * 70)
print()
print("1. Restart Flask app (Ctrl+C, then START_APP.bat)")
print("2. Login as admin (admin/admin123)")
print("3. In camera dropdown, you should see:")
print(f"   • '📱 DroidCam {DROIDCAM_CAMERA_INDEX} (1920x1080)' ← Select this!")
if not HIDE_CAMERA_2:
    print("   • 'Camera 2' ← DO NOT select (blur!)")
print("4. Click 'Start Detection'")
print("5. ✅ Image should be SHARP (not blur)")
print()
print("Expected resolution: 1920x1080 (NOT 640x480)")
print()
