import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///chessmon.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # ========== DROIDCAM SETTINGS ==========
    # Set camera source: integer (0, 1, 2) untuk webcam/DroidCam virtual camera
    # atau URL string untuk DroidCam IP camera
    CAMERA_SOURCE = os.environ.get('CAMERA_SOURCE') or 0  # Default: webcam 0
    
    # DroidCam IP examples:
    # CAMERA_SOURCE = 'http://192.168.1.100:4747/video'  # DroidCam via WiFi
    # CAMERA_SOURCE = 1  # DroidCam virtual camera (biasanya index 1 atau 2)
    
    # Detection thresholds - adjust untuk kualitas camera berbeda
    DETECTION_CONFIDENCE = float(os.environ.get('DETECTION_CONFIDENCE', 0.5))  # Default 0.5
    NMS_IOU_THRESHOLD = float(os.environ.get('NMS_IOU_THRESHOLD', 0.4))  # Default 0.4