"""
Quick test to verify app can run
"""
import sys
sys.path.insert(0, 'app')

print("="*70)
print("TESTING APP STARTUP")
print("="*70)

try:
    print("\n1. Testing imports...")
    from flask import Flask
    from chess_detection import ChessDetectionService
    print("   ✅ Flask imported")
    print("   ✅ ChessDetectionService imported")
    
    print("\n2. Testing ChessDetectionService initialization...")
    detector = ChessDetectionService(
        model_path='app/model/best.pt',
        use_onnx=False  # Python 3.14 belum support ONNX
    )
    print("   ✅ ChessDetectionService initialized successfully")
    
    print("\n3. Testing model classes...")
    # Model should be loaded
    if detector.model:
        print(f"   ✅ Model loaded with {len(detector.model.names)} classes")
        print("   Classes:", list(detector.model.names.values())[:3], "...")
    
    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED - APP READY TO RUN!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Run: cd app")
    print("  2. Run: python app.py")
    print("  3. Open: http://localhost:5000")
    print("  4. Login: admin / admin123")
    print("="*70)
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    print("\n" + "="*70)
    print("TROUBLESHOOTING:")
    print("  - Run: pip install flask flask-socketio flask-login flask-bcrypt")
    print("  - Check: python comprehensive_test.py")
    print("="*70)
