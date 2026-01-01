"""
GPU Availability Check for ONNX Runtime
"""
import sys

print("\n" + "="*70)
print("GPU DIAGNOSTIC CHECK")
print("="*70)

# 1. Check ONNX Runtime providers
print("\n1️⃣ ONNX Runtime Providers:")
try:
    import onnxruntime as ort
    print(f"   ONNX Runtime version: {ort.__version__}")
    available = ort.get_available_providers()
    print(f"   Available providers: {available}")
    
    if 'CUDAExecutionProvider' in available:
        print("   ✅ CUDA provider available")
    else:
        print("   ❌ CUDA provider NOT available")
        print("   ⚠️ This means GPU inference will NOT work!")
except Exception as e:
    print(f"   ❌ Error: {e}")

# 2. Check CUDA availability
print("\n2️⃣ NVIDIA CUDA:")
try:
    import torch
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   GPU count: {torch.cuda.device_count()}")
        print(f"   GPU name: {torch.cuda.get_device_name(0)}")
        print("   ✅ PyTorch can use GPU")
    else:
        print("   ❌ PyTorch cannot detect CUDA")
except ImportError:
    print("   ⚠️ PyTorch not installed (optional)")
except Exception as e:
    print(f"   ❌ Error: {e}")

# 3. Check NVIDIA driver
print("\n3️⃣ NVIDIA Driver:")
import subprocess
try:
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        lines = result.stdout.split('\n')
        for line in lines:
            if 'NVIDIA' in line or 'Driver Version' in line or 'CUDA Version' in line:
                print(f"   {line.strip()}")
        print("   ✅ NVIDIA driver detected")
    else:
        print("   ❌ nvidia-smi failed")
except FileNotFoundError:
    print("   ❌ nvidia-smi not found - NVIDIA driver may not be installed")
except Exception as e:
    print(f"   ❌ Error: {e}")

# 4. Recommendations
print("\n" + "="*70)
print("DIAGNOSIS & RECOMMENDATIONS")
print("="*70)

try:
    import onnxruntime as ort
    available = ort.get_available_providers()
    
    if 'DmlExecutionProvider' in available:
        print("\n✅ DirectML provider is available!")
        print("   GPU inference via DirectML should work.")
        print("   This uses Windows native GPU acceleration.")
    elif 'CUDAExecutionProvider' not in available:
        print("\n❌ PROBLEM: CUDA provider not available in ONNX Runtime")
        print("\n📋 POSSIBLE CAUSES:")
        print("   1. CUDA Toolkit not installed")
        print("   2. Wrong CUDA version for onnxruntime-gpu")
        print("   3. CUDA not in PATH")
        
        print("\n💡 SOLUTIONS:")
        print("\n   Option 1: Install CUDA Toolkit (RECOMMENDED)")
        print("   - Download CUDA 11.8 or 12.x from:")
        print("     https://developer.nvidia.com/cuda-downloads")
        print("   - After install, restart computer")
        print("   - Run this script again to verify")
        
        print("\n   Option 2: Use DirectML (Windows GPU without CUDA)")
        print("   - Run: pip uninstall onnxruntime-gpu")
        print("   - Run: pip install onnxruntime-directml")
        print("   - DirectML works with any GPU on Windows 10/11")
        
        print("\n   Option 3: Reduce model size (if GPU not available)")
        print("   - Use smaller input size (640 or 416 instead of 736)")
        print("   - Export model with quantization (INT8)")
    else:
        print("\n✅ CUDA provider is available!")
        print("   GPU inference should work.")
        print("   If still slow, check:")
        print("   1. Model is actually using GPU (check logs)")
        print("   2. CUDA driver is up to date")
        print("   3. GPU is not throttled (check nvidia-smi)")
except Exception as e:
    print(f"\n❌ Cannot diagnose: {e}")

print("\n" + "="*70)
