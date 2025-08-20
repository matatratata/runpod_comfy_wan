#!/bin/bash
# CUDA/GPU verification script for ComfyUI container troubleshooting

echo "===== CUDA/GPU Verification Tool for ComfyUI Container ====="
echo "This script will help diagnose issues with CUDA/GPU availability"
echo

echo "1. Checking for NVIDIA drivers:"
if command -v nvidia-smi &> /dev/null; then
    echo "✓ nvidia-smi found. Driver information:"
    nvidia-smi
else
    echo "✗ nvidia-smi command not found! NVIDIA drivers are not installed or accessible."
    echo "  - If running in Docker, ensure the container was started with '--gpus all'"
    echo "  - If on a server, ensure NVIDIA drivers are installed"
    echo "  - If using RunPod, check you selected a GPU instance"
fi

echo
echo "2. Checking CUDA environment variables:"
echo "CUDA_HOME = ${CUDA_HOME:-Not set}"
echo "LD_LIBRARY_PATH = ${LD_LIBRARY_PATH:-Not set}"
echo "PATH entries related to CUDA:"
echo $PATH | tr ':' '\n' | grep -i cuda

echo
echo "3. Testing PyTorch CUDA access:"
python3 -c "
import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('Device count:', torch.cuda.device_count())
    print('Current device:', torch.cuda.current_device())
    print('Device name:', torch.cuda.get_device_name(0))
    print('CUDA version:', torch.version.cuda)
    # Test basic CUDA operation
    print('Testing CUDA tensor:')
    x = torch.rand(5, 3).cuda()
    print(x)
else:
    print('CUDA NOT AVAILABLE TO PYTORCH!')
    print('This means ComfyUI will run in CPU-only mode (very slow).')
"

echo
echo "4. Common issues and solutions:"
echo "  - Docker: Container must be started with '--gpus all'"
echo "  - Missing NVIDIA Container Toolkit: Install nvidia-docker2"
echo "  - RunPod: Select a GPU pod type, not CPU"
echo "  - Check compatibility between CUDA version and GPU drivers"
echo
echo "For ComfyUI container, modify your start command to:"
echo "docker run --gpus all -p 8189:8189 -v /your/models:/WAN your-image-name"
echo

chmod +x /cuda_verification.sh
