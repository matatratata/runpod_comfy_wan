# Fixing CUDA Issues in ComfyUI Docker Container

The error `Found no NVIDIA driver on your system` indicates that the container cannot access your GPU, even though we're using the NVIDIA CUDA base image.

## Key Fixes for the Dockerfile

1. Add diagnostic tools to the Dockerfile:

```dockerfile
# Add at the beginning after FROM line
ENV PATH="${PATH}:/usr/local/cuda/bin"
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/cuda/lib64"
ENV CUDA_HOME=/usr/local/cuda

# Install nvidia-smi
RUN apt-get update && apt-get install -y nvidia-utils-535 || apt-get install -y nvidia-utils-530 || echo "Could not install nvidia-utils"
```

2. Add CUDA verification to the start.sh script:

```dockerfile
# Inside your start.sh script, add these before launching ComfyUI:
echo "--- Checking CUDA availability ---"
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('CUDA Version:', torch.version.cuda if torch.cuda.is_available() else 'N/A'); print('GPU Device Count:', torch.cuda.device_count() if torch.cuda.is_available() else 0)"
if ! python -c "import torch; assert torch.cuda.is_available()"; then
    echo "WARNING: CUDA is not available! ComfyUI will run in CPU-only mode, which is extremely slow."
    echo "Please make sure you're running the container with GPU support: '--gpus all'"
    echo "If using RunPod, ensure you selected a GPU instance type."
fi
```

## Running the Container Correctly

The container must be started with NVIDIA GPU access:

```bash
docker run --gpus all -p 8189:8189 -v /your/models:/WAN your-image-name
```

## On RunPod

If using RunPod:
- Ensure you've selected a GPU template, not a CPU template
- The runpod.io platform should automatically mount the GPU

## Testing the GPU

Use the included verification script to test your GPU setup:

```bash
bash /cuda_verification.sh
```

This will provide detailed diagnostics about your GPU configuration.
