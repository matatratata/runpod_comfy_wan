# Use multi-stage build with caching optimizations
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04 AS base

# Set environment variables to avoid interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/WAN/huggingface

# Install essential packages, including from the reference Dockerfile for better compatibility
# Corrected libgl1-mesa-glx to libgl1 for Ubuntu 24.04 compatibility
# Added ffmpeg for video processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    wget \
    aria2 \
    unzip \
    ffmpeg \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y python3.12 python3.12-venv python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set python3.12 as default
RUN ln -sf /usr/bin/python3.12 /usr/bin/python
RUN ln -sf /usr/bin/python3.12 /usr/bin/python3

# Set working directory
WORKDIR /

# Clone the ComfyUI repository
RUN git clone https://github.com/comfyanonymous/ComfyUI.git

# Install/upgrade Python build tools first to prevent compilation errors later
# Added --ignore-installed to prevent conflicts with debian-managed packages
# Added 'packaging' as a dependency for SageAttention setup
RUN python -m pip install --upgrade --ignore-installed pip setuptools wheel packaging --break-system-packages

# Install Python dependencies for ComfyUI using Python 3.12
# Added --break-system-packages to all pip installs to bypass PEP 668 error on Ubuntu 24.04
RUN python -m pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128 --break-system-packages
# Install other dependencies (xformers removed as requested)
RUN python -m pip install triton --break-system-packages

# Set target CUDA architectures for the build environment to compile SageAttention without a live GPU
# This provides the build script with the necessary info for Ampere, Ada, Hopper GPUs etc.
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"

# Compile and install SageAttention from source using setup.py as requested
RUN git clone https://github.com/thu-ml/SageAttention.git /tmp/SageAttention && \
    cd /tmp/SageAttention && \
    python setup.py install && \
    cd / && \
    rm -rf /tmp/SageAttention

# Install packages from ComfyUI's requirements file
RUN python -m pip install -r /ComfyUI/requirements.txt --break-system-packages

# Clone custom nodes for ComfyUI
WORKDIR /ComfyUI/custom_nodes
RUN git clone https://github.com/Comfy-Org/ComfyUI-Manager.git
RUN git clone https://github.com/kijai/ComfyUI-WanVideoWrapper.git
RUN git clone https://github.com/kijai/ComfyUI-KJNodes.git
RUN git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git

# Set back to the main ComfyUI directory
WORKDIR /ComfyUI

# Expose the default ComfyUI port
EXPOSE 8188

# Create the extra_model_paths.yaml to map the volume paths
# This tells ComfyUI where to look for models inside the /WAN volume
RUN echo 'wan:' > /extra_model_paths.yaml && \
    echo '  base_path: /WAN' >> /extra_model_paths.yaml && \
    echo '  checkpoints: models/diffusion_models' >> /extra_model_paths.yaml && \
    echo '  vae: models/vae' >> /extra_model_paths.yaml && \
    echo '  loras: models/loras' >> /extra_model_paths.yaml && \
    echo '  clip_vision: models/clip_vision' >> /extra_model_paths.yaml && \
    echo '  unclip_models: models/text_encoder' >> /extra_model_paths.yaml

# Create the startup script that downloads models on-demand
# This script is executed every time the pod starts
COPY --chown=root:root <<EOF /start.sh
#!/bin/bash

# Define directories within the /WAN volume
DIFFUSION_MODELS_DIR="/WAN/models/diffusion_models"
VAE_DIR="/WAN/models/vae"
LORAS_DIR="/WAN/models/loras"
TEXT_ENCODER_DIR="/WAN/models/text_encoder"
CLIP_VISION_DIR="/WAN/models/clip_vision"

# Create directories if they don't exist
mkdir -p "$DIFFUSION_MODELS_DIR" "$VAE_DIR" "$LORAS_DIR" "$TEXT_ENCODER_DIR" "$CLIP_VISION_DIR"

# --- Model Download Section ---
# For each model, check if the file exists. If not, download it.

# Main Model
if [ ! -f "$DIFFUSION_MODELS_DIR/Wan2_1-I2V-ATI-14B_fp8_e4m3fn.safetensors" ]; then
    echo "Downloading main model..."
    aria2c -x 16 -s 16 -k 1M -d "$DIFFUSION_MODELS_DIR" -o "Wan2_1-I2V-ATI-14B_fp8_e4m3fn.safetensors" "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1-I2V-ATI-14B_fp8_e4m3fn.safetensors"
fi

# VAE
if [ ! -f "$VAE_DIR/Wan2_1_VAE_fp32.safetensors" ]; then
    echo "Downloading VAE..."
    aria2c -x 16 -s 16 -k 1M -d "$VAE_DIR" -o "Wan2_1_VAE_fp32.safetensors" "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1_VAE_fp32.safetensors"
fi

# LoRA
if [ ! -f "$LORAS_DIR/Wan21_CausVid_14B_T2V_lora_rank32_v2.safetensors" ]; then
    echo "Downloading LoRA..."
    aria2c -x 16 -s 16 -k 1M -d "$LORAS_DIR" -o "Wan21_CausVid_14B_T2V_lora_rank32_v2.safetensors" "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan21_CausVid_14B_T2V_lora_rank32_v2.safetensors"
fi

# Text Encoder
if [ ! -f "$TEXT_ENCODER_DIR/umt5_xxl_fp16.safetensors" ]; then
    echo "Downloading Text Encoder..."
    aria2c -x 16 -s 16 -k 1M -d "$TEXT_ENCODER_DIR" -o "umt5_xxl_fp16.safetensors" "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp16.safetensors"
fi

# CLIP Vision
if [ ! -f "$CLIP_VISION_DIR/clip_vision_h.safetensors" ]; then
    echo "Downloading CLIP Vision..."
    aria2c -x 16 -s 16 -k 1M -d "$CLIP_VISION_DIR" -o "clip_vision_h.safetensors" "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/clip_vision/clip_vision_h.safetensors"
fi

echo "All models checked. Starting ComfyUI..."

# Launch ComfyUI
cd /ComfyUI
python main.py --listen --port 8188 --extra-model-paths-config /extra_model_paths.yaml
EOF

# Make the startup script executable
RUN chmod +x /start.sh

# Set the entrypoint for the container
CMD ["/start.sh"]