# Use a base image with CUDA 12.8
FROM runpod/base:cuda-12.8.0-devel-ubuntu22.04

# Set environment variables to avoid interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Install essential packages, including prerequisites for adding PPAs
# and then add the deadsnakes PPA for newer Python versions
RUN apt-get update && apt-get install -y \
    git \
    wget \
    aria2 \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y python3.12 python3.12-venv python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set python3.12 as default
RUN ln -sf /usr/bin/python3.12 /usr/bin/python
RUN ln -sf /usr/bin/python3.12 /usr/bin/python3

# Upgrade pip for the new python version
RUN python -m pip install --upgrade pip

# Set working directory
WORKDIR /

# Clone the ComfyUI repository
RUN git clone https://github.com/comfyanonymous/ComfyUI.git

# Install Python dependencies for ComfyUI using Python 3.12
RUN python -m pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
RUN python -m pip install xformers triton sage-attention
RUN python -m pip install -r /ComfyUI/requirements.txt

# Create directories for models and other data inside the volume
RUN mkdir -p /WAN/models/diffusion_models \
    /WAN/models/vae \
    /WAN/models/loras \
    /WAN/models/text_encoder \
    /WAN/models/clip_vision

# Download models using aria2 for parallel downloads
# This is more efficient for large files.
# Main Model (Corrected Path)
RUN aria2c -x 16 -s 16 -k 1M -d /WAN/models/diffusion_models -o Wan2_1-I2V-ATI-14B_fp8_e4m3fn.safetensors "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1-I2V-ATI-14B_fp8_e4m3fn.safetensors"

# VAE
RUN aria2c -x 16 -s 16 -k 1M -d /WAN/models/vae -o Wan2_1_VAE_fp32.safetensors "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1_VAE_fp32.safetensors"

# LoRA
RUN aria2c -x 16 -s 16 -k 1M -d /WAN/models/loras -o Wan21_CausVid_14B_T2V_lora_rank32_v2.safetensors "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan21_CausVid_14B_T2V_lora_rank32_v2.safetensors"

# Text Encoder
RUN aria2c -x 16 -s 16 -k 1M -d /WAN/models/text_encoder -o umt5_xxl_fp16.safetensors "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp16.safetensors"

# CLIP Vision
RUN aria2c -x 16 -s 16 -k 1M -d /WAN/models/clip_vision -o clip_vision_h.safetensors "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/clip_vision/clip_vision_h.safetensors"

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

# Create a startup script
RUN echo '#!/bin/bash' > /start.sh && \
    echo 'cd /ComfyUI' >> /start.sh && \
    echo 'python main.py --listen --port 8188 --extra-model-paths-config /extra_model_paths.yaml' >> /start.sh && \
    chmod +x /start.sh

# Create the extra_model_paths.yaml to map the volume paths (Corrected Path)
RUN echo 'wan:' > /extra_model_paths.yaml && \
    echo '  base_path: /WAN' >> /extra_model_paths.yaml && \
    echo '  checkpoints: models/diffusion_models' >> /extra_model_paths.yaml && \
    echo '  vae: models/vae' >> /extra_model_paths.yaml && \
    echo '  loras: models/loras' >> /extra_model_paths.yaml && \
    echo '  clip_vision: models/clip_vision' >> /extra_model_paths.yaml && \
    echo '  unclip_models: models/text_encoder' >> /extra_model_paths.yaml


# Set the entrypoint for the container
CMD ["/start.sh"]
