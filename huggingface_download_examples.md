# Downloading Files from Hugging Face in Bash

This document demonstrates different methods to download files and models from Hugging Face using bash commands.

## Prerequisites

Install the required tools:

```bash
# Install the Hugging Face Hub CLI
pip install -U "huggingface_hub[cli]"

# For downloading entire repositories
pip install -U "huggingface_hub[hf_transfer]"
```

## Method 1: Using the Hugging Face CLI

The `huggingface-cli` command provides a simple interface for downloading files:

```bash
# Download a specific file
huggingface-cli download <repo_id> <filename> --local-dir <output_directory>

# Example: Download a specific model file
huggingface-cli download Kijai/WanVideo_comfy_fp8_scaled T2V/Wan2_2-T2V-A14B-LOW_fp8_e5m2_scaled_KJ.safetensors --local-dir ./models
```

## Method 2: Using `snapshot_download()` via Python in Bash

You can use the Python `huggingface_hub` library from a bash script:

```bash
# Download an entire repository
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='mattmdjaga/segformer_b2_clothes', local_dir='./models/segformer_b2_clothes', local_dir_use_symlinks=False)"

# Download specific files from a repository
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='Kijai/WanVideo_comfy', filename='Wan2_1_VAE_fp32.safetensors', local_dir='./models/vae')"
```

## Method 3: Using hf_transfer for Faster Downloads

The `hf_transfer` protocol provides significantly faster downloads:

```bash
# Set environment variable to use hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1

# Then use snapshot_download
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Kijai/WanVideo_comfy', local_dir='./models')"
```

## Method 4: Creating a Bash Function for Multiple Files

Create a reusable bash function for downloading multiple files:

```bash
download_from_hf() {
  local repo_id=$1
  local filename=$2
  local output_dir=$3
  
  mkdir -p "$output_dir"
  python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='$repo_id', filename='$filename', local_dir='$output_dir')"
  echo "Downloaded $filename to $output_dir"
}

# Usage example
download_from_hf "Kijai/WanVideo_comfy" "Wan2_1_VAE_fp32.safetensors" "./models/vae"
download_from_hf "Kijai/WanVideo_comfy_fp8_scaled" "T2V/Wan2_2-T2V-A14B-LOW_fp8_e5m2_scaled_KJ.safetensors" "./models/diffusion_models"
```

## Method 5: Authenticating for Private Repositories

For private repositories, you'll need to authenticate:

```bash
# Login via CLI (interactive)
huggingface-cli login

# Or use a token non-interactively
export HF_TOKEN="your_token_here"

# Then download as normal
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='private-org/private-repo', local_dir='./models')"
```

## Method 6: Batch Downloading from a List

For downloading multiple files in batch:

```bash
#!/bin/bash

# Define a list of models to download
declare -A models=(
  ["Kijai/WanVideo_comfy_fp8_scaled#T2V/Wan2_2-T2V-A14B-LOW_fp8_e5m2_scaled_KJ.safetensors"]="models/diffusion_models"
  ["Kijai/WanVideo_comfy#Wan2_1_VAE_fp32.safetensors"]="models/vae"
  ["Kijai/WanVideo_comfy#umt5-xxl-enc-bf16.safetensors"]="models/text_encoders"
)

# Download each model
for key in "${!models[@]}"; do
  # Split the key into repo_id and filename
  repo_id=${key%%#*}
  filename=${key#*#}
  output_dir=${models[$key]}
  
  echo "Downloading $filename from $repo_id to $output_dir"
  mkdir -p "$output_dir"
  python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='$repo_id', filename='$filename', local_dir='$output_dir')"
done
```

## Method 7: Using curl with Hugging Face API

For simple direct downloads, you can also use curl:

```bash
# For public models
curl -L "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1_VAE_fp32.safetensors" -o ./models/vae/Wan2_1_VAE_fp32.safetensors

# For private models with authentication
curl -L "https://huggingface.co/private-org/private-model/resolve/main/model.safetensors" \
  -H "Authorization: Bearer $HF_TOKEN" \
  -o ./models/model.safetensors
```

## Comparison with aria2c

While not a Hugging Face-specific tool, `aria2c` (as used in your existing scripts) provides better performance for large files with its multi-connection capabilities:

```bash
aria2c -x16 -s16 -k1M \
  --file-allocation=none \
  --continue=true \
  --dir=./models/vae \
  --out=Wan2_1_VAE_fp32.safetensors \
  "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1_VAE_fp32.safetensors"
```

Choose the method that best suits your specific needs - `hf_transfer` for whole repositories, `huggingface-cli` for simplicity, or `aria2c` for optimized large file downloads.

```bash
cd custom_nodes
git clone https://github.com/yolain/ComfyUI-Easy-Use.git
git clone https://github.com/westNeighbor/ComfyUI-ultimate-openpose-editor.git
cd ..
cd ..
git clone https://github.com/matatratata/runpod_comfy_wan.git
cd runpod_comfy_wan
python -m venv venv
source venv/bin/activate
pip install -U "huggingface_hub[hf_transfer]"
git pull
cp dwpose_scale_limbs.py ../ComfyUI/custom_nodes/dwpose_scale_limbs.py
python3 load_models_to_net_phase01_splines.py --base_path="/WAN"
python main.py --listen --port 8189 --extra-model-paths-config /extra_model_paths.yaml
```
