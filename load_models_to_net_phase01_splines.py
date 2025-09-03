#!/usr/bin/env python3

import os
import argparse
import subprocess
from huggingface_hub import snapshot_download


def download_file(url, save_path):
    """Download a file using aria2c with progress reporting"""
    try:
        # Check if file already exists
        if os.path.exists(save_path):
            print(f"File already exists, skipping: {save_path}")
            return True

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Prepare aria2c command
        command = [
            "aria2c",
            "--max-connection-per-server=16",  # Maximum connections
            "--split=16",  # Maximum split
            "--min-split-size=1M",  # Minimum split size
            "--dir",
            os.path.dirname(save_path),
            "--out",
            os.path.basename(save_path),
            "--summary-interval=0",  # Disable console summary
            "--show-console-readout=false",  # Disable console readout
            url,
        ]

        print(f"Downloading: {os.path.basename(save_path)}")
        result = subprocess.run(command, check=True)

        if result.returncode == 0:
            print(f"Successfully downloaded {save_path}")
            return True
        else:
            print(f"Failed to download {url}")
            return False

    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False


def download_models(base_path):
    """Download all models to the specified base path"""
    # Create all required directories
    directories = [
        "models/diffusion_models",
        "models/vae",
        "models/text_encoders",
        "models/loras",
        "models/unet",
        "models/clip",
        "models/upscale_models",
        "models/clip_vision",
    ]

    for directory in directories:
        os.makedirs(os.path.join(base_path, directory), exist_ok=True)

    # List of models to download
    downloads = [
        # T2V models
        {
            "url": "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1-T2V-14B_fp8_e5m2.safetensors",
            "dir": "models/diffusion_models",
            "out": "Wan2_1-T2V-14B_fp8_e5m2.safetensors",
        },
        # VAE
        {
            "url": "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1_VAE_fp32.safetensors",
            "dir": "models/vae",
            "out": "Wan2_1_VAE_fp32.safetensors",
        },
        # Text encoders
        {
            "url": "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp16.safetensors",
            "dir": "models/text_encoders",
            "out": "umt5_xxl_fp16.safetensors",
        },
        {
            "url": "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1_VAE_bf16.safetensors",
            "dir": "models/vae",
            "out": "Wan2_1_VAE_bf16.safetensors",
        },
        # More models - all remaining from the original script
        {
            "url": "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1-VACE_module_14B_fp8_e4m3fn.safetensors",
            "dir": "models/diffusion_models",
            "out": "Wan2_1-VACE_module_14B_fp8_e4m3fn.safetensors",
        },
        {
            "url": "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors",
            "dir": "models/loras",
            "out": "Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors",
        },
        {
            "url": "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Lightx2v/lightx2v_14B_T2V_cfg_step_distill_lora_adaptive_rank_quantile_0.15_bf16.safetensors",
            "dir": "models/loras",
            "out": "lightx2v_14B_T2V_cfg_step_distill_lora_adaptive_rank_quantile_0.15_bf16.safetensors",
        },
        {
            "url": "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/clip_vision/clip_vision_h.safetensors",
            "dir": "models/clip_vision",
            "out": "clip_vision_h.safetensors",
        },
        {
            "url": "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan21_CausVid_14B_T2V_lora_rank32_v2.safetensors",
            "dir": "models/loras",
            "out": "Wan21_CausVid_14B_T2V_lora_rank32_v2.safetensors",
        },
        {
            "url": "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/taew2_1.safetensors",
            "dir": "models/vae",
            "out": "taew2_1.safetensors",
        },
    ]

    # Download files one at a time with aria2c
    successful_downloads = 0
    failed_downloads = 0
    skipped_downloads = 0

    for item in downloads:
        save_path = os.path.join(base_path, item["dir"], item["out"])

        if os.path.exists(save_path):
            print(f"File already exists, skipping: {item['out']}")
            skipped_downloads += 1
            continue

        result = download_file(item["url"], save_path)
        if result:
            successful_downloads += 1
        else:
            failed_downloads += 1

    print(
        f"Download summary: {successful_downloads} successful, {failed_downloads} failed, {skipped_downloads} skipped"
    )

    # Download segformer models using HuggingFace Hub
    segformer_b2_dir = os.path.join(base_path, "models/segformer_b2_clothes")
    if os.path.exists(segformer_b2_dir) and os.listdir(segformer_b2_dir):
        print(
            f"segformer_b2_clothes already exists at {segformer_b2_dir}, skipping download"
        )
    else:
        print("Downloading segformer_b2_clothes model...")
        os.makedirs(segformer_b2_dir, exist_ok=True)
        snapshot_download(
            repo_id="mattmdjaga/segformer_b2_clothes",
            local_dir=segformer_b2_dir,
            local_dir_use_symlinks=False,
        )
        print(f"Download complete: Model files saved to {segformer_b2_dir}")

    segformer_b3_dir = os.path.join(base_path, "models/segformer_b3_clothes")
    if os.path.exists(segformer_b3_dir) and os.listdir(segformer_b3_dir):
        print(
            f"segformer_b3_clothes already exists at {segformer_b3_dir}, skipping download"
        )
    else:
        print("Downloading segformer_b3_clothes model...")
        os.makedirs(segformer_b3_dir, exist_ok=True)
        snapshot_download(
            repo_id="sayeed99/segformer_b3_clothes",
            local_dir=segformer_b3_dir,
            local_dir_use_symlinks=False,
        )
        print(f"Download complete: Model files saved to {segformer_b3_dir}")


def main():
    parser = argparse.ArgumentParser(description="Download WAN models for ComfyUI")
    parser.add_argument(
        "--base_path",
        type=str,
        default="/workspace",
        help="Base directory where models will be stored (default: /workspace)",
    )
    args = parser.parse_args()

    print(f"Starting download of models to {args.base_path}")
    download_models(args.base_path)
    print("All downloads completed!")


if __name__ == "__main__":
    main()
