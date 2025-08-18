```bash
# install everything in one shot
apt-get update -qq && \
apt-get install -y --no-install-recommends aria2 python3-pip && \
pip install -q "huggingface_hub[hf_xet]" 
URL="https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled/resolve/main/T2V/Wan2_2-T2V-A14B-LOW_fp8_e5m2_scaled_KJ.safetensors" && \
aria2c -x16 -s16 -k1M \
  --file-allocation=none \
  --continue=true \
  --dir=/workspace/models/diffusion_models \
  --out=Wan2_2-T2V-A14B-LOW_fp8_e5m2_scaled_KJ.safetensors \
  --save-session=/workspace/aria2.session \
  "$URL" && 
URL="https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled/resolve/main/T2V/Wan2_2-T2V-A14B-HIGH_fp8_e5m2_scaled_KJ.safetensors" && \
aria2c -x16 -s16 -k1M \
  --file-allocation=none \
  --continue=true \
  --dir=/workspace/models/diffusion_models \
  --out=Wan2_2-T2V-A14B-HIGH_fp8_e5m2_scaled_KJ.safetensors \
  --save-session=/workspace/aria2.session \
  "$URL" &&
URL="https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1_VAE_fp32.safetensors" && \
aria2c -x16 -s16 -k1M \
  --file-allocation=none \
  --continue=true \
  --dir=/workspace/models/vae \
  --out=Wan2_1_VAE_fp32.safetensors \
  --save-session=/workspace/aria2.session \
  "$URL" && 
URL="https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp16.safetensors" && \
aria2c -x16 -s16 -k1M \
  --file-allocation=none \
  --continue=true \
  --dir=/workspace/models/text_encoders \
  --out=umt5_xxl_fp16.safetensors \
  --save-session=/workspace/aria2.session \
  "$URL" && 
URL="https://civitai.com/api/download/models/2079658?type=Model&format=SafeTensor" && \
aria2c -x16 -s16 -k1M \
  --file-allocation=none \
  --continue=true \
  --dir=/workspace/models/loras \
  --out=WAN2.2-HighNoise_SmartphoneSnapshotPhotoReality_v3_by-AI_Characters.safetensors \
  --save-session=/workspace/aria2.session \
  "$URL" && 
URL="https://civitai.com/api/download/models/2079614?type=Model&format=SafeTensor" && \
aria2c -x16 -s16 -k1M \
  --file-allocation=none \
  --continue=true \
  --dir=/workspace/models/loras \
  --out=wan2.2-lownoise_smartphonesnapshotphotoreality_v3_by-ai_characters.safetensors \
  --save-session=/workspace/aria2.session \
  "$url" && 
URL="https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/umt5-xxl-enc-bf16.safetensors" && \
aria2c -x16 -s16 -k1M \
  --file-allocation=none \
  --continue=true \
  --dir=/workspace/models/text_encoders \
  --out=umt5-xxl-enc-bf16.safetensors \
  --save-session=/workspace/aria2.session \
  "$URL" && 
URL="https://huggingface.co/unsloth/FLUX.1-Kontext-dev-GGUF/resolve/main/flux1-kontext-dev-Q8_0.gguf" && \
aria2c -x16 -s16 -k1M \
  --file-allocation=none \
  --continue=true \
  --dir=/workspace/models/unet \
  --out=flux1-kontext-dev-Q8_0.gguf \
  --save-session=/workspace/aria2.session \
  "$URL"
URL="https://huggingface.co/zer0int/CLIP-KO-ViT-L-14-336-TypoAttack/resolve/main/ViT-L-14-336-KO-LITE-FULL-model-OpenAI-format.safetensors" && \
aria2c -x16 -s16 -k1M \
  --file-allocation=none \
  --continue=true \
  --dir=/workspace/models/clip \
  --out=ViT-L-14-336-KO-LITE-FULL-model-OpenAI-format.safetensors \
  --save-session=/workspace/aria2.session \
  "$URL"
URL="https://huggingface.co/zer0int/CLIP-GmP-ViT-L-14/resolve/main/ViT-L-14-TEXT-detail-improved-hiT-GmP-HF.safetensors" && \
aria2c -x16 -s16 -k1M \
  --file-allocation=none \
  --continue=true \
  --dir=/workspace/models/clip \
  --out=ViT-L-14-TEXT-detail-improved-hiT-GmP-HF.safetensors \
  --save-session=/workspace/aria2.session \
  "$URL"
URL="https://huggingface.co/StableDiffusionVN/Flux/resolve/main/Vae/flux_vae.safetensors" && \
aria2c -x16 -s16 -k1M \
  --file-allocation=none \
  --continue=true \
  --dir=/workspace/models/vae \
  --out=flux_vae.safetensors \
  --save-session=/workspace/aria2.session \
  "$URL"
URL="https://huggingface.co/easygoing0114/AI_upscalers/resolve/main/4x-PBRify_RPLKSRd_V3.pth" && \
aria2c -x16 -s16 -k1M \
  --file-allocation=none \
  --continue=true \
  --dir=/workspace/models/upscale_models \
  --out=4x-PBRify_RPLKSRd_V3.pth \
  --save-session=/workspace/aria2.session \
  "$URL"
URL="https://huggingface.co/easygoing0114/AI_upscalers/resolve/main/4x_foolhardy_Remacri_ExtraSmoother.pth" && \
aria2c -x16 -s16 -k1M \
  --file-allocation=none \
  --continue=true \
  --dir=/workspace/models/upscale_models \
  --out=4x_foolhardy_Remacri_ExtraSmoother.pth \
  --save-session=/workspace/aria2.session \
  "$URL"
URL="https://huggingface.co/StableDiffusionVN/Flux/resolve/main/Clip/t5xxl_fp16.safetensors" && \
aria2c -x16 -s16 -k1M \
  --file-allocation=none \
  --continue=true \
  --dir=/workspace/models/clip \
  --out=t5xxl_fp16.safetensors \
  --save-session=/workspace/aria2.session \
  "$URL"
URL="https://huggingface.co/StableDiffusionVN/Flux/resolve/main/Clip/clip_l.safetensors" && \
aria2c -x16 -s16 -k1M \
  --file-allocation=none \
  --continue=true \
  --dir=/workspace/models/clip \
  --out=clip_l.safetensors \
  --save-session=/workspace/aria2.session \
  "$URL"



URL="https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled/resolve/main/T2V/Wan2_1-T2V-14B-Phantom_fp8_e5m2_scaled_KJ.safetensors" && \
aria2c -x16 -s16 -k1M \
  --file-allocation=none \
  --continue=true \
  --dir=/workspace/models/diffusion_models \
  --out=Wan2_1-T2V-14B-Phantom_fp8_e5m2_scaled_KJ.safetensors \
  --save-session=/workspace/aria2.session \
  "$URL"
URL="https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan21_Uni3C_controlnet_fp16.safetensors" && \
aria2c -x16 -s16 -k1M \
  --file-allocation=none \
  --continue=true \
  --dir=/workspace/models/diffusion_models \
  --out=Wan21_Uni3C_controlnet_fp16.safetensors \
  --save-session=/workspace/aria2.session \
  "$URL"
URL="https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1-VACE_module_14B_bf16.safetensors" && \
aria2c -x16 -s16 -k1M \
  --file-allocation=none \
  --continue=true \
  --dir=/workspace/models/diffusion_models \
  --out=Wan2_1-VACE_module_14B_bf16.safetensors \
  --save-session=/workspace/aria2.session \
  "$URL"
URL="https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled/resolve/main/I2V/Wan2_1-I2V-14B-MAGREF_fp8_e4m3fn_scaled_KJ.safetensors" && \
aria2c -x16 -s16 -k1M \
  --file-allocation=none \
  --continue=true \
  --dir=/workspace/models/diffusion_models \
  --out=Wan2_1-I2V-14B-MAGREF_fp8_e4m3fn_scaled_KJ.safetensors \
  --save-session=/workspace/aria2.session \
  "$URL"
URL="https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled/resolve/main/I2V/Wan2_2-I2V-A14B-HIGH_fp8_e5m2_scaled_KJ.safetensors" && \
aria2c -x16 -s16 -k1M \
  --file-allocation=none \
  --continue=true \
  --dir=/workspace/models/diffusion_models \
  --out=Wan2_2-I2V-A14B-HIGH_fp8_e5m2_scaled_KJ.safetensors \
  --save-session=/workspace/aria2.session \
  "$URL"
URL="https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled/resolve/main/I2V/Wan2_2-I2V-A14B-LOW_fp8_e5m2_scaled_KJ.safetensors" && \
aria2c -x16 -s16 -k1M \
  --file-allocation=none \
  --continue=true \
  --dir=/workspace/models/diffusion_models \
  --out=Wan2_2-I2V-A14B-LOW_fp8_e5m2_scaled_KJ.safetensors \
  --save-session=/workspace/aria2.session \
  "$URL"
URL="https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/FantasyPortrait/Wan2_1_FantasyPortrait_fp16.safetensors" && \
aria2c -x16 -s16 -k1M \
  --file-allocation=none \
  --continue=true \
  --dir=/workspace/models/diffusion_models \
  --out=Wan2_1_FantasyPortrait_fp16.safetensors \
  --save-session=/workspace/aria2.session \
  "$URL"
URL="https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Stand-In/Stand-In_wan2.1_T2V_14B_ver1.0_fp32.safetensors" && \
aria2c -x16 -s16 -k1M \
  --file-allocation=none \
  --continue=true \
  --dir=/workspace/models/diffusion_models \
  --out=Stand-In_wan2.1_T2V_14B_ver1.0_fp32.safetensors \
  --save-session=/workspace/aria2.session \
  "$URL"
URL="https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Skyreels/Wan2_1_Skyreels-v2-I2V-720P_LoRA_rank_adaptive_quantile_0.20_fp16.safetensors" && \
aria2c -x16 -s16 -k1M \
  --file-allocation=none \
  --continue=true \
  --dir=/workspace/models/loras \
  --out=Wan2_1_Skyreels-v2-I2V-720P_LoRA_rank_adaptive_quantile_0.20_fp16.safetensors \
  --save-session=/workspace/aria2.session \
  "$URL"
URL="https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan22-Lightning/Wan2.2-Lightning_T2V-v1.1-A14B-4steps-lora_HIGH_fp16.safetensors" && \
aria2c -x16 -s16 -k1M \
  --file-allocation=none \
  --continue=true \
  --dir=/workspace/models/loras \
  --out=Wan2.2-Lightning_T2V-v1.1-A14B-4steps-lora_HIGH_fp16.safetensors \
  --save-session=/workspace/aria2.session \
  "$URL"
URL="https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan22-Lightning/Wan2.2-Lightning_T2V-v1.1-A14B-4steps-lora_LOW_fp16.safetensors" && \
aria2c -x16 -s16 -k1M \
  --file-allocation=none \
  --continue=true \
  --dir=/workspace/models/loras \
  --out=Wan2.2-Lightning_T2V-v1.1-A14B-4steps-lora_LOW_fp16.safetensors \
  --save-session=/workspace/aria2.session \
  "$URL"
URL="https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan22-Lightning/Wan2.2-Lightning_I2V-A14B-4steps-lora_HIGH_fp16.safetensors" && \
aria2c -x16 -s16 -k1M \
  --file-allocation=none \
  --continue=true \
  --dir=/workspace/models/loras \
  --out=Wan2.2-Lightning_I2V-A14B-4steps-lora_HIGH_fp16.safetensors \
  --save-session=/workspace/aria2.session \
  "$URL"
URL="https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan22-Lightning/Wan2.2-Lightning_I2V-A14B-4steps-lora_LOW_fp16.safetensors" && \
aria2c -x16 -s16 -k1M \
  --file-allocation=none \
  --continue=true \
  --dir=/workspace/models/loras \
  --out=Wan2.2-Lightning_I2V-A14B-4steps-lora_LOW_fp16.safetensors \
  --save-session=/workspace/aria2.session \
  "$URL"
URL="https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors" && \
aria2c -x16 -s16 -k1M \
  --file-allocation=none \
  --continue=true \
  --dir=/workspace/models/loras \
  --out=Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors \
  --save-session=/workspace/aria2.session \
  "$URL"
URL="https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Lightx2v/lightx2v_14B_T2V_cfg_step_distill_lora_adaptive_rank_quantile_0.15_bf16.safetensors" && \
aria2c -x16 -s16 -k1M \
  --file-allocation=none \
  --continue=true \
  --dir=/workspace/models/loras \
  --out=lightx2v_14B_T2V_cfg_step_distill_lora_adaptive_rank_quantile_0.15_bf16.safetensors \
  --save-session=/workspace/aria2.session \
  "$URL"
URL="https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Pusa/Wan21_PusaV1_LoRA_14B_rank512_bf16.safetensors" && \
aria2c -x16 -s16 -k1M \
  --file-allocation=none \
  --continue=true \
  --dir=/workspace/models/loras \
  --out=Wan21_PusaV1_LoRA_14B_rank512_bf16.safetensors \
  --save-session=/workspace/aria2.session \
  "$URL"
URL="https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/clip_vision/clip_vision_h.safetensors" && \
aria2c -x16 -s16 -k1M \
  --file-allocation=none \
  --continue=true \
  --dir=/workspace/models/clip_vision \
  --out=clip_vision_h.safetensors \
  --save-session=/workspace/aria2.session \
  "$URL"
URL="https://huggingface.co/ByteDance/DreamO/resolve/main/comfyui/dreamo_cfg_distill_comfyui.safetensors" && \
aria2c -x16 -s16 -k1M \
  --file-allocation=none \
  --continue=true \
  --dir=/workspace/models/loras \
  --out=dreamo_cfg_distill_comfyui.safetensors \
  --save-session=/workspace/aria2.session \
  "$URL"
URL="https://huggingface.co/ByteDance/DreamO/resolve/main/comfyui/dreamo_comfyui.safetensors" && \
aria2c -x16 -s16 -k1M \
  --file-allocation=none \
  --continue=true \
  --dir=/workspace/models/loras \
  --out=dreamo_comfyui.safetensors \
  --save-session=/workspace/aria2.session \
  "$URL"
URL="https://huggingface.co/ByteDance/DreamO/resolve/main/comfyui/dreamo_quality_lora_neg_comfyui.safetensors" && \
aria2c -x16 -s16 -k1M \
  --file-allocation=none \
  --continue=true \
  --dir=/workspace/models/loras \
  --out=dreamo_quality_lora_neg_comfyui.safetensors \
  --save-session=/workspace/aria2.session \
  "$URL"
URL="https://huggingface.co/ByteDance/DreamO/resolve/main/comfyui/dreamo_quality_lora_pos_comfyui.safetensors" && \
aria2c -x16 -s16 -k1M \
  --file-allocation=none \
  --continue=true \
  --dir=/workspace/models/loras \
  --out=dreamo_quality_lora_pos_comfyui.safetensors \
  --save-session=/workspace/aria2.session \
  "$URL"
URL="https://huggingface.co/alibaba-pai/Wan2.1-Fun-Reward-LoRAs/resolve/main/Wan2.1-Fun-14B-InP-MPS.safetensors" && \
aria2c -x16 -s16 -k1M \
  --file-allocation=none \
  --continue=true \
  --dir=/workspace/models/loras \
  --out=Wan2.1-Fun-14B-InP-MPS.safetensors \
  --save-session=/workspace/aria2.session \
  "$URL"
URL="https://huggingface.co/drozbay/Wan2.2_A14B_lora_extract/resolve/main/Wan22_A14B_T2V_lora_extract_r64.safetensors" && \
aria2c -x16 -s16 -k1M \
  --file-allocation=none \
  --continue=true \
  --dir=/workspace/models/loras \
  --out=Wan22_A14B_T2V_lora_extract_r64.safetensors \
  --save-session=/workspace/aria2.session \
  "$URL"
URL="https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan21_CausVid_14B_T2V_lora_rank32_v2.safetensors" && \
aria2c -x16 -s16 -k1M \
  --file-allocation=none \
  --continue=true \
  --dir=/workspace/models/loras \
  --out=Wan21_CausVid_14B_T2V_lora_rank32_v2.safetensors \
  --save-session=/workspace/aria2.session \
  "$URL"

# Create directory for the model
mkdir -p /workspace/models/segformer_b2_clothes

# Use Python with huggingface_hub to download the model files
python3 -c "
from huggingface_hub import snapshot_download
import os

# Download the complete repository
snapshot_download(
    repo_id='mattmdjaga/segformer_b2_clothes',
    local_dir='/workspace/models/segformer_b2_clothes',
    local_dir_use_symlinks=False
)

print('Download complete: Model files saved to /ComfyUI/models/segformer_b2_clothes')
"

mkdir -p /workspace/models/segformer_b3_clothes

python3 -c "
from huggingface_hub import snapshot_download
import os

# Download the complete repository
snapshot_download(
    repo_id='sayeed99/segformer_b3_clothes',
    local_dir='/workspace/models/segformer_b3_clothes',
    local_dir_use_symlinks=False
)

print('Download complete: Model files saved to /workspace/models/segformer_b3_clothes')
"