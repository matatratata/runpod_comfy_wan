```bash
# install everything in one shot
apt-get update -qq && \
apt-get install -y --no-install-recommends aria2 python3-pip && \
pip install -q "huggingface_hub[hf_xet]" && \

# grab the direct HTTPS link for the single file
# URL=$(huggingface-cli hf_hub_url \
#         --repo-type model \
#         unsloth/Qwen3-30B-A3B-GGUF \
#         Qwen3-30B-A3B-UD-Q8_K_XL.gguf) && \

URL="https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled/resolve/main/T2V/Wan2_2-T2V-A14B-LOW_fp8_e5m2_scaled_KJ.safetensors" &&

aria2c -x16 -s16 -k1M \
  --file-allocation=none \
  --continue=true \
  --dir=/workspace/models/diffusion_models \
  --out=Wan2_2-T2V-A14B-LOW_fp8_e5m2_scaled_KJ.safetensors \
  --save-session=/workspace/aria2.session \
  "$URL" &&

URL="https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled/resolve/main/T2V/Wan2_2-T2V-A14B-HIGH_fp8_e5m2_scaled_KJ.safetensors" &&

aria2c -x16 -s16 -k1M \
  --file-allocation=none \
  --continue=true \
  --dir=/workspace/models/diffusion_models \
  --out=Wan2_2-T2V-A14B-HIGH_fp8_e5m2_scaled_KJ.safetensors \
  --save-session=/workspace/aria2.session \
  "$URL" &&

URL="https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1_VAE_fp32.safetensors"

aria2c -x16 -s16 -k1M \
  --file-allocation=none \
--continue=true \
  --dir=/workspace/models/vae \
  --out=Wan2_1_VAE_fp32.safetensors \
  --save-session=/workspace/aria2.session \
  "$URL" &&

URL="https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp16.safetensors" &&

aria2c -x16 -s16 -k1M \
  --file-allocation=none \
  --continue=true \
  --dir=/workspace/models/text_encoders \
  --out=umt5_xxl_fp16.safetensors \
  --save-session=/workspace/aria2.session \
  "$URL" &&

URL="https://civitai.com/api/download/models/2079658?type=Model&format=SafeTensor" &&

aria2c -x16 -s16 -k1M \
  --file-allocation=none \
  --continue=true \
  --dir=/workspace/models/loras \
  --out=WAN2.2-HighNoise_SmartphoneSnapshotPhotoReality_v3_by-AI_Characters.safetensors \
  --save-session=/workspace/aria2.session \
  "$URL" &&

URL="https://civitai.com/api/download/models/2079614?type=Model&format=SafeTensor" &&

aria2c -x16 -s16 -k1M \
  --file-allocation=none \
  --continue=true \
  --dir=/workspace/models/loras \
  --out=WAN2.2-LowNoise_SmartphoneSnapshotPhotoReality_v3_by-AI_Characters.safetensors \
  --save-session=/workspace/aria2.session \
  "$URL"
