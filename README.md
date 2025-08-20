RunPod ComfyUI Template for WanVideo

This repository contains a RunPod template for ComfyUI, specifically configured for use with the WanVideo models and nodes. It uses a Docker-based setup to ensure a consistent and reproducible environment.
Features

    CUDA 12.8: Utilizes a newer CUDA version for better performance with recent GPUs.

    PyTorch Nightly: Includes a recent nightly build of PyTorch to leverage the latest features and optimizations, including support for f16fast.

    Pre-installed Models: Automatically downloads the required WanVideo models, VAE, LoRA, and text encoders.

    Custom Nodes: Includes popular and necessary custom nodes for a complete workflow:

        ComfyUI-Manager

        ComfyUI-WanVideoWrapper

        ComfyUI-KJNodes

        ComfyUI-VideoHelperSuite

    Volume-based Storage: Designed to be used with a RunPod Volume (named WAN) to persist your models and outputs.

How to Use with RunPod

    Create a RunPod Template:

        Go to My Templates in your RunPod account.

        Click New Template.

        Give your template a name (e.g., ComfyUI-WanVideo).

        For the Container Image, enter the name for your image, which you will build and push to a container registry (like Docker Hub). For example: yourdockerhubusername/comfyui-wan:latest.

        Set the Container Disk to at least 25GB to accommodate the models and ComfyUI.

        You do not need to set a Volume Path here, as we will be mounting it during pod creation.

        In the Docker Start Command field, you can leave this blank as the CMD is specified in the Dockerfile.

        Save the template.

    Build and Push the Docker Image:

        Make sure you have Docker installed on your local machine.

        Clone this repository:

        git clone https://github.com/matatratata/runpod_comfy_wan.git
        cd runpod_comfy_wan

        Build the Docker image. Replace yourdockerhubusername with your actual Docker Hub username.

        docker build -t yourdockerhubusername/comfyui-wan:latest .

        Log in to Docker Hub:

        docker login

        Push the image to Docker Hub:

        docker push yourdockerhubusername/comfyui-wan:latest

    Launch a Pod:

        Go to the Community Cloud or Secure Cloud to rent a GPU.

        Select your desired GPU pod. A 4090 or similar is recommended.

        In the template selection, choose your ComfyUI-WanVideo template.

        Under Volume, either create a new 64 GB volume or attach your existing one. Crucially, set the Mount Path to /WAN. This is the path the Dockerfile expects.

        Customize any other settings as needed (e.g., ports).

        Click Deploy.

    Connect to ComfyUI:

        Once your pod is running, click Connect.

        You will see a button to Connect to Port 8188 (or whichever port you have configured). Click it.

        This will open the ComfyUI interface in a new tab, ready to use with all your models and custom nodes.

File Structure

    Dockerfile: The main file that defines the Docker image, installs dependencies, and downloads all necessary files.

    .dockerignore: To exclude files from the Docker build context.

    README.md: This file.

# ComfyUI RunPod Container

## Important: Running with CUDA Support

To ensure CUDA is properly available inside the container, you must start it with GPU support:

```bash
docker run --gpus all -p 8189:8189 -v /path/to/your/models:/WAN your-image-name
```

### Prerequisites:

1. NVIDIA GPU with up-to-date drivers
2. Docker with NVIDIA Container Toolkit installed (nvidia-docker2)

### Verifying CUDA:

The startup script will automatically check for CUDA availability and print information about your GPU. If you see a warning about CUDA not being available, check that:

1. You've started the container with the `--gpus all` flag
2. Your host system has properly installed NVIDIA drivers
3. The NVIDIA Container Toolkit is correctly installed and configured

For RunPod users, CUDA should work automatically when deployed on GPU instances.

Enjoy creating amazing videos with your custom ComfyUI setup!