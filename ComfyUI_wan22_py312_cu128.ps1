# Setup ComfyUI with Virtual Environment for Windows

# Get the directory where the script is located
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition

# Check if Python 3.12 is installed
try {
    $pythonVersion = python --version
    if (-not ($pythonVersion -match "Python 3.12")) {
        Write-Host "Warning: This script is designed for Python 3.12. You are using $pythonVersion." -ForegroundColor Yellow
        Write-Host "You may need to install Python 3.12 from https://www.python.org/downloads/" -ForegroundColor Yellow
        $continue = Read-Host "Continue anyway? (y/n)"
        if ($continue -ne "y") {
            exit
        }
    }
}
catch {
    Write-Host "Error: Python is not installed or not in PATH." -ForegroundColor Red
    Write-Host "Please install Python 3.12 from https://www.python.org/downloads/" -ForegroundColor Red
    exit
}

# Check if Git is installed
try {
    git --version | Out-Null
}
catch {
    Write-Host "Error: Git is not installed or not in PATH." -ForegroundColor Red
    Write-Host "Please install Git from https://git-scm.com/download/win" -ForegroundColor Red
    exit
}

# Step 1: Clone ComfyUI repository
Write-Host "Cloning ComfyUI repository..." -ForegroundColor Green
Push-Location $scriptDir
if (-not (Test-Path "ComfyUI")) {
    git clone https://github.com/comfyanonymous/ComfyUI.git
}
else {
    Write-Host "ComfyUI directory already exists, skipping clone" -ForegroundColor Cyan
}
Pop-Location

$comfyUIDir = Join-Path $scriptDir "ComfyUI"


# Step 2: Create and activate Python virtual environment
Write-Host "Creating Python virtual environment..." -ForegroundColor Green
$venvPath = Join-Path $comfyUIDir "venv"
if (-not (Test-Path $venvPath)) {
    python -m venv $venvPath
}

# Activate the virtual environment
& "$venvPath\Scripts\Activate.ps1"


# Step 3: Install/upgrade Python build tools - FIXED COMMAND
Write-Host "Installing/upgrading Python build tools..." -ForegroundColor Green
& "$venvPath\Scripts\python.exe" -m pip install --upgrade pip setuptools wheel packaging


# Step 4: Install PyTorch with CUDA support
Write-Host "Installing PyTorch with CUDA support..." -ForegroundColor Green
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Step 5: Install other dependencies
Write-Host "Installing other dependencies..." -ForegroundColor Green
pip install -U "triton-windows<3.5"
# Using Windows-specific wheel for sageattention
pip install https://huggingface.co/Kijai/PrecompiledWheels/resolve/main/sageattention-2.2.0-cp312-cp312-win_amd64.whl
pip install opencv-python-headless accelerate

# Step 6: Install ComfyUI requirements
Write-Host "Installing ComfyUI requirements..." -ForegroundColor Green
$comfyUIDir = Join-Path $scriptDir "ComfyUI"
pip install -r "$comfyUIDir\requirements.txt"

# Step 7: Install custom nodes
Write-Host "Installing custom nodes..." -ForegroundColor Green
$customNodesDir = Join-Path $comfyUIDir "custom_nodes"
if (-not (Test-Path $customNodesDir)) {
    New-Item -ItemType Directory -Path $customNodesDir | Out-Null
}
Push-Location $customNodesDir

# Clone custom node repositories
$customNodes = @(
    "https://github.com/Comfy-Org/ComfyUI-Manager.git",
    "https://github.com/kijai/ComfyUI-WanVideoWrapper.git",
    "https://github.com/kijai/ComfyUI-KJNodes.git",
    "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git",
    "https://github.com/rgthree/rgthree-comfy.git",
    "https://github.com/kijai/ComfyUI-DepthAnythingV2.git",
    "https://github.com/M1kep/ComfyLiterals.git",
    "https://github.com/ClownsharkBatwing/RES4LYF.git",
    "https://github.com/chflame163/ComfyUI_LayerStyle.git",
    "https://github.com/Fannovel16/comfyui_controlnet_aux.git"
)

foreach ($repo in $customNodes) {
    $repoName = $repo -replace '.*/(.*?)\.git$', '$1'
    if (-not (Test-Path $repoName)) {
        Write-Host "Cloning $repoName" -ForegroundColor Cyan
        git clone $repo
    }
    else {
        Write-Host "$repoName already exists, skipping" -ForegroundColor Cyan
    }
}

# Install requirements for each custom node
Write-Host "Installing requirements for custom nodes..." -ForegroundColor Green
$customNodeDirs = Get-ChildItem -Directory
foreach ($dir in $customNodeDirs) {
    $reqFile = Join-Path $dir.FullName "requirements.txt"
    if (Test-Path $reqFile) {
        Write-Host "Installing requirements for $($dir.Name)" -ForegroundColor Cyan
        pip install -r $reqFile
    }
}

Pop-Location

Write-Host "Setup complete! ComfyUI is ready to use." -ForegroundColor Green
Write-Host "To start ComfyUI, run:" -ForegroundColor Yellow
Write-Host "cd '$comfyUIDir'" -ForegroundColor Yellow
Write-Host "python main.py" -ForegroundColor Yellow

# Offer to launch ComfyUI
$launch = Read-Host "Do you want to launch ComfyUI now? (y/n)"
if ($launch -eq "y") {
    Push-Location $comfyUIDir
    python main.py
}