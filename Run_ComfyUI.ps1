# Get the directory where the script is located
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition

# Define path to ComfyUI
$comfyUIDir = Join-Path $scriptDir "ComfyUI"

# Go to ComfyUI directory
Push-Location $comfyUIDir

# Activate the virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Green
$venvActivate = Join-Path $comfyUIDir "venv\Scripts\Activate.ps1"
& $venvActivate

# Run ComfyUI
Write-Host "Starting ComfyUI..." -ForegroundColor Green
python main.py

# Return to original directory when script exits
Pop-Location