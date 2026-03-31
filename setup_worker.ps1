# ==========================================
# SENTRIX WORKER SETUP (WINDOWS)
# ==========================================
# This script sets up a Windows machine as a 
# Sentrix-AI Camera Worker.
# ==========================================

Write-Host "----------------------------------------" -ForegroundColor Cyan
Write-Host "  SENTRIX-AI: Worker Setup Starting..." -ForegroundColor Cyan
Write-Host "----------------------------------------" -ForegroundColor Cyan

# 1. Check for Python
if (!(Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "❌ ERROR: Python not found!" -ForegroundColor Red
    Write-Host "💡 Please install Python from https://python.org/ and check 'Add to PATH'." -ForegroundColor Yellow
    exit 1
}

# 2. Create Virtual Environment
Write-Host "🐍 Creating virtual environment (venv_worker)..."
python -m venv venv_worker

# 3. Activate venv & Install requirements
Write-Host "🚀 Installing Python packages..."
& ".\venv_worker\Scripts\Activate.ps1"
pip install --upgrade pip

if (Test-Path "requirements_worker.txt") {
    pip install -r requirements_worker.txt
} else {
    pip install opencv-python requests ultralytics numpy
}

# 4. Check for Model
Write-Host "🧠 Checking for YOLOv8 face model..."
if (!(Test-Path "models")) { New-Item -ItemType Directory -Path "models" | Out-Null }
if (!(Test-Path "models/best.onnx")) {
    Write-Host "⚠  models/best.onnx not found!" -ForegroundColor Yellow
    Write-Host "💡 TIP: Copy 'models/best.onnx' from the server to this folder." -ForegroundColor Yellow
} else {
    Write-Host "✅ Model found: models/best.onnx" -ForegroundColor Green
}

Write-Host "----------------------------------------" -ForegroundColor Green
Write-Host "  SUCCESS: Windows Setup Complete!" -ForegroundColor Green
Write-Host "----------------------------------------" -ForegroundColor Green
Write-Host "To start the worker, run:"
Write-Host "  .\venv_worker\Scripts\Activate.ps1"
Write-Host "  python worker_agent.py --server http://<SERVER-IP>:8000 --user <user> --password <pass>"
Write-Host "----------------------------------------"
