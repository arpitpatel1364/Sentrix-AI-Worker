#!/bin/bash

# ==========================================
# SENTRIX WORKER SETUP SCRIPT
# ==========================================
# This script sets up a Linux machine as a 
# Sentrix-AI Camera Worker.
# ==========================================

# Exit on error
set -e

echo "----------------------------------------"
echo "  SENTRIX-AI: Worker Setup Starting..."
echo "----------------------------------------"

# 1. Update & Install system dependencies (OpenCV needs these)
echo "📦 Installing system dependencies..."
sudo apt-get update && sudo apt-get install -y \
    python3-pip \
    python3-venv \
    libgl1-mesa-glx \
    libglib2.0-0

# 2. Create Virtual Environment
echo "🐍 Creating virtual environment (venv_worker)..."
python3 -m venv venv_worker

# 3. Activate venv & Install requirements
echo "🚀 Installing Python packages..."
source venv_worker/bin/activate
pip install --upgrade pip
if [ -f "requirements_worker.txt" ]; then
    pip install -r requirements_worker.txt
else
    echo "⚠  requirements_worker.txt not found! Installing defaults..."
    pip install opencv-python requests ultralytics numpy onnxruntime
fi

# 4. Check for Model
echo "🧠 Checking for YOLOv8 face model..."
mkdir -p models
if [ ! -f "models/best.onnx" ]; then
    echo "⚠  models/best.onnx not found!"
    echo "💡 TIP: Copy 'models/best.onnx' from the server to this folder."
else
    echo "✅ Model found: models/best.onnx"
fi

echo "----------------------------------------"
echo "  SUCCESS: Setup Complete!"
echo "----------------------------------------"
echo "To add this worker to the system, run on your MANAGER (Server) machine:"
echo "  python3 admin/manage_workers.py add --id <cam-id> --location '<location>' --user <user> --password <pass>"
echo ""
echo "To start the worker (Multi-Camera Mode):"
echo "  source venv_worker/bin/activate"
echo "  python3 worker_agent.py --server http://<server-ip>:8000 --user <user> --password <pass> --camera 0 1"
echo "----------------------------------------"
