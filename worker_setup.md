# 🛰️ Sentrix-AI: Worker Setup Guide

This document provides the official setup process for physical and virtual camera nodes in the Sentrix mesh.

---

## 🏗️ Platform Selection

### 🐧 Linux (Ubuntu/Debian)
1. **Automated Setup**:
   ```bash
   chmod +x setup_worker.sh
   ./setup_worker.sh
   ```
2. **Manual Setup**:
   - Install system dependencies: `sudo apt update && sudo apt install -y python3-venv libgl1-mesa-glx libglib2.0-0`
   - Create Venv: `python3 -m venv venv_worker && source venv_worker/bin/activate`
   - Install requirements: `pip install -r requirements_worker.txt`

### 🪟 Windows
1. **Automated Setup (PowerShell)**:
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
   .\setup_worker.ps1
   ```
2. **Manual Setup**:
   - Install Python 3.10+ from python.org.
   - Create Venv: `python -m venv venv_worker`
   - Activate: `.\venv_worker\Scripts\Activate.ps1`
   - Install requirements: `pip install -r requirements_worker.txt`

---

## 🚀 Registration & Deployment

### 1. Register Local Identity
Run this on your **Central Hub (Server)** to authorize this worker node:
```bash
python manage_workers.py add --id "local-cam" --location "Office" --camera 0 --user "worker1" --password "pass123"
```

### 2. Launch the Intelligent Agent
Run the agent on the machine where the cameras are connected.

#### **Standard Launch**
```bash
python worker_agent.py --server http://<HUB-IP>:8000 --user "worker1" --password "pass123" --camera 0
```

#### **Multi-Camera Deployment**
The agent supports native multi-threading for multiple cameras or RTSP streams:
```bash
python worker_agent.py --server http://<HUB-IP>:8000 --user "worker1" --password "pass123" \
  --camera 0 1 \
  --camera-id "gate" "lobby" \
  --location "Main Entrance" "Front Desk"
```

---

## 🧠 Advanced Configuration

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--interval` | Seconds between detection passes | `3.0` |
| `--no-model` | Skip local YOLO detection (Server-side only) | `False` |
| `--model` | Path to custom `.onnx` or `.pt` model | `models/best.onnx` |
| `--server` | Target Hub URL | `http://localhost:8000` |

---

## 🔍 Common Issues

- **Multiple Cameras on One Hub**: If you encounter errors when running 2+ USB cameras, try plugging them into different USB buses (e.g., one on the front, one on the back). Simple Windows/Linux hubs often lack the bandwidth for multiple raw streams.
- **CUDA Activation**: If you have an NVIDIA GPU but the logs say "Using CPU", ensure `onnxruntime-gpu` is installed in your `venv_worker`.
- **Permissions**: On Linux, if the camera won't open, run `sudo usermod -a -G video $USER`.

---

> [!IMPORTANT]
> **Bandwidth Savings**: The worker agent automatically detects motion and only uploads face crops, saving up to 95% of network bandwidth compared to raw video streaming.

