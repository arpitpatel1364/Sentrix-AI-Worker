# 🛰️ Sentrix-AI: Worker Setup Guide (Windows & Linux)

This document provides the official setup process for physical and virtual camera nodes in the Sentrix mesh.

---

## 🐧 Linux Setup (Ubuntu/Debian)

### Option A: Automated (Fastest)
```bash
# Clone/Copy project, then run:
chmod +x setup_worker.sh
./setup_worker.sh
```

### Option B: Manual
1. **Install System Dependencies**:
   ```bash
   sudo apt update && sudo apt install -y python3-venv libgl1-mesa-glx libglib2.0-0
   ```
2. **Create Environment**:
   ```bash
   python3 -m venv venv_worker
   source venv_worker/bin/activate
   ```
3. **Install Packages**:
   ```bash
   pip install -r requirements_worker.txt
   ```

---

## 🪟 Windows Setup

### Option A: Automated PowerShell
1. Open **PowerShell** as Administrator.
2. Run the following command in the project folder:
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
   .\setup_worker.ps1
   ```

### Option B: Manual
1. **Install Python**: Download from [python.org](https://www.python.org/) (Ensure "Add to PATH" is checked).
2. **Create Environment**:
   ```powershell
   python -m venv venv_worker
   ```
3. **Activate Environment**:
   ```powershell
   .\venv_worker\Scripts\Activate.ps1
   ```
4. **Install Packages**:
   ```powershell
   pip install -r requirements_worker.txt
   ```

---

## 🏗️ Registration & Deployment (All Platforms)

1. **Register the Worker** (Run this on the **Server** machine):
   ```bash
   python manage_workers.py add --id "cam-01" --location "Office" --camera 0 --user "worker1" --password "pass123"
   ```

2. **Run the Agent** (Run this on the **Worker** machine):
   ```bash
   # Make sure venv is active!
   python worker_agent.py --server http://<SERVER-IP>:8000 --user "worker1" --password "pass123" --camera 0
   ```

---

## 🐳 Docker Setup (Enterprise)
If you have Docker installed (Windows/Linux/Mac):
```bash
# Build the image
docker build -t sentrix-worker -f Dockerfile.worker .

# Run (Linux example)
docker run -d --name worker-1 --device=/dev/video0:/dev/video0 sentrix-worker
```

---

## 🔍 Common Issues
- **Camera Not Found**: Ensure no other application (Zoom, Teams) is using the camera.
- **Connection Refused**: Verify the Server's Firewall allows traffic on port 8000.
- **Permissions (Linux)**: Run `sudo usermod -a -G video $USER` if access is denied.
