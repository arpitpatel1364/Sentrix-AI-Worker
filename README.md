# Sentrix AI | Field Intelligence Worker

This repository contains the **Worker Node** software for the Sentrix AI Distributed Intelligence Mesh. The Worker acts as the "eyes" of the system, performing edge-side face detection and streaming intelligence to the Central Hub.

---

## 🛠️ Worker Role & Architecture

The Sentrix Worker is designed to run on-site near the camera source (Edge Computing). It minimizes bandwidth by processing video locally and only sending relevant facial data to the backend for recognition.

### File Structure
```text
worker/
├── worker_agent.py      # Core logic: detection, tracking, and streaming
└── setup_worker.sh      # Automated environment and dependency installer
