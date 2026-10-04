# uav-osd-telemetry-parsing
UAV OSD Telemetry Parsing &amp; Video Cleaning Pipeline using YOLOv8, OpenCV Inpainting, and Qwen VLM (EPS 2026).

# 🛸 UAV OSD Telemetry Parsing & Video Cleaning System

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green.svg)](https://github.com/ultralytics/ultralytics)
[![Qwen2-VL](https://img.shields.io/badge/VLM-Qwen2--VL-orange.svg)](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end computer vision and VLM pipeline designed for **Engineering Project Sprint (EPS 2026)**. The system detects On-Screen Display (OSD) telemetry elements on raw UAV video feeds, extracts structured JSON telemetry data, and restores clean video feeds via OpenCV Inpainting.

---

## 🌟 Key Features
- **Object Detection:** Fine-tuned `YOLOv8n` trained on a custom Roboflow dataset for high-speed OSD element localization.
- **Structured Extraction:** Multi-modal `Qwen2-VL-2B-Instruct` integration mapping visible flight parameters to **25 fixed JSON schema keys**.
- **Video Restoration:** Automated frame-by-frame OSD removal and background texture inpainting (`cv2.inpaint`).
- **Memory Optimized:** Safe execution under free-tier GPU constraints (manual VRAM garbage collection, dynamic resizing, `torch.no_grad()`).

---

## 🏗️ Architecture Pipeline

1. **YOLOv8 Detection:** Locates bounding boxes for UI elements (flight mode, battery voltage, coordinates, RSSI, etc.).
2. **Crop & Frame Assembly:** Extracts high-contrast OSD crops and constructs a coordinate-annotated frame.
3. **VLM Inference:** Passes crops and global context to Qwen2-VL to parse numbers and text into a validated JSON object.
4. **OpenCV Inpainting:** Generates dynamic dynamic binary masks based on YOLO predictions and fills masked regions using Telea inpainting.

---

## 🚀 Quick Start

### 1. Installation
```bash
git clone [https://github.com/3olena3/uav-osd-telemetry-parsing.git](https://github.com/3olena3/uav-osd-telemetry-parsing.git)
cd uav-osd-telemetry-parsing
pip install -r requirements.txt
