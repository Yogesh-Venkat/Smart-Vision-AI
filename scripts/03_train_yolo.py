# ============================================================
# SMARTVISION AI - YOLOv8 TRAINING SCRIPT
# - Fine-tunes yolov8s on 25-class SmartVision detection dataset
# ============================================================

import os
import torch
from ultralytics import YOLO

# ------------------------------------------------------------
# 1. PATHS & CONFIG
# ------------------------------------------------------------

BASE_DIR  = "smartvision_dataset"
DET_DIR   = os.path.join(BASE_DIR, "detection")
DATA_YAML = os.path.join(DET_DIR, "data.yaml")

# YOLO model size:
#   - yolov8n.pt : nano
#   - yolov8s.pt : small (good tradeoff) ✅
MODEL_WEIGHTS = "yolov8s.pt"

# Auto-select device
device = "0" if torch.cuda.is_available() else "cpu"
print("🚀 Using device:", device)
print("📂 DATA_YAML:", DATA_YAML)

# ------------------------------------------------------------
# 2. LOAD BASE MODEL
# ------------------------------------------------------------

print(f"📥 Loading YOLOv8 model from: {MODEL_WEIGHTS}")
model = YOLO(MODEL_WEIGHTS)

# ------------------------------------------------------------
# 3. TRAIN
# ------------------------------------------------------------

results = model.train(
    data=DATA_YAML,
    epochs=50,
    imgsz=640,
    batch=8,                # smaller for CPU
    lr0=0.01,
    optimizer="SGD",
    device=device,
    project="yolo_runs",
    name="smartvision_yolov8s",
    pretrained=True,
    plots=True,
    verbose=True,
)

print("\n✅ YOLO training complete.")
print("📁 Run directory: yolo_runs/smartvision_yolov8s/")
print("📦 Best weights:  yolo_runs/smartvision_yolov8s/weights/best.pt")
