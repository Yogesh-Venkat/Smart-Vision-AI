# ============================================================
# SMARTVISION AI - YOLOv8 TRAIN + EVAL SCRIPT
# - Uses separate train / val / test splits
# - QUICK_TEST flag lets you sanity-check the whole pipeline
#   with just 1 epoch before doing full training
# ============================================================

import os
import glob
import time
import json
import torch
from ultralytics import YOLO

# ------------------------------------------------------------
# 0. CONFIG: QUICK TEST OR FULL TRAINING?
# ------------------------------------------------------------
# First run with QUICK_TEST = True (1 epoch, debug run).
# If everything runs end-to-end without errors, set it to False.
QUICK_TEST = True   # <<< CHANGE TO False FOR FULL TRAINING

FULL_EPOCHS = 50
DEBUG_EPOCHS = 1

EPOCHS = DEBUG_EPOCHS if QUICK_TEST else FULL_EPOCHS
RUN_NAME = "smartvision_yolov8s_debug" if QUICK_TEST else "smartvision_yolov8s"

print("⚙️  QUICK_TEST :", QUICK_TEST)
print("⚙️  EPOCHS     :", EPOCHS)
print("⚙️  RUN_NAME   :", RUN_NAME)

# ------------------------------------------------------------
# 1. PATHS & CONFIG
# ------------------------------------------------------------

BASE_DIR  = "smartvision_dataset"
DET_DIR   = os.path.join(BASE_DIR, "detection")
DATA_YAML = os.path.join(DET_DIR, "data.yaml")

# Expected folder structure:
# smartvision_dataset/detection/
#   data.yaml
#   images/train, images/val, images/test
#   labels/train, labels/val, labels/test

RUN_PROJECT   = "yolo_runs"
MODEL_WEIGHTS = "yolov8s.pt"   # base checkpoint to fine-tune

VAL_IMAGES_DIR = os.path.join(DET_DIR, "images", "val")

# Auto-select device
device = "0" if torch.cuda.is_available() else "cpu"
print("🚀 Using device:", device)
print("📂 DATA_YAML   :", DATA_YAML)

# Basic path checks (fail fast if something is wrong)
if not os.path.exists(DATA_YAML):
    raise FileNotFoundError(f"data.yaml not found at: {DATA_YAML}")

for split in ["train", "val", "test"]:
    img_dir = os.path.join(DET_DIR, "images", split)
    lab_dir = os.path.join(DET_DIR, "labels", split)
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"Images directory missing: {img_dir}")
    if not os.path.isdir(lab_dir):
        raise FileNotFoundError(f"Labels directory missing: {lab_dir}")
    if len(glob.glob(os.path.join(img_dir, "*.jpg"))) == 0:
        print(f"⚠️ Warning: No .jpg images found in {img_dir}")

# ------------------------------------------------------------
# 2. LOAD BASE MODEL
# ------------------------------------------------------------

print(f"\n📥 Loading YOLOv8 base model from: {MODEL_WEIGHTS}")
model = YOLO(MODEL_WEIGHTS)

# ------------------------------------------------------------
# 3. TRAIN
# ------------------------------------------------------------

print("\n===== STARTING TRAINING =====")
print("(This is a QUICK TEST run)" if QUICK_TEST else "(Full training run)")

results = model.train(
    data=DATA_YAML,
    epochs=EPOCHS,
    imgsz=640,
    batch=8,                # adjust if more GPU RAM
    lr0=0.01,
    optimizer="SGD",
    device=device,
    project=RUN_PROJECT,
    name=RUN_NAME,
    pretrained=True,
    plots=True,
    verbose=True,
)

print("\n✅ YOLO training complete.")
RUN_DIR      = os.path.join(RUN_PROJECT, RUN_NAME)
BEST_WEIGHTS = os.path.join(RUN_DIR, "weights", "best.pt")
print("📁 Run directory:", RUN_DIR)
print("📦 Best weights :", BEST_WEIGHTS)

if not os.path.exists(BEST_WEIGHTS):
    raise FileNotFoundError(f"best.pt not found at: {BEST_WEIGHTS}")

# ------------------------------------------------------------
# 4. LOAD TRAINED MODEL (best.pt)
# ------------------------------------------------------------

print("\n📥 Loading trained model from best.pt")
model = YOLO(BEST_WEIGHTS)
print("✅ Loaded trained YOLOv8 model.")
print("📜 Class mapping (model.names):")
print(model.names)

# ------------------------------------------------------------
# 5. VALIDATION & TEST METRICS
# ------------------------------------------------------------

print("\n===== RUNNING VALIDATION (val split) =====")
metrics_val = model.val(
    data=DATA_YAML,
    split="val",     # images/val + labels/val
    imgsz=640,
    save_json=False
)

print("\n===== YOLOv8 Validation Metrics =====")
print(f"[VAL] mAP@0.5      : {metrics_val.box.map50:.4f}")
print(f"[VAL] mAP@0.5:0.95 : {metrics_val.box.map:.4f}")

print("\nPer-class mAP@0.5 on VAL (first 10 classes):")
for i, m in enumerate(metrics_val.box.maps[:10]):
    print(f"  Class {i}: {m:.4f}")

print("\n===== RUNNING TEST EVALUATION (test split) =====")
metrics_test = model.val(
    data=DATA_YAML,
    split="test",    # images/test + labels/test
    imgsz=640,
    save_json=False
)

print("\n===== YOLOv8 Test Metrics =====")
print(f"[TEST] mAP@0.5      : {metrics_test.box.map50:.4f}")
print(f"[TEST] mAP@0.5:0.95 : {metrics_test.box.map:.4f}")

# ------------------------------------------------------------
# 6. INFERENCE SPEED (FPS) ON VAL IMAGES
# ------------------------------------------------------------

print("\n===== MEASURING INFERENCE SPEED (FPS) ON VAL IMAGES =====")

val_images = glob.glob(os.path.join(VAL_IMAGES_DIR, "*.jpg"))
val_images = sorted(val_images)

num_test_images = min(10 if QUICK_TEST else 50, len(val_images))
test_images = val_images[:num_test_images]

print(f"Found {len(val_images)} images in {VAL_IMAGES_DIR}")
print(f"Using {len(test_images)} images for speed test.")

time_per_image = 0.0
fps = 0.0

if len(test_images) == 0:
    print("⚠️ No images found for FPS test. Skipping speed measurement.")
else:
    start = time.perf_counter()
    _ = model.predict(
        source=test_images,
        imgsz=640,
        conf=0.5,
        verbose=False
    )
    end = time.perf_counter()

    total_time = end - start
    time_per_image = total_time / len(test_images)
    fps = 1.0 / time_per_image

    print(f"Total time       : {total_time:.2f} sec for {len(test_images)} images")
    print(f"Avg time / image : {time_per_image*1000:.2f} ms")
    print(f"Approx FPS       : {fps:.2f} images/sec")

# ------------------------------------------------------------
# 7. SAVE SAMPLE PREDICTION IMAGES (FROM VAL)
# ------------------------------------------------------------

print("\n===== SAVING SAMPLE PREDICTION IMAGES (VAL) =====")

sample_out_project = "yolo_vis"
sample_out_name    = "samples_debug" if QUICK_TEST else "samples"

if len(test_images) == 0:
    print("⚠️ No val images available for sample visualization. Skipping sample predictions.")
else:
    _ = model.predict(
        source=test_images[:4 if QUICK_TEST else 8],
        imgsz=640,
        conf=0.5,
        save=True,
        project=sample_out_project,
        name=sample_out_name,
        verbose=False,
    )
    print(f"✅ Saved sample predictions (with boxes & labels) to: {sample_out_project}/{sample_out_name}/")

# ------------------------------------------------------------
# 8. SAVE METRICS TO JSON
# ------------------------------------------------------------

print("\n===== SAVING METRICS TO JSON =====")

os.makedirs("yolo_metrics", exist_ok=True)
metrics_json_path = os.path.join("yolo_metrics", "yolov8s_metrics_debug.json" if QUICK_TEST else "yolov8s_metrics.json")

yolo_metrics = {
    "model_name": "yolov8s_smartvision",
    "quick_test": QUICK_TEST,
    "epochs": EPOCHS,
    "run_dir": RUN_DIR,
    "best_weights": BEST_WEIGHTS,
    "val_map_50": float(metrics_val.box.map50),
    "val_map_50_95": float(metrics_val.box.map),
    "test_map_50": float(metrics_test.box.map50),
    "test_map_50_95": float(metrics_test.box.map),
    "num_val_images_for_speed_test": int(len(test_images)),
    "avg_inference_time_sec": float(time_per_image),
    "fps": float(fps),
}

with open(metrics_json_path, "w") as f:
    json.dump(yolo_metrics, f, indent=2)

print(f"✅ Saved YOLO metrics JSON to: {metrics_json_path}")
print("\n🎯 YOLOv8 training + evaluation script finished.")
