# ============================================================
# SMARTVISION AI - YOLOv8 EVALUATION SCRIPT
# - Loads best.pt from training
# - Computes mAP, per-class metrics
# - Measures inference speed (FPS)
# - Saves sample prediction images
# - Saves metrics to JSON for reporting
# ============================================================

import os
import glob
import time
import json
from ultralytics import YOLO

# ------------------------------------------------------------
# 1. PATHS
# ------------------------------------------------------------

BASE_DIR  = "smartvision_dataset"
DET_DIR   = os.path.join(BASE_DIR, "detection")
DATA_YAML = os.path.join(DET_DIR, "data.yaml")

# Folder created by your train_yolo.py script
RUN_DIR       = "yolo_runs/smartvision_yolov8s"
BEST_WEIGHTS  = os.path.join(RUN_DIR, "weights", "best.pt")

# NOTE: all detection images are in detection/images (no "val" subfolder)
VAL_IMAGES_DIR = os.path.join(DET_DIR, "images")

print("📂 DATA_YAML   :", DATA_YAML)
print("📦 BEST_WEIGHTS:", BEST_WEIGHTS)
print("📁 VAL_IMAGES  :", VAL_IMAGES_DIR)

# ------------------------------------------------------------
# 2. LOAD TRAINED MODEL
# ------------------------------------------------------------

model = YOLO(BEST_WEIGHTS)
print("\n✅ Loaded trained YOLOv8 model from best.pt")

# ------------------------------------------------------------
# 3. VALIDATION METRICS (mAP, precision, recall)
# ------------------------------------------------------------

print("\n===== RUNNING VALIDATION (YOLO model.val) =====")
metrics = model.val(
    data=DATA_YAML,
    split="val",     # uses val split defined in data.yaml (here both train/val point to 'images')
    imgsz=640,
    save_json=False
)

print("\n===== YOLOv8 Validation Metrics =====")
print(f"mAP@0.5      : {metrics.box.map50:.4f}")
print(f"mAP@0.5:0.95 : {metrics.box.map:.4f}")

# metrics.box.maps is a list of per-class mAP values in the same order as names
print("\nPer-class mAP@0.5 (first 10 classes):")
for i, m in enumerate(metrics.box.maps[:10]):
    print(f"  Class {i}: {m:.4f}")

# ------------------------------------------------------------
# 4. INFERENCE SPEED (FPS) ON VALIDATION IMAGES
# ------------------------------------------------------------

print("\n===== MEASURING INFERENCE SPEED (FPS) =====")

# Collect all JPG images in detection/images
val_images = glob.glob(os.path.join(VAL_IMAGES_DIR, "*.jpg"))
val_images = sorted(val_images)

num_test_images = min(50, len(val_images))  # test on up to 50 images
test_images = val_images[:num_test_images]

print(f"Found {len(val_images)} images in {VAL_IMAGES_DIR}")
print(f"Using {len(test_images)} images for speed test.")

# Defaults in case there are no images
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
# 5. SAVE SAMPLE PREDICTIONS (BOXES + LABELS)
# ------------------------------------------------------------

print("\n===== SAVING SAMPLE PREDICTION IMAGES =====")

sample_out_project = "yolo_vis"
sample_out_name    = "samples"

if len(test_images) == 0:
    print("⚠️ No images available for sample visualization. Skipping sample predictions.")
else:
    sample_results = model.predict(
        source=test_images[:8],  # first 8 images
        imgsz=640,
        conf=0.5,
        save=True,               # save annotated images
        project=sample_out_project,
        name=sample_out_name,
        verbose=False
    )

    print(f"✅ Saved sample predictions (with boxes & labels) to: {sample_out_project}/{sample_out_name}/")

# ------------------------------------------------------------
# 6. SAVE METRICS TO JSON (FOR REPORTING)
# ------------------------------------------------------------

print("\n===== SAVING METRICS TO JSON =====")

yolo_metrics = {
    "model_name": "yolov8s_smartvision",
    "map_50": float(metrics.box.map50),
    "map_50_95": float(metrics.box.map),
    "num_val_images_for_speed_test": int(len(test_images)),
    "avg_inference_time_sec": float(time_per_image),
    "fps": float(fps),
}

os.makedirs("yolo_metrics", exist_ok=True)
metrics_json_path = os.path.join("yolo_metrics", "yolov8s_metrics.json")

with open(metrics_json_path, "w") as f:
    json.dump(yolo_metrics, f, indent=2)

print(f"✅ Saved YOLO metrics JSON to: {metrics_json_path}")
print("\n🎯 YOLOv8 evaluation complete.")
