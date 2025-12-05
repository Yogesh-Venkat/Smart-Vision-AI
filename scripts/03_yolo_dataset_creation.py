# ============================================================
# SMARTVISION DATASET BUILDER – FIXED VERSION
# - Streams COCO
# - Selects 25 classes
# - Builds train/val/test for YOLO
# - Uses correct image width/height for normalization
# ============================================================

import os
import json
import random
from tqdm import tqdm
from datasets import load_dataset
from PIL import Image

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

BASE_DIR = "smartvision_dataset"
IMAGES_PER_CLASS = 100        # you can increase if needed

TARGET_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "truck", "traffic light", "stop sign", "bench", "bird", "cat",
    "dog", "horse", "cow", "elephant", "bottle", "cup", "bowl",
    "pizza", "cake", "chair", "couch", "bed", "potted plant"
]

# COCO full classes (80)
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

COCO_NAME_TO_INDEX = {name: i for i, name in enumerate(COCO_CLASSES)}
SELECTED = {name: COCO_NAME_TO_INDEX[name] for name in TARGET_CLASSES}

os.makedirs(BASE_DIR, exist_ok=True)

# ------------------------------------------------------------
# STEP 1 — STREAM COCO & COLLECT IMAGES
# ------------------------------------------------------------

print("📥 Loading COCO dataset (streaming mode)...")
dataset = load_dataset("detection-datasets/coco", split="train", streaming=True)

class_images = {c: [] for c in TARGET_CLASSES}
class_count = {c: 0 for c in TARGET_CLASSES}

print("🔍 Collecting images...")
max_iterations = 100000  # safety cap

for idx, item in enumerate(dataset):
    if idx >= max_iterations:
        print(f"⚠️ Reached safety limit of {max_iterations} samples, stopping collection.")
        break

    ann = item["objects"]

    # Get image and its size (this is the reference for bbox coordinates)
    img = item["image"]
    orig_width, orig_height = img.size

    for cat_id in ann["category"]:
        # If this category is one of our target classes
        for cname, coco_id in SELECTED.items():
            if cat_id == coco_id and class_count[cname] < IMAGES_PER_CLASS:

                class_images[cname].append({
                    "image": img,                    # PIL image
                    "orig_width": orig_width,        # width used for normalization
                    "orig_height": orig_height,      # height used for normalization
                    "bboxes": ann["bbox"],           # list of bboxes
                    "cats": ann["category"],         # list of categories
                })
                class_count[cname] += 1
                break

    # Stop if all collected
    if all(count >= IMAGES_PER_CLASS for count in class_count.values()):
        break

print("🎉 Collection complete")
print("📊 Images per class:")
for cname, cnt in class_count.items():
    print(f"  {cname:15s}: {cnt}")

# ------------------------------------------------------------
# STEP 2 — CREATE FOLDERS
# ------------------------------------------------------------

DET_IMG_ROOT = os.path.join(BASE_DIR, "detection", "images")
DET_LAB_ROOT = os.path.join(BASE_DIR, "detection", "labels")

for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(DET_IMG_ROOT, split), exist_ok=True)
    os.makedirs(os.path.join(DET_LAB_ROOT, split), exist_ok=True)

# ------------------------------------------------------------
# STEP 3 — TRAIN/VAL/TEST SPLIT
# ------------------------------------------------------------

train_data = {}
val_data = {}
test_data = {}

for cname, items in class_images.items():
    random.shuffle(items)
    n = len(items)
    if n == 0:
        print(f"⚠️ No images collected for class: {cname}")
        continue

    t1 = int(0.7 * n)
    t2 = int(0.85 * n)
    train_data[cname] = items[:t1]
    val_data[cname] = items[t1:t2]
    test_data[cname] = items[t2:]

split_dict = {
    "train": train_data,
    "val": val_data,
    "test": test_data,
}

print("\n📊 Split sizes (per class):")
for cname in TARGET_CLASSES:
    tr = len(train_data.get(cname, []))
    va = len(val_data.get(cname, []))
    te = len(test_data.get(cname, []))
    print(f"  {cname:15s} -> Train={tr:3d}, Val={va:3d}, Test={te:3d}")

# ------------------------------------------------------------
# STEP 4 — SAVE DETECTION IMAGES & LABELS (FIXED NORMALIZATION)
# ------------------------------------------------------------

print("\n📁 Saving detection images + labels with correct coordinates...\n")

YOLO_NAME_TO_ID = {name: i for i, name in enumerate(TARGET_CLASSES)}

global_idx = 0
stats = {"train": 0, "val": 0, "test": 0}
label_stats = {"train": 0, "val": 0, "test": 0}
object_stats = {"train": 0, "val": 0, "test": 0}

for split, cls_dict in split_dict.items():
    print(f"\n🔹 Processing {split.upper()} ...")

    for cname, items in tqdm(cls_dict.items(), desc=f"{split} classes"):
        for item in items:

            img = item["image"]
            orig_w = item["orig_width"]
            orig_h = item["orig_height"]

            img_filename = f"image_{global_idx:06d}.jpg"
            img_path = os.path.join(DET_IMG_ROOT, split, img_filename)
            lab_path = os.path.join(DET_LAB_ROOT, split, img_filename.replace(".jpg", ".txt"))

            img.save(img_path, quality=95)
            stats[split] += 1

            bboxes = item["bboxes"]
            cats = item["cats"]

            yolo_lines = []
            obj_count = 0

            for bbox, cat in zip(bboxes, cats):
                # Only use 25 SmartVision classes
                coco_class_name = COCO_CLASSES[cat]
                if coco_class_name not in YOLO_NAME_TO_ID:
                    continue

                yolo_id = YOLO_NAME_TO_ID[coco_class_name]

                x, y, w, h = bbox  # COCO: pixel values

                # Normalize using image size
                x_center = (x + w / 2) / orig_w
                y_center = (y + h / 2) / orig_h
                w_norm = w / orig_w
                h_norm = h / orig_h

                # discard invalid
                if not (0 <= x_center <= 1 and 0 <= y_center <= 1):
                    continue
                if not (0 < w_norm <= 1 and 0 < h_norm <= 1):
                    continue

                yolo_lines.append(
                    f"{yolo_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"
                )
                obj_count += 1

            if yolo_lines:
                with open(lab_path, "w") as f:
                    f.write("\n".join(yolo_lines))
                label_stats[split] += 1
                object_stats[split] += obj_count

            global_idx += 1

print("\n🎉 All detection data saved successfully!")
for split in ["train", "val", "test"]:
    print(
        f"  {split.upper():5s} -> "
        f"images: {stats[split]:4d}, "
        f"label_files: {label_stats[split]:4d}, "
        f"objects: {object_stats[split]:5d}"
    )

# ------------------------------------------------------------
# STEP 5 — WRITE data.yaml
# ------------------------------------------------------------

print("\n📝 Writing data.yaml ...")

yaml = f"""
# SmartVision Dataset - YOLOv8 Configuration (with splits)
path: {os.path.abspath(os.path.join(BASE_DIR, "detection"))}

train: images/train
val: images/val
test: images/test

nc: {len(TARGET_CLASSES)}
names:
""" + "\n".join([f"  {i}: {name}" for i, name in enumerate(TARGET_CLASSES)])

data_yaml_path = os.path.join(BASE_DIR, "detection", "data.yaml")
os.makedirs(os.path.dirname(data_yaml_path), exist_ok=True)

with open(data_yaml_path, "w") as f:
    f.write(yaml)

print(f"✅ Created data.yaml at: {data_yaml_path}")
