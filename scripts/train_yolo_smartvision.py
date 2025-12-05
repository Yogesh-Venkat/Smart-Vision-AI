#!/usr/bin/env python3
"""
train_yolo_smartvision_alltrain.py

Train YOLOv8 on ALL images (train+val+test) by creating images/train_all & labels/train_all,
then validate/test only on original val/test splits.

Features:
- Robust linking/copying with retries (hard link when possible, fallback copy).
- Manifest generation (train_all_manifest.json) with failures and post-check.
- Temporary data_all.yaml created and removed by default.
- Helpful early-failure checks so training doesn't crash with FileNotFoundError.
"""
import os
import sys
import time
import json
import glob
import shutil
import argparse
import pathlib

import torch
from ultralytics import YOLO

# ---------------------------
# Utilities
# ---------------------------

def safe_makedirs(path):
    os.makedirs(path, exist_ok=True)
    return path

def link_or_copy(src, dst, max_retries=3, allow_copy=True):
    """
    Try to create a hard link. If it fails, fall back to shutil.copy2.
    Retries on transient failures. Returns tuple (ok:bool, method:str, error:str|None).
    method in {'link', 'copy', 'exists', 'failed', 'copied_existing'}
    """
    dst_dir = os.path.dirname(dst)
    os.makedirs(dst_dir, exist_ok=True)
    if os.path.exists(dst):
        return True, "exists", None

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            os.link(src, dst)
            return True, "link", None
        except Exception as e_link:
            last_err = str(e_link)
            if not allow_copy:
                time.sleep(0.1)
                continue
            # try copying
            try:
                shutil.copy2(src, dst)
                return True, "copy", None
            except Exception as e_copy:
                last_err = f"link_err: {e_link}; copy_err: {e_copy}"
                time.sleep(0.1)
                continue
    return False, "failed", last_err

def unique_name(split, basename, used):
    """
    Create a unique filename under train_all to avoid collisions.
    Format: {split}__{basename} and if collision append index.
    """
    base = f"{split}__{basename}"
    name = base
    idx = 1
    while name in used:
        name = f"{split}__{idx}__{basename}"
        idx += 1
    used.add(name)
    return name

# ---------------------------
# Create train_all (robust)
# ---------------------------

def create_train_all(det_dir, splits=("train", "val", "test")):
    """
    Create images/train_all and labels/train_all by linking/copying
    all files from images/<split> and labels/<split>.
    Returns (out_imgs, out_labs, counters, manifest_path)
    where manifest contains details and failures.
    """
    img_root = os.path.join(det_dir, "images")
    lab_root = os.path.join(det_dir, "labels")

    out_imgs = os.path.join(det_dir, "images", "train_all")
    out_labs = os.path.join(det_dir, "labels", "train_all")
    safe_makedirs(out_imgs)
    safe_makedirs(out_labs)

    used_names = set()
    counters = {"images": 0, "labels": 0}
    manifest = {"images": [], "labels": [], "failures": [], "post_check_missing": []}

    for split in splits:
        imgs_dir = os.path.join(img_root, split)
        labs_dir = os.path.join(lab_root, split)
        if not os.path.isdir(imgs_dir) or not os.path.isdir(labs_dir):
            # skip missing split
            continue

        # collect possible image extensions
        img_files = sorted(glob.glob(os.path.join(imgs_dir, "*.jpg")) +
                           glob.glob(os.path.join(imgs_dir, "*.jpeg")) +
                           glob.glob(os.path.join(imgs_dir, "*.png")))

        for img_path in img_files:
            basename = os.path.basename(img_path)
            new_basename = unique_name(split, basename, used_names)
            dst_img = os.path.join(out_imgs, new_basename)

            ok_img, method_img, err_img = link_or_copy(img_path, dst_img, max_retries=3, allow_copy=True)
            if not ok_img:
                manifest["failures"].append({
                    "type": "image_copy_failed",
                    "src": img_path,
                    "dst": dst_img,
                    "error": err_img
                })
                continue

            counters["images"] += 1
            manifest["images"].append({"src": img_path, "dst": dst_img, "method": method_img})

            # create or link label
            orig_label_base = os.path.splitext(basename)[0]
            lab_src = os.path.join(labs_dir, orig_label_base + ".txt")
            dst_lab = os.path.join(out_labs, os.path.splitext(new_basename)[0] + ".txt")

            if os.path.exists(lab_src):
                ok_lab, method_lab, err_lab = link_or_copy(lab_src, dst_lab, max_retries=3, allow_copy=True)
                if not ok_lab:
                    manifest["failures"].append({
                        "type": "label_copy_failed",
                        "src": lab_src,
                        "dst": dst_lab,
                        "error": err_lab
                    })
                else:
                    counters["labels"] += 1
                    manifest["labels"].append({"src": lab_src, "dst": dst_lab, "method": method_lab})
            else:
                # Create empty label file so YOLO treats it as background (explicit)
                try:
                    open(dst_lab, "w").close()
                    counters["labels"] += 1
                    manifest["labels"].append({"src": None, "dst": dst_lab, "method": "empty_created"})
                except Exception as e:
                    manifest["failures"].append({
                        "type": "label_create_failed",
                        "src": None,
                        "dst": dst_lab,
                        "error": str(e)
                    })

    # Final verification: every label should have at least one matching image with same base (any ext)
    out_img_bases = set(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(out_imgs, "*")))
    missing_pairs = []
    for lab in glob.glob(os.path.join(out_labs, "*.txt")):
        base = os.path.splitext(os.path.basename(lab))[0]
        if base not in out_img_bases:
            # Labels that don't have corresponding image
            missing_pairs.append(base)

    manifest["post_check_missing"] = missing_pairs

    manifest_path = os.path.join(det_dir, "train_all_manifest.json")
    try:
        with open(manifest_path, "w") as f:
            json.dump({"counters": counters, "manifest": manifest}, f, indent=2)
    except Exception as e:
        # fallback printing
        print("⚠️ Could not write manifest:", e)

    return out_imgs, out_labs, counters, manifest_path

# ---------------------------
# Write temporary data YAML
# ---------------------------

def write_temp_data_yaml(det_dir, data_yaml_path, train_rel="images/train_all", val_rel="images/val", test_rel="images/test", names_list=None):
    """
    Writes a temporary data YAML for training.
    """
    if names_list is None:
        orig = os.path.join(det_dir, "data.yaml")
        if os.path.exists(orig):
            try:
                import yaml
                with open(orig, "r") as f:
                    d = yaml.safe_load(f)
                    names_list = d.get("names") or d.get("names", None)
                    if isinstance(names_list, dict):
                        # convert mapping to ordered list by int key
                        sorted_items = sorted(names_list.items(), key=lambda x: int(x[0]))
                        names_list = [v for k, v in sorted_items]
            except Exception:
                names_list = None
    if names_list is None:
        # safe default if reading fails
        names_list = [f"class{i}" for i in range(25)]

    abs_path = os.path.abspath(det_dir)
    yaml_str = f"path: {abs_path}\n\ntrain: {train_rel}\nval: {val_rel}\ntest: {test_rel}\n\nnc: {len(names_list)}\nnames:\n"
    for i, n in enumerate(names_list):
        yaml_str += f"  {i}: {n}\n"

    with open(data_yaml_path, "w") as f:
        f.write(yaml_str)

    return data_yaml_path

# ---------------------------
# Main flow
# ---------------------------

def main(
    base_dir="smartvision_dataset",
    run_project="yolo_runs",
    run_name="smartvision_yolov8s_alltrain",
    model_weights="yolov8s.pt",
    quick_test=False,
    epochs_full=50,
    batch=8,
    keep_temp=False,
):
    DET_DIR = os.path.join(base_dir, "detection")
    DATA_YAML_ORIG = os.path.join(DEТ_DIR := DET_DIR, "data.yaml")  # preserve original var name for readability

    # safety checks
    if not os.path.exists(DET_DIR):
        raise FileNotFoundError(f"Detection directory not found: {DET_DIR}")
    if not os.path.exists(DATA_YAML_ORIG):
        raise FileNotFoundError(f"Original data.yaml not found: {DATA_YAML_ORIG}")

    # show basic dataset split counts
    for split in ["train", "val", "test"]:
        img_dir = os.path.join(DET_DIR, "images", split)
        lab_dir = os.path.join(DET_DIR, "labels", split)
        num_imgs = len(glob.glob(os.path.join(img_dir, "*.jpg"))) + len(glob.glob(os.path.join(img_dir, "*.png"))) + len(glob.glob(os.path.join(img_dir, "*.jpeg")))
        num_labs = len(glob.glob(os.path.join(lab_dir, "*.txt")))
        print(f"✅ {split.upper():5s}: {num_imgs} images, {num_labs} label files")

    # Read class names from original data.yaml (if possible)
    try:
        import yaml
        with open(DATA_YAML_ORIG, "r") as f:
            orig_yaml = yaml.safe_load(f)
            names = orig_yaml.get("names")
            if isinstance(names, dict):
                sorted_items = sorted(names.items(), key=lambda x: int(x[0]))
                names_list = [v for k, v in sorted_items]
            else:
                names_list = names
    except Exception:
        names_list = None

    print("🧩 Creating combined train_all (train+val+test)...")
    imgs_train_all, labs_train_all, counters, manifest_path = create_train_all(DET_DIR, splits=("train", "val", "test"))
    print(f"  ➜ train_all images: {counters['images']}, labels: {counters['labels']}")
    print(f"  ➜ manifest written to: {manifest_path}")

    # read manifest and abort early on issues
    try:
        with open(manifest_path, "r") as f:
            manifest_data = json.load(f)
            manifest = manifest_data.get("manifest", {})
    except Exception:
        manifest = {}

    failures = manifest.get("failures", [])
    post_missing = manifest.get("post_check_missing", [])

    if failures:
        print("\n❌ Errors found while creating train_all (see manifest). Aborting training.")
        print(f"  Failures count: {len(failures)}. Sample:")
        for f in failures[:10]:
            print("   -", f)
        print(f"\nInspect and fix ({manifest_path}) then re-run.")
        return

    if post_missing:
        print("\n❌ Post-creation check failed: some labels don't have matching images.")
        print(f"  Missing pairs count: {len(post_missing)}. Sample: {post_missing[:20]}")
        print(f"Please inspect the labels/images under {labs_train_all} and {imgs_train_all}. Aborting.")
        return

    # write temporary data yaml
    temp_data_yaml = os.path.join(DET_DIR, "data_all.yaml")
    write_temp_data_yaml(DET_DIR, temp_data_yaml, train_rel="images/train_all", val_rel="images/val", test_rel="images/test", names_list=names_list)
    print(f"📝 Temporary data yaml created at: {temp_data_yaml}")

    # determine epochs
    EPOCHS = 1 if quick_test else epochs_full
    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Device: {device}; QUICK_TEST: {quick_test}; EPOCHS: {EPOCHS}")

    # load base model
    print(f"\n📥 Loading YOLOv8 base model from: {model_weights}")
    model = YOLO(model_weights)

    # Train on train_all
    run_name_final = run_name
    print("\n===== STARTING TRAINING on ALL IMAGES (train_all) =====")
    results = model.train(
        data=temp_data_yaml,
        epochs=EPOCHS,
        imgsz=640,
        batch=batch,
        lr0=0.01,
        optimizer="SGD",
        device=device,
        project=run_project,
        name=run_name_final,
        pretrained=True,
        plots=True,
        verbose=True,
    )
    print("\n✅ Training finished.")

    run_dir = os.path.join(run_project, run_name_final)
    best_weights = os.path.join(run_dir, "weights", "best.pt")
    if not os.path.exists(best_weights):
        print("⚠️ best.pt not found after training — attempting to use last.pt")
        last = os.path.join(run_dir, "weights", "last.pt")
        if os.path.exists(last):
            best_weights = last
        else:
            raise FileNotFoundError("No trained weights found (best.pt or last.pt).")

    # Load trained model
    print(f"\n📥 Loading trained model from: {best_weights}")
    model = YOLO(best_weights)
    print("✅ Model loaded. Running val/test on original val & test splits...")

    # Validation (val split)
    print("\n===== VALIDATION (original val split) =====")
    metrics_val = model.val(data=DATA_YAML_ORIG, split="val", imgsz=640, save_json=False)
    print(f"[VAL] mAP@0.5 : {metrics_val.box.map50:.4f}   mAP@0.5:0.95 : {metrics_val.box.map:.4f}")

    # Test (test split)
    print("\n===== TEST (original test split) =====")
    metrics_test = model.val(data=DATA_YAML_ORIG, split="test", imgsz=640, save_json=False)
    print(f"[TEST] mAP@0.5 : {metrics_test.box.map50:.4f}   mAP@0.5:0.95 : {metrics_test.box.map:.4f}")

    # FPS test on val images (small subset)
    val_images_dir = os.path.join(DET_DIR, "images", "val")
    val_images = sorted(glob.glob(os.path.join(val_images_dir, "*.jpg")) +
                        glob.glob(os.path.join(val_images_dir, "*.png")) +
                        glob.glob(os.path.join(val_images_dir, "*.jpeg")))
    n_proc = min(50, len(val_images))
    test_imgs = val_images[:n_proc]
    if test_imgs:
        print(f"\n🏃 Running speed test on {len(test_imgs)} val images...")
        start = time.perf_counter()
        _ = model.predict(source=test_imgs, imgsz=640, conf=0.5, verbose=False)
        duration = time.perf_counter() - start
        print(f"  Total {duration:.2f}s -> {duration/len(test_imgs)*1000:.2f} ms/img -> {1.0/(duration/len(test_imgs)):.2f} FPS")
    else:
        print("⚠️ No val images found for speed test.")

    # Save metrics to JSON
    metrics_out = {
        "train_all_counters": counters,
        "val_map50": float(metrics_val.box.map50),
        "test_map50": float(metrics_test.box.map50),
        "val_map50_95": float(metrics_val.box.map),
        "test_map50_95": float(metrics_test.box.map),
        "run_dir": run_dir,
        "best_weights": best_weights,
    }
    os.makedirs("yolo_metrics", exist_ok=True)
    json_path = os.path.join("yolo_metrics", f"yolov8s_metrics_alltrain.json")
    with open(json_path, "w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"\n💾 Saved metrics to: {json_path}")

    # Cleanup if requested
    if not keep_temp:
        try:
            print("\n🧹 Cleaning temporary train_all files and temp data yaml...")
            shutil.rmtree(os.path.join(DET_DIR, "images", "train_all"), ignore_errors=True)
            shutil.rmtree(os.path.join(DET_DIR, "labels", "train_all"), ignore_errors=True)
            if os.path.exists(temp_data_yaml):
                os.remove(temp_data_yaml)
            if os.path.exists(manifest_path):
                os.remove(manifest_path)
            print("✅ Temp cleanup done.")
        except Exception as e:
            print("⚠️ Cleanup error:", e)
    else:
        print(f"\nℹ️ Kept temp train_all and temp yaml as requested. Path: {os.path.join(DET_DIR, 'images', 'train_all')}")

    print("\n🎯 ALL DONE.")

# ---------------------------
# CLI
# ---------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 on ALL images (train+val+test) then validate/test on original splits.")
    parser.add_argument("--dataset-dir", "-d", default="smartvision_dataset", help="Base dataset directory (default: smartvision_dataset)")
    parser.add_argument("--model", "-m", default="yolov8s.pt", help="Base yolov8 weights (default: yolov8s.pt)")
    parser.add_argument("--quick", action="store_true", help="Quick test (1 epoch, small speed test)")
    parser.add_argument("--epochs", type=int, default=50, help="Full epochs when not quick")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--no-clean", dest="keep_temp", action="store_true", help="Do NOT remove temp train_all folder and temp yaml after run")
    parser.add_argument("--project", default="yolo_runs", help="Ultralytics runs project folder")
    parser.add_argument("--name", default="smartvision_yolov8s_alltrain", help="Run name")
    args = parser.parse_args()

    main(
        base_dir=args.dataset_dir,
        run_project=args.project,
        run_name=args.name,
        model_weights=args.model,
        quick_test=args.quick,
        epochs_full=args.epochs,
        batch=args.batch,
        keep_temp=args.keep_temp,
    )
