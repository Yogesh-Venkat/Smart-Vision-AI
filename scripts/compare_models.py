"""
SMARTVISION AI - Step 2.5: Model Comparison & Selection

This script:
- Loads metrics.json and confusion_matrix.npy for all models.
- Compares accuracy, precision, recall, F1, top-5 accuracy, speed, and model size.
- Generates bar plots for metrics.
- Generates confusion matrix heatmaps per model.
- Selects the best model using an accuracy–speed tradeoff rule.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 0. CONFIG – resolve paths relative to this file
# ------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(SCRIPT_DIR)           # one level up from scripts/
METRICS_DIR = os.path.join(ROOT_DIR, "smartvision_metrics")
PLOTS_DIR   = os.path.join(METRICS_DIR, "comparison_plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

print(f"[INFO] Using METRICS_DIR = {METRICS_DIR}")
print(f"[INFO] Existing subfolders in METRICS_DIR: {os.listdir(METRICS_DIR) if os.path.exists(METRICS_DIR) else 'NOT FOUND'}")

# Map "pretty" model names to their metrics subdirectories
MODEL_PATHS = {
    "VGG16"    : "vgg16_v2_stage2",
    "ResNet50" : "resnet50_v2_stage2",
    "MobileNetV2"           : "mobilenetv2_v2",
    "efficientnetb0"           : "efficientnetb0",
    # Optional: add more models here, e.g.:
    # "ResNet50 v2 (Stage 1)"  : "resnet50_v2_stage1",
}

# Class names (COCO-style 25 classes)
CLASS_NAMES = [
    "airplane", "bed", "bench", "bicycle", "bird",
    "bottle", "bowl", "bus", "cake", "car",
    "cat", "chair", "couch", "cow", "cup",
    "dog", "elephant", "horse", "motorcycle", "person",
    "pizza", "potted plant", "stop sign", "traffic light", "truck",
]


# ------------------------------------------------------------
# 1. LOAD METRICS & CONFUSION MATRICES
# ------------------------------------------------------------

def load_model_results():
    model_metrics = {}
    model_cms = {}

    for nice_name, folder_name in MODEL_PATHS.items():
        metrics_path = os.path.join(METRICS_DIR, folder_name, "metrics.json")
        cm_path      = os.path.join(METRICS_DIR, folder_name, "confusion_matrix.npy")

        print(f"[DEBUG] Looking for {nice_name} metrics at: {metrics_path}")
        print(f"[DEBUG] Looking for {nice_name} CM at     : {cm_path}")

        if not os.path.exists(metrics_path):
            print(f"[WARN] Skipping {nice_name}: missing {metrics_path}")
            continue
        if not os.path.exists(cm_path):
            print(f"[WARN] Skipping {nice_name}: missing {cm_path}")
            continue

        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        cm = np.load(cm_path)

        model_metrics[nice_name] = metrics
        model_cms[nice_name] = cm
        print(f"[INFO] Loaded metrics & CM for {nice_name}")

    return model_metrics, model_cms


# ------------------------------------------------------------
# 2. PLOTTING HELPERS
# ------------------------------------------------------------

def plot_bar_metric(model_metrics, metric_key, ylabel, filename, higher_is_better=True):
    names = list(model_metrics.keys())
    values = [model_metrics[n][metric_key] for n in names]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(names, values)
    plt.ylabel(ylabel)
    plt.xticks(rotation=20, ha="right")

    for bar, val in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    title_prefix = "Higher is better" if higher_is_better else "Lower is better"
    plt.title(f"{metric_key} comparison ({title_prefix})")
    plt.tight_layout()

    out_path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[PLOT] Saved {metric_key} comparison to {out_path}")


def plot_confusion_matrix(cm, classes, title, filename, normalize=True):
    if normalize:
        cm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-12)

    plt.figure(figsize=(6, 5))
    im = plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar(im, fraction=0.046, pad=0.04)

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    # annotate diagonal only to reduce clutter
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i == j:
                plt.text(
                    j,
                    i,
                    f"{cm[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > 0.5 else "black",
                    fontsize=6,
                )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()

    out_path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[PLOT] Saved confusion matrix to {out_path}")


# ------------------------------------------------------------
# 3. MODEL SELECTION (ACCURACY–SPEED TRADEOFF)
# ------------------------------------------------------------

def pick_best_model(model_metrics):
    """
    Rule:
      1. Prefer highest accuracy.
      2. If two models are within 0.5% accuracy, prefer higher images_per_second.
    """
    best_name = None
    best_acc = -1.0
    best_speed = -1.0

    for name, m in model_metrics.items():
        acc = m["accuracy"]
        speed = m.get("images_per_second", 0.0)

        if acc > best_acc + 0.005:  # clearly better
            best_name = name
            best_acc = acc
            best_speed = speed
        elif abs(acc - best_acc) <= 0.005:  # within 0.5%, use speed as tie-breaker
            if speed > best_speed:
                best_name = name
                best_acc = acc
                best_speed = speed

    return best_name, best_acc, best_speed


# ------------------------------------------------------------
# 4. MAIN
# ------------------------------------------------------------

def main():
    model_metrics, model_cms = load_model_results()

    if not model_metrics:
        print("[ERROR] No models found with valid metrics. Check METRICS_DIR and MODEL_PATHS.")
        return

    print("\n===== MODEL METRICS SUMMARY =====")
    print(
        f"{'Model':30s}  {'Acc':>6s}  {'Prec':>6s}  {'Rec':>6s}  {'F1':>6s}  {'Top5':>6s}  {'img/s':>7s}  {'Size(MB)':>8s}"
    )
    for name, m in model_metrics.items():
        print(
            f"{name:30s}  "
            f"{m['accuracy']:.3f}  "
            f"{m['precision_weighted']:.3f}  "
            f"{m['recall_weighted']:.3f}  "
            f"{m['f1_weighted']:.3f}  "
            f"{m['top5_accuracy']:.3f}  "
            f"{m['images_per_second']:.2f}  "
            f"{m['model_size_mb']:.1f}"
        )

    # ---- Comparison plots ----
    plot_bar_metric(model_metrics, "accuracy", "Accuracy", "accuracy_comparison.png")
    plot_bar_metric(
        model_metrics, "f1_weighted", "Weighted F1-score", "f1_comparison.png"
    )
    plot_bar_metric(
        model_metrics, "top5_accuracy", "Top-5 Accuracy", "top5_comparison.png"
    )
    plot_bar_metric(
        model_metrics,
        "images_per_second",
        "Images per second",
        "speed_comparison.png",
    )
    plot_bar_metric(
        model_metrics,
        "model_size_mb",
        "Model size (MB)",
        "size_comparison.png",
        higher_is_better=False,
    )

    # ---- Confusion matrices ----
    print("\n===== SAVING CONFUSION MATRICES =====")
    for name, cm in model_cms.items():
        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")
        filename = f"{safe_name}_cm.png"
        plot_confusion_matrix(
            cm,
            classes=CLASS_NAMES,
            title=f"Confusion Matrix - {name}",
            filename=filename,
            normalize=True,
        )

    # ---- Best model ----
    best_name, best_acc, best_speed = pick_best_model(model_metrics)

    print("\n===== BEST MODEL SELECTION =====")
    print(f"Selected best model: {best_name}")
    print(f"  Test Accuracy      : {best_acc:.4f}")
    print(f"  Images per second  : {best_speed:.2f}")
    print("\nRationale:")
    print("- Highest accuracy is preferred.")
    print("- If models are within 0.5% accuracy, the faster model (higher img/s) is chosen.")

    print("\nSuggested text for report:")
    print(
        f"\"Among all evaluated architectures, {best_name} achieved the best accuracy–speed "
        f"tradeoff on the SmartVision AI test set, with a top-1 accuracy of {best_acc:.3f} "
        f"and an inference throughput of {best_speed:.2f} images per second on the "
        f"evaluation hardware.\""
    )


if __name__ == "__main__":
    main()
