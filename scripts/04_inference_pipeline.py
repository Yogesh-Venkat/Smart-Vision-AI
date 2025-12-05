# ============================================================
# SMARTVISION AI - PHASE 4
# Model Integration & Inference Pipeline (YOLOv8 + ResNet50 v2)
# ============================================================

import os
import time
from typing import List, Dict, Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from ultralytics import YOLO

print("TensorFlow version:", tf.__version__)

# ------------------------------------------------------------
# 1. CONFIGURATION
# ------------------------------------------------------------

# Dataset & models
BASE_DIR = "smartvision_dataset"
CLASS_DIR = os.path.join(BASE_DIR, "classification")
TRAIN_DIR = os.path.join(CLASS_DIR, "train")

# YOLO & classifier weights
YOLO_WEIGHTS = "yolo_runs/smartvision_yolov8s6 - Copy/weights/best.pt"  # adjust if needed
CLASSIFIER_WEIGHTS_PATH = os.path.join(
    "saved_models", "resnet50_v2_stage2_best.weights.h5"
)

IMG_SIZE = (224, 224)
NUM_CLASSES = 25

# Where to save annotated outputs
OUTPUT_DIR = "inference_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------
# 2. CLASS NAMES (MUST MATCH TRAINING ORDER)
#    From your training logs:
#    ['airplane', 'bed', 'bench', 'bicycle', 'bird', 'bottle', 'bowl',
#     'bus', 'cake', 'car', 'cat', 'chair', 'couch', 'cow', 'cup', 'dog',
#     'elephant', 'horse', 'motorcycle', 'person', 'pizza', 'potted plant',
#     'stop sign', 'traffic light', 'truck']
# ------------------------------------------------------------

CLASS_NAMES = [
    "airplane", "bed", "bench", "bicycle", "bird", "bottle", "bowl",
    "bus", "cake", "car", "cat", "chair", "couch", "cow", "cup", "dog",
    "elephant", "horse", "motorcycle", "person", "pizza", "potted plant",
    "stop sign", "traffic light", "truck"
]

assert len(CLASS_NAMES) == NUM_CLASSES, "CLASS_NAMES length must be 25"

# ------------------------------------------------------------
# 3. DATA AUGMENTATION (same as training, but no effect in inference)
# ------------------------------------------------------------

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.04),       # ~±15°
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.15),
        layers.Lambda(
            lambda x: tf.image.random_brightness(x, max_delta=0.15)
        ),
        layers.Lambda(
            lambda x: tf.image.random_saturation(x, 0.85, 1.15)
        ),
    ],
    name="data_augmentation",
)

# ------------------------------------------------------------
# 4. BUILD RESNET50 v2 CLASSIFIER (MATCHES TRAINING ARCHITECTURE)
# ------------------------------------------------------------

def build_resnet50_model_v2():
    """
    Build the ResNet50 v2 classifier with the SAME architecture as in training.
    (data_augmentation + Lambda(resnet50.preprocess_input) + ResNet50 backbone + head)
    """
    inputs = keras.Input(shape=(*IMG_SIZE, 3), name="input_layer")

    # Augmentation (no randomness in inference mode, Keras handles that)
    x = data_augmentation(inputs)

    # ResNet50-specific preprocessing
    x = layers.Lambda(
        keras.applications.resnet50.preprocess_input,
        name="resnet50_preprocess",
    )(x)

    # Pretrained ResNet50 backbone
    base_model = keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMG_SIZE, 3),
    )

    x = base_model(x)

    # Custom classification head (same as training file)
    x = layers.GlobalAveragePooling2D(name="global_average_pooling2d")(x)

    x = layers.BatchNormalization(name="head_batchnorm")(x)
    x = layers.Dropout(0.4, name="head_dropout")(x)

    x = layers.Dense(
        256,
        activation="relu",
        name="head_dense",
    )(x)

    x = layers.BatchNormalization(name="head_batchnorm_2")(x)
    x = layers.Dropout(0.5, name="head_dropout_2")(x)

    outputs = layers.Dense(
        NUM_CLASSES,
        activation="softmax",
        name="predictions",
    )(x)

    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="ResNet50_smartvision_v2_infer",
    )

    return model, base_model


def load_classifier(weights_path: str):
    """
    Build the ResNet50 v2 model and load fine-tuned weights from
    resnet50_v2_stage2_best.weights.h5
    """
    if not os.path.exists(weights_path):
        print(f"⚠️ Classifier weights not found at: {weights_path}")
        print("   Using ImageNet-pretrained ResNet50 base + randomly initialized head.")
        model, _ = build_resnet50_model_v2()
        return model

    model, _ = build_resnet50_model_v2()
    model.load_weights(weights_path)
    print(f"✅ Loaded classifier weights from: {weights_path}")
    return model

# ------------------------------------------------------------
# 5. LOAD YOLO MODEL
# ------------------------------------------------------------

def load_yolo_model(weights_path: str = YOLO_WEIGHTS) -> YOLO:
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"YOLO weights not found at: {weights_path}")
    model = YOLO(weights_path)
    print(f"✅ Loaded YOLOv8 model from: {weights_path}")
    return model

# ------------------------------------------------------------
# 6. HELPER: PREPROCESS CROP FOR CLASSIFIER
# ------------------------------------------------------------

def preprocess_crop_for_classifier(crop_img: Image.Image,
                                   img_size=IMG_SIZE) -> np.ndarray:
    """
    Resize PIL image crop to 224x224 and prepare as batch tensor.
    NOTE: No manual rescaling here; model already has preprocess_input inside.
    """
    crop_resized = crop_img.resize(img_size, Image.BILINEAR)
    arr = np.array(crop_resized, dtype=np.float32)  # shape (H,W,3)
    arr = np.expand_dims(arr, axis=0)  # (1,H,W,3)
    return arr

# ------------------------------------------------------------
# 7. DRAWING UTIL: BOUNDING BOXES + LABELS (Pillow 10+ SAFE)
# ------------------------------------------------------------

def draw_boxes_with_labels(
    pil_img: Image.Image,
    detections: List[Dict[str, Any]],
    font_path: str = None
) -> Image.Image:
    """
    Draw bounding boxes & labels on an image.

    detections: list of dicts with keys:
      - x1, y1, x2, y2
      - label (str)
      - conf_yolo (float)
      - cls_label (optional, str)
      - cls_conf (optional, float)
    """
    draw = ImageDraw.Draw(pil_img)

    # Try to load a TTF font, fallback to default
    if font_path and os.path.exists(font_path):
        font = ImageFont.truetype(font_path, 16)
    else:
        font = ImageFont.load_default()

    for det in detections:
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        yolo_label = det["label"]
        conf_yolo = det["conf_yolo"]
        cls_label = det.get("cls_label")
        cls_conf = det.get("cls_conf")

        # Text to display
        if cls_label is not None:
            text = f"{yolo_label} {conf_yolo:.2f} | CLS: {cls_label} {cls_conf:.2f}"
        else:
            text = f"{yolo_label} {conf_yolo:.2f}"

        # Box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

        # Compute text size safely (Pillow 10+)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        # Text background (clamp to top of image)
        text_bg = [x1,
                   max(0, y1 - text_h - 2),
                   x1 + text_w + 4,
                   y1]
        draw.rectangle(text_bg, fill="black")
        draw.text((x1 + 2, max(0, y1 - text_h - 1)), text, fill="white", font=font)

    return pil_img

# ------------------------------------------------------------
# 8. SINGLE-IMAGE PIPELINE
#    user_image → YOLO → (optional ResNet verify) → annotated image
# ------------------------------------------------------------

def run_inference_on_image(
    image_path: str,
    yolo_model: YOLO,
    classifier: keras.Model = None,
    conf_threshold: float = 0.5,
    save_name: str = None
) -> Dict[str, Any]:
    """
    Full pipeline on a single image.

    - Runs YOLO detection (with NMS internally).
    - Filters by conf_threshold.
    - Optionally runs ResNet50 classifier on each crop.
    - Draws bounding boxes + labels.
    - Saves annotated image to OUTPUT_DIR.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    print(f"\n🔍 Processing image: {image_path}")
    pil_img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = pil_img.size

    # YOLO prediction (NMS is automatically applied)
    t0 = time.perf_counter()
    results = yolo_model.predict(
        source=image_path,
        imgsz=640,
        conf=conf_threshold,
        device="cpu",     # change to "0" if you have a GPU
        verbose=False
    )
    t1 = time.perf_counter()
    infer_time = t1 - t0
    print(f"YOLO inference time: {infer_time*1000:.2f} ms")

    res = results[0]  # one image
    boxes = res.boxes  # Boxes object

    detections = []

    for box in boxes:
        # xyxy coordinates
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cls_id = int(box.cls[0].item())
        conf_yolo = float(box.conf[0].item())
        label = yolo_model.names[cls_id]  # class name from YOLO model

        # Clip coords to image size, just in case
        x1 = max(0, min(x1, orig_w - 1))
        y1 = max(0, min(y1, orig_h - 1))
        x2 = max(0, min(x2, orig_w - 1))
        y2 = max(0, min(y2, orig_h - 1))

        # Optional classification verification
        cls_label = None
        cls_conf = None
        if classifier is not None:
            crop = pil_img.crop((x1, y1, x2, y2))
            arr = preprocess_crop_for_classifier(crop)
            probs = classifier.predict(arr, verbose=0)[0]  # shape (25,)
            cls_idx = int(np.argmax(probs))
            cls_label = CLASS_NAMES[cls_idx]
            cls_conf = float(probs[cls_idx])

        detection_info = {
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "class_id_yolo": cls_id,
            "label": label,
            "conf_yolo": conf_yolo,
            "cls_label": cls_label,
            "cls_conf": cls_conf,
        }
        detections.append(detection_info)

    # Draw boxes
    annotated = pil_img.copy()
    annotated = draw_boxes_with_labels(annotated, detections)

    # Save output image
    if save_name is None:
        base = os.path.basename(image_path)
        name_wo_ext, _ = os.path.splitext(base)
        save_name = f"{name_wo_ext}_annotated.jpg"

    save_path = os.path.join(OUTPUT_DIR, save_name)
    annotated.save(save_path)
    print(f"✅ Saved annotated image to: {save_path}")

    return {
        "image_path": image_path,
        "output_path": save_path,
        "num_detections": len(detections),
        "detections": detections,
        "yolo_inference_time_sec": infer_time,
    }

# ------------------------------------------------------------
# 9. BATCH PIPELINE (MULTIPLE IMAGES)
# ------------------------------------------------------------

def run_inference_on_folder(
    folder_path: str,
    yolo_model: YOLO,
    classifier: keras.Model = None,
    conf_threshold: float = 0.5,
    max_images: int = None
) -> List[Dict[str, Any]]:
    """
    Run the full pipeline on all images in a folder.
    """
    supported_ext = (".jpg", ".jpeg", ".png")
    image_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(supported_ext)
    ]
    image_files.sort()

    if max_images is not None:
        image_files = image_files[:max_images]

    results_all = []
    for img_path in image_files:
        res = run_inference_on_image(
            img_path,
            yolo_model=yolo_model,
            classifier=classifier,
            conf_threshold=conf_threshold
        )
        results_all.append(res)

    return results_all

# ------------------------------------------------------------
# 10. SIMPLE QUANTIZATION (CLASSIFIER → TFLITE FLOAT16)
# ------------------------------------------------------------

def export_classifier_tflite_float16(
    keras_model: keras.Model,
    export_path: str = "resnet50_smartvision_float16.tflite"
):
    """
    Export the classifier to a TFLite model with float16 quantization.
    This is suitable for faster inference on CPU / mobile.
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]

    tflite_model = converter.convert()
    with open(export_path, "wb") as f:
        f.write(tflite_model)

    size_mb = os.path.getsize(export_path) / (1024 * 1024)
    print(f"✅ Exported float16 TFLite model to: {export_path} ({size_mb:.2f} MB)")

# ------------------------------------------------------------
# 11. MAIN (for quick testing)
# ------------------------------------------------------------

if __name__ == "__main__":
    print("🔧 Loading models...")
    yolo_model = load_yolo_model(YOLO_WEIGHTS)
    classifier_model = load_classifier(CLASSIFIER_WEIGHTS_PATH)

    # Example: run on a single test image
    test_image = os.path.join(BASE_DIR, "detection", "images", "test", "image_002126.jpg")
    if os.path.exists(test_image):
        _ = run_inference_on_image(
            image_path=test_image,
            yolo_model=yolo_model,
            classifier=classifier_model,
            conf_threshold=0.5,
        )
    else:
        print(f"⚠️ Example image not found: {test_image}")

    # Example: run on a folder of images
    # folder = os.path.join(BASE_DIR, "detection", "images")
    # _ = run_inference_on_folder(
    #     folder_path=folder,
    #     yolo_model=yolo_model,
    #     classifier=classifier_model,
    #     conf_threshold=0.5,
    #     max_images=10,
    # )

    # Example: export quantized classifier
    # export_classifier_tflite_float16(classifier_model)
