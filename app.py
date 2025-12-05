import os
import time
import json
from typing import Dict, Any, List

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import streamlit as st
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from ultralytics import YOLO

# Keras application imports
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input as vgg16_preprocess
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input as effnet_preprocess
from pathlib import Path
# ------------------------------------------------------------
# GLOBAL CONFIG
# ------------------------------------------------------------
# ------------------------------------------------------------
# GLOBAL CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="SmartVision AI",
    page_icon="🧠",
    layout="wide",
)

# ---- Compact Header Styling ----
st.markdown("""
<style>
/* Reduce Streamlit's default top padding */
.block-container {
    padding-top: 1rem !important;
}

/* Tighten spacing between header lines */
h1 {
    margin-top: 0.2rem !important;
    margin-bottom: 0.1rem !important;
}

h3 {
    margin-top: -0.3rem !important;
    margin-bottom: 0.1rem !important;
}

/* Center text utility */
.center-text {
    text-align: center !important;
}
</style>
""", unsafe_allow_html=True)

# ---- Compact Header ----
st.markdown("""
<h1 class='center-text'>🤖⚡ <b>SmartVision AI</b> ⚡🤖</h1>
<h3 class='center-text'>🔎🎯 Intelligent Multi-Class Object Recognition System 🎯🔎</h3>
<p class='center-text' style='color: gray; margin-top:-6px;'>
    End-to-end computer vision pipeline on a COCO subset of 25 everyday object classes
</p>
""", unsafe_allow_html=True)

st.divider()

# Resolve repository root relative to this file (streamlit_app/app.py)
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parent  # repo/
SAVED_MODELS_DIR = REPO_ROOT / "saved_models"
YOLO_RUNS_DIR = REPO_ROOT / "yolo_runs"
SMARTVISION_METRICS_DIR = REPO_ROOT / "smartvision_metrics"
SMARTVISION_DATASET_DIR = REPO_ROOT / "smartvision_dataset"

# Then turn constants into Path objects / strings
YOLO_WEIGHTS_PATH = str(YOLO_RUNS_DIR / "smartvision_yolov8s6 - Copy" / "weights" / "best.pt")

CLASSIFIER_MODEL_CONFIGS = {
    "VGG16": {
        "type": "vgg16",
        "path": str(SAVED_MODELS_DIR / "vgg16_v2_stage2_best.h5"),
    },
    "ResNet50": {
        "type": "resnet50",
        "path": str(SAVED_MODELS_DIR / "resnet50_v2_stage2_best.weights.h5"),
    },
    "MobileNetV2": {
        "type": "mobilenetv2",
        "path": str(SAVED_MODELS_DIR / "mobilenetv2_v2_stage2_best.weights.h5"),
    },
    "EfficientNetB0": {
        "type": "efficientnetb0",
        "path": str(SAVED_MODELS_DIR / "efficientnetb0_stage2_best.weights.h5"),
    },
}

CLASS_METRIC_PATHS = {
    "VGG16": str(SMARTVISION_METRICS_DIR / "vgg16_v2_stage2" / "metrics.json"),
    "ResNet50": str(SMARTVISION_METRICS_DIR / "resnet50_v2_stage2" / "metrics.json"),
    "MobileNetV2": str(SMARTVISION_METRICS_DIR / "mobilenetv2_v2" / "metrics.json"),
    "EfficientNetB0": str(SMARTVISION_METRICS_DIR / "efficientnetb0" / "metrics.json"),
}

YOLO_METRICS_JSON = str(REPO_ROOT / "yolo_metrics" / "yolov8s_metrics.json")
BASE_DIR = str(SMARTVISION_DATASET_DIR)
CLASS_DIR = str(SMARTVISION_DATASET_DIR / "classification")
DET_DIR = str(SMARTVISION_DATASET_DIR / "detection")

IMG_SIZE = (224, 224)
NUM_CLASSES = 25

CLASS_NAMES = [
    "airplane", "bed", "bench", "bicycle", "bird", "bottle", "bowl",
    "bus", "cake", "car", "cat", "chair", "couch", "cow", "cup", "dog",
    "elephant", "horse", "motorcycle", "person", "pizza", "potted plant",
    "stop sign", "traffic light", "truck"
]
assert len(CLASS_NAMES) == NUM_CLASSES




# ------------------------------------------------------------
# BUILDERS – MATCH TRAINING ARCHITECTURES
# ------------------------------------------------------------

# ---------- VGG16 v2 ----------
def build_vgg16_model_v2():
    inputs = keras.Input(shape=(*IMG_SIZE, 3), name="input_layer")

    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.04),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.2),
            layers.Lambda(lambda x: tf.image.random_brightness(x, max_delta=0.2)),
            layers.Lambda(lambda x: tf.image.random_saturation(x, 0.8, 1.2)),
        ],
        name="data_augmentation",
    )

    x = data_augmentation(inputs)

    x = layers.Lambda(
        lambda z: vgg16_preprocess(tf.cast(z, tf.float32)),
        name="vgg16_preprocess",
    )(x)

    base_model = VGG16(
        include_top=False,
        weights="imagenet",
        input_tensor=x,
    )

    x = layers.GlobalAveragePooling2D(name="global_average_pooling2d")(base_model.output)
    x = layers.Dense(256, activation="relu", name="dense_256")(x)
    x = layers.Dropout(0.5, name="dropout_0_5")(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="VGG16_smartvision_v2")
    return model


# ---------- ResNet50 v2 ----------
def build_resnet50_model_v2():
    inputs = keras.Input(shape=(*IMG_SIZE, 3), name="input_layer")

    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.04),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.15),
            layers.Lambda(lambda x: tf.image.random_brightness(x, max_delta=0.15)),
            layers.Lambda(lambda x: tf.image.random_saturation(x, 0.85, 1.15)),
        ],
        name="data_augmentation",
    )

    x = data_augmentation(inputs)

    x = layers.Lambda(
        keras.applications.resnet50.preprocess_input,
        name="resnet50_preprocess",
    )(x)

    base_model = keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMG_SIZE, 3),
    )

    x = base_model(x)
    x = layers.GlobalAveragePooling2D(name="global_average_pooling2d")(x)
    x = layers.BatchNormalization(name="head_batchnorm")(x)
    x = layers.Dropout(0.4, name="head_dropout")(x)
    x = layers.Dense(256, activation="relu", name="head_dense")(x)
    x = layers.BatchNormalization(name="head_batchnorm_2")(x)
    x = layers.Dropout(0.5, name="head_dropout_2")(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="ResNet50_smartvision_v2")
    return model


# ---------- MobileNetV2 v2 ----------
def build_mobilenetv2_model_v2():
    """
    Same architecture as the MobileNetV2 v2 training script.
    """
    inputs = keras.Input(shape=(*IMG_SIZE, 3), name="input_layer")

    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.04),  # ~±15°
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.15),
            layers.Lambda(lambda x: tf.image.random_brightness(x, max_delta=0.15)),
            layers.Lambda(lambda x: tf.image.random_saturation(x, 0.85, 1.15)),
        ],
        name="data_augmentation",
    )

    x = data_augmentation(inputs)

    x = layers.Lambda(
        keras.applications.mobilenet_v2.preprocess_input,
        name="mobilenetv2_preprocess",
    )(x)

    base_model = keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMG_SIZE, 3),
    )

    x = base_model(x)
    x = layers.GlobalAveragePooling2D(name="global_average_pooling2d")(x)

    x = layers.BatchNormalization(name="head_batchnorm_1")(x)
    x = layers.Dropout(0.4, name="head_dropout_1")(x)

    x = layers.Dense(
        256,
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-4),
        name="head_dense_1",
    )(x)

    x = layers.BatchNormalization(name="head_batchnorm_2")(x)
    x = layers.Dropout(0.5, name="head_dropout_2")(x)

    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="predictions")(x)

    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="MobileNetV2_smartvision_v2",
    )
    return model


# ---------- EfficientNetB0 ----------
def bright_jitter(x):
    x_f32 = tf.cast(x, tf.float32)
    x_f32 = tf.image.random_brightness(x_f32, max_delta=0.25)
    return tf.cast(x_f32, x.dtype)

def sat_jitter(x):
    x_f32 = tf.cast(x, tf.float32)
    x_f32 = tf.image.random_saturation(x_f32, lower=0.7, upper=1.3)
    return tf.cast(x_f32, x.dtype)

def build_efficientnetb0_model():
    inputs = keras.Input(shape=(*IMG_SIZE, 3), name="input_layer")

    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.15),
            layers.RandomContrast(0.3),
            layers.RandomTranslation(0.1, 0.1),
            layers.Lambda(bright_jitter),
            layers.Lambda(sat_jitter),
        ],
        name="advanced_data_augmentation",
    )

    x = data_augmentation(inputs)

    x = layers.Lambda(
        lambda z: effnet_preprocess(tf.cast(z, tf.float32)),
        name="effnet_preprocess",
    )(x)

    # ✅ FIXED: No 'name' argument
    base_model = EfficientNetB0(
        include_top=False,
        weights="imagenet"
    )
    x = base_model(x, training=False)

    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization(name="head_bn_1")(x)
    x = layers.Dense(256, activation="relu", name="head_dense_1")(x)
    x = layers.BatchNormalization(name="head_bn_2")(x)
    x = layers.Dropout(0.4, name="head_dropout")(x)

    outputs = layers.Dense(
        NUM_CLASSES,
        activation="softmax",
        dtype="float32",
        name="predictions",
    )(x)

    model = keras.Model(inputs, outputs, name="EfficientNetB0_smartvision")
    return model


# ------------------------------------------------------------
# CACHED MODEL LOADERS
# ------------------------------------------------------------
@st.cache_resource(show_spinner=True)
def load_yolo_model() -> YOLO:
    if not os.path.exists(YOLO_WEIGHTS_PATH):
        raise FileNotFoundError(f"YOLO weights not found: {YOLO_WEIGHTS_PATH}")
    model = YOLO(YOLO_WEIGHTS_PATH)
    return model


@st.cache_resource(show_spinner=True)
def load_classification_models() -> Dict[str, keras.Model]:
    """
    Build each architecture fresh, then TRY to load your trained weights.
    If loading fails or path is None, the model is still returned
    (ImageNet-pretrained backbone + random head), so all 4 are enabled.
    """
    models: Dict[str, keras.Model] = {}

    for name, cfg in CLASSIFIER_MODEL_CONFIGS.items():
        model_type = cfg["type"]
        path = cfg["path"]

        # 1) Build the architecture
        if model_type == "vgg16":
            model = build_vgg16_model_v2()
        elif model_type == "resnet50":
            model = build_resnet50_model_v2()
        elif model_type == "mobilenetv2":
            model = build_mobilenetv2_model_v2()
        elif model_type == "efficientnetb0":
            model = build_efficientnetb0_model()
        else:
            continue

        # 2) Try to load your training weights (if path is provided and file exists)
        if path is not None and os.path.exists(path):
            try:
                model.load_weights(path)
            except Exception as e:
                st.sidebar.warning(
                    f"⚠️ Could not fully load weights for {name} from {path}: {e}\n"
                    "   Using ImageNet-pretrained backbone + random head."
                )
        elif path is not None:
            st.sidebar.warning(
                f"⚠️ Weights file for {name} not found at {path}. "
                "Using ImageNet-pretrained backbone + random head."
            )
        # if path is None → silently use ImageNet + random head

        models[name] = model

    return models


# ------------------------------------------------------------
# IMAGE HELPERS
# ------------------------------------------------------------
def read_image_file(uploaded_file) -> Image.Image:
    image = Image.open(uploaded_file).convert("RGB")
    return image


def preprocess_for_classifier(pil_img: Image.Image) -> np.ndarray:
    img_resized = pil_img.resize(IMG_SIZE, Image.BILINEAR)
    arr = np.array(img_resized, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)  # (1, H, W, 3)
    return arr


# ------------------------------------------------------------
# DRAW BOXES FOR DETECTION
# ------------------------------------------------------------
def draw_boxes_with_labels(
    pil_img: Image.Image,
    detections: List[Dict[str, Any]],
    font_path: str = None
) -> Image.Image:
    draw = ImageDraw.Draw(pil_img)

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

        if cls_label is not None:
            text = f"{yolo_label} {conf_yolo:.2f} | CLS: {cls_label} {cls_conf:.2f}"
        else:
            text = f"{yolo_label} {conf_yolo:.2f}"

        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        text_bg = [x1,
                   max(0, y1 - text_h - 2),
                   x1 + text_w + 4,
                   y1]
        draw.rectangle(text_bg, fill="black")
        draw.text((x1 + 2, max(0, y1 - text_h - 1)), text, fill="white", font=font)

    return pil_img


def run_yolo_with_optional_classifier(
    pil_img: Image.Image,
    yolo_model: YOLO,
    classifier_model: keras.Model = None,
    conf_threshold: float = 0.5
) -> Dict[str, Any]:
    """Run YOLO on a PIL image, optionally verify each box with classifier."""
    orig_w, orig_h = pil_img.size

    t0 = time.perf_counter()
    results = yolo_model.predict(
        pil_img,
        imgsz=640,
        conf=conf_threshold,
        device="cpu",  # change to "0" if GPU available
        verbose=False,
    )
    t1 = time.perf_counter()
    infer_time = t1 - t0

    res = results[0]
    boxes = res.boxes

    detections = []

    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cls_id = int(box.cls[0].item())
        conf_yolo = float(box.conf[0].item())
        label = res.names[cls_id]

        x1 = max(0, min(x1, orig_w - 1))
        y1 = max(0, min(y1, orig_h - 1))
        x2 = max(0, min(x2, orig_w - 1))
        y2 = max(0, min(y2, orig_h - 1))

        cls_label = None
        cls_conf = None
        if classifier_model is not None:
            crop = pil_img.crop((x1, y1, x2, y2))
            arr = preprocess_for_classifier(crop)
            probs = classifier_model.predict(arr, verbose=0)[0]
            idx = int(np.argmax(probs))
            cls_label = CLASS_NAMES[idx]
            cls_conf = float(probs[idx])

        detections.append(
            {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "label": label,
                "conf_yolo": conf_yolo,
                "cls_label": cls_label,
                "cls_conf": cls_conf,
            }
        )

    annotated = pil_img.copy()
    annotated = draw_boxes_with_labels(annotated, detections)

    return {
        "annotated_image": annotated,
        "detections": detections,
        "yolo_inference_time_sec": infer_time,
    }


# ------------------------------------------------------------
# METRICS LOADING
# ------------------------------------------------------------
@st.cache_data
def load_classification_metrics() -> pd.DataFrame:
    rows = []
    for name, path in CLASS_METRIC_PATHS.items():
        if os.path.exists(path):
            with open(path, "r") as f:
                m = json.load(f)
            rows.append(
                {
                    "Model": name,
                    "Accuracy": m.get("accuracy", None),
                    "F1 (weighted)": m.get("f1_weighted", None),
                    "Top-5 Accuracy": m.get("top5_accuracy", None),
                    "Images/sec": m.get("images_per_second", None),
                    "Size (MB)": m.get("model_size_mb", None),
                }
            )
    df = pd.DataFrame(rows)
    return df


@st.cache_data
def load_yolo_metrics() -> Dict[str, Any]:
    if not os.path.exists(YOLO_METRICS_JSON):
        return {}
    with open(YOLO_METRICS_JSON, "r") as f:
        return json.load(f)


# ------------------------------------------------------------
# SIDEBAR NAVIGATION
# ------------------------------------------------------------
PAGES = [
    "🏠 Home",
    "🖼️ Image Classification",
    "📦 Object Detection",
    "📊 Model Performance",
    "📷 Webcam Detection (snapshot)",
    "ℹ️ About",
]

page = st.sidebar.radio("Navigate", PAGES)

# ------------------------------------------------------------
# PAGE 1 – HOME
# ------------------------------------------------------------
if page == "🏠 Home":
    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.subheader("📌 Project Overview")
        st.markdown(
            """
SmartVision AI is a complete computer vision pipeline built on a curated subset
of **25 COCO classes**. It brings together:

- 🧠 **Image Classification** using multiple CNN backbones:  
  `VGG16 · ResNet50 · MobileNetV2 · EfficientNetB0`
- 🎯 **Object Detection** using **YOLOv8s**, fine-tuned on the same 25 classes
- 🔗 **Integrated Pipeline** where YOLO detects objects and  
  **ResNet50** verifies the cropped regions
- 📊 **Interactive Streamlit Dashboard** for demos, metrics visualization, and experiments
            """
        )


        st.markdown("""
        ### 🏷️ COCO Subset – 25 Classes Used for Training

        <style>
        .badge {
            display: inline-block;
            padding: 6px 12px;
            margin: 4px;
            background-color: #f0f2f6;
            border-radius: 12px;
            font-size: 14px;
        }
        </style>
        """, unsafe_allow_html=True)

        classes = [
            'person','bicycle','car','motorcycle','airplane','bus','truck','traffic light',
            'stop sign','bench','bird','cat','dog','horse','cow','elephant','bottle','cup',
            'bowl','pizza','cake','chair','couch','bed','potted plant'
        ]

        # Capitalize first letter of each word
        html = "".join([f"<span class='badge'>{c.title()}</span>" for c in classes])

        st.markdown(html, unsafe_allow_html=True)


    with col2:
        st.subheader("🕹️ How to Use This App")
        st.markdown(
            """
1. **🖼️ Image Classification**  
   Upload an image with a **single dominant object** to classify it.

2. **📦 Object Detection**  
   Upload a **scene with multiple objects** to run YOLOv8 detection.

3. **📊 Model Performance**  
   Explore **accuracy, F1-score, speed, and confusion matrices** for all models.

4. **📷 Webcam Detection (Snapshot)** *(optional)*  
   Capture an image via webcam and run **real-time YOLO detection**.
            """
        )
        st.markdown(
            """
> 💡 Tip: Start with **Object Detection** to see YOLOv8 in action,  
> then inspect misclassifications in **Model Performance**.
            """
        )

    st.divider()

    st.subheader("🧪 Sample Annotated Outputs")

    sample_dir = "inference_outputs"
    if os.path.exists(sample_dir):
        imgs = [
            os.path.join(sample_dir, f)
            for f in os.listdir(sample_dir)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ]
        if imgs:
            cols = st.columns(min(3, len(imgs)))
            for i, img_path in enumerate(imgs[:3]):
                with cols[i]:
                    st.image(img_path, caption=os.path.basename(img_path), width= 520)
        else:
            st.info("No sample images found in `inference_outputs/` yet.")
    else:
        st.info("`inference_outputs/` folder not found yet – run inference to create samples.")

# ------------------------------------------------------------
# PAGE 2 – IMAGE CLASSIFICATION
# ------------------------------------------------------------
elif page == "🖼️ Image Classification":
    st.subheader("Image Classification – 4 CNN Models")

    st.write(
        """
Upload an image that mainly contains **one object**.  
The app will run **all 4 CNN models** and show **top-5 predictions** per model.
"""
    )

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        pil_img = read_image_file(uploaded_file)
        st.image(pil_img, caption="Uploaded image", width=520)

        with st.spinner("Loading classification models..."):
            cls_models = load_classification_models()

        if not cls_models:
            st.error("No classification models could be loaded. Check your saved_models/ folder.")
        else:
            arr = preprocess_for_classifier(pil_img)

            st.markdown("### Predictions")
            cols = st.columns(len(cls_models))

            for (model_name, model), col in zip(cls_models.items(), cols):
                with col:
                    st.markdown(f"**{model_name}**")
                    probs = model.predict(arr, verbose=0)[0]
                    top5_idx = probs.argsort()[-5:][::-1]
                    top5_labels = [CLASS_NAMES[i] for i in top5_idx]
                    top5_probs = [probs[i] for i in top5_idx]

                    st.write(f"**Top-1:** {top5_labels[0]} ({top5_probs[0]:.3f})")
                    st.write("Top-5:")
                    for lbl, p in zip(top5_labels, top5_probs):
                        st.write(f"- {lbl}: {p:.3f}")


# ------------------------------------------------------------
# PAGE 3 – OBJECT DETECTION
# ------------------------------------------------------------
elif page == "📦 Object Detection":
    st.subheader("Object Detection – YOLOv8 + Optional ResNet Verification")

    st.write(
        """
Upload an image containing one or more of the 25 COCO classes.  
YOLOv8 will detect all objects and optionally verify them with the best classifier (ResNet50).
"""
    )

# ---- Replace your current detection block with this ----
    uploaded_file = None

    with st.form("detection_form"):
        
        # Put uploader inside the form so uploading doesn't trigger detection
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

        conf_th = st.slider("Confidence threshold", 0.1, 0.9, 0.5, 0.05)
        use_classifier = st.checkbox("Use ResNet50 classifier verification", value=True)

        # Submit button (the form will only submit when this is clicked)
        submitted = st.form_submit_button("Run Detection")

    # Only proceed when user clicked the button AND a file is present
    if submitted:
        if uploaded_file is None:
            st.warning("Please upload an image before clicking Run Detection.")
        else:
            pil_img = read_image_file(uploaded_file)

            # placeholders for in-place updates (prevents DOM insert/remove jitter)
            left_col, right_col = st.columns(2)
            left_ph = left_col.empty()
            right_ph = right_col.empty()
            table_ph = st.empty()
            meta_ph = st.empty()

            # show uploaded image immediately (fixed width)
            left_ph.image(pil_img, caption="Uploaded Image", width=520)

            # load yolo (cached) and optionally classifier (cached builder or loader)
            with st.spinner("Loading YOLO model..."):
                yolo_model = load_yolo_model()

            classifier_model = None
            if use_classifier:
                with st.spinner("Loading ResNet50 classifier..."):
                    # prefer cached loader if available
                    try:
                        cls_models = load_classification_models()
                        classifier_model = cls_models.get("ResNet50")
                    except Exception:
                        # fallback: build & load weights (rare)
                        classifier_model = build_resnet50_model_v2()
                        weights_path = CLASSIFIER_MODEL_CONFIGS["ResNet50"]["path"]
                        if os.path.exists(weights_path):
                            try:
                                classifier_model.load_weights(weights_path)
                            except Exception as e:
                                st.warning(f"Could not load ResNet50 weights: {e}")
                                classifier_model = None
                        else:
                            st.warning("ResNet50 weights not found – classifier verification disabled.")
                            classifier_model = None

            # run detection (only now)
            with st.spinner("Running detection..."):
                result = run_yolo_with_optional_classifier(
                    pil_img=pil_img,
                    yolo_model=yolo_model,
                    classifier_model=classifier_model,
                    conf_threshold=conf_th,
                )

            # update annotated image in-place (same fixed width)
            right_ph.image(result["annotated_image"], caption="Detected Result",  width=520)

            meta_ph.write(f"YOLO inference time: {result['yolo_inference_time_sec']*1000:.1f} ms — Detections: {len(result['detections'])}")

            if result["detections"]:
                df_det = pd.DataFrame([
                    {
                        "YOLO label": det["label"],
                        "YOLO confidence level": det["conf_yolo"],
                        "CLS label": det.get("cls_label"),
                        "CLS confidence level": det.get("cls_conf"),
                    }
                    for det in result["detections"]
                ])
                table_ph.dataframe(df_det)
            else:
                table_ph.info("No detections found.")

# ------------------------------------------------------------
# PAGE 4 – MODEL PERFORMANCE
# ------------------------------------------------------------
elif page == "📊 Model Performance":
    st.subheader("Model Performance – Classification vs Detection")

    # --- Classification metrics ---
    st.markdown("### 🧠 Classification Models (VGG16, ResNet50, MobileNetV2, EfficientNetB0)")
    df_cls = load_classification_metrics()
    if df_cls.empty:
        st.info("No classification metrics found yet in `smartvision_metrics/`.")
    else:
        st.dataframe(df_cls, width=520)

        col1, col2 = st.columns(2)
        with col1:
            st.bar_chart(
                df_cls.set_index("Model")["Accuracy"],
                width=520,
            )
        with col2:
            st.bar_chart(
                df_cls.set_index("Model")["F1 (weighted)"],
                width=520,
            )

        st.markdown("#### Inference Speed (images/sec)")
        st.bar_chart(
            df_cls.set_index("Model")["Images/sec"],
            width=520,
        )

    # --- YOLO metrics ---
    st.markdown("### 📦 YOLOv8 Detection Model")
    yolo_m = load_yolo_metrics()
    if not yolo_m:
        st.info("No YOLO metrics found yet in `yolo_metrics/`.")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("mAP@0.5", f"{yolo_m.get('map_50', 0):.3f}")
        with col2:
            st.metric("mAP@0.5:0.95", f"{yolo_m.get('map_50_95', 0):.3f}")
        with col3:
            st.metric("YOLO FPS", f"{yolo_m.get('fps', 0):.2f}")

        st.write("YOLO metrics JSON:", YOLO_METRICS_JSON)

    # --- Confusion matrix & comparison plots (if available) ---
    st.markdown("### 📈 Comparison Plots & Confusion Matrices")

    comp_dir = os.path.join("smartvision_metrics", "comparison_plots")
    if os.path.exists(comp_dir):
        imgs = [
            os.path.join(comp_dir, f)
            for f in os.listdir(comp_dir)
            if f.lower().endswith(".png")
        ]
        if imgs:
            for img in sorted(imgs):
                st.image(img, caption=os.path.basename(img), width=520)
        else:
            st.info("No comparison plots found in `smartvision_metrics/comparison_plots/`.")
    else:
        st.info("Folder `smartvision_metrics/comparison_plots/` not found.")


# ------------------------------------------------------------
# PAGE 5 – WEBCAM DETECTION (SNAPSHOT)
# ------------------------------------------------------------
elif page == "📷 Webcam Detection (snapshot)":
    st.subheader("Webcam Detection (Snapshot-based)")

    st.write(
        """
This page uses Streamlit's `camera_input` to grab a **single frame**
from your webcam and run YOLOv8 detection on it.

(For true real-time streaming, you would typically use `streamlit-webrtc`.)
"""
    )

    conf_th = st.slider("Confidence threshold", 0.1, 0.9, 0.5, 0.05)

    cam_image = st.camera_input("Capture image from webcam")

    if cam_image is not None:
        pil_img = Image.open(cam_image).convert("RGB")

        with st.spinner("Loading YOLO model..."):
            yolo_model = load_yolo_model()

        with st.spinner("Running detection..."):
            result = run_yolo_with_optional_classifier(
                pil_img=pil_img,
                yolo_model=yolo_model,
                classifier_model=None,  # detection-only for speed
                conf_threshold=conf_th,
            )

        st.image(result["annotated_image"], caption="Detections", width='520')
        st.write(f"YOLO inference time: {result['yolo_inference_time_sec']*1000:.1f} ms")
        st.write(f"Number of detections: {len(result['detections'])}")
        if result["detections"]:
            st.markdown("### Detected objects")
            df_det = pd.DataFrame([
                {
                    "YOLO label": det["label"],
                    "YOLO confidence level": det["conf_yolo"],

                }
                for det in result["detections"]
            ])
            st.dataframe(df_det, width = 520)


# ------------------------------------------------------------
# PAGE 6 – ABOUT
# ------------------------------------------------------------
elif page == "ℹ️ About":
    st.subheader("About SmartVision AI")

    
    st.markdown(
        """
**Dataset:**  
- Subset of MS COCO with 25 commonly occurring classes  
- Split into train/val/test for both classification & detection

**Models used:**
- **Classification:**
  - VGG16
  - ResNet50
  - MobileNetV2
  - EfficientNetB0
- **Detection:**
  - YOLOv8s fine-tuned on the same 25 classes

**Pipeline Highlights:**
- Integrated pipeline: YOLO detects → ResNet50 verifies object crops
- Performance metrics:
  - CNN test accuracy, F1, Top-5 accuracy, images/sec, model size
  - YOLO mAP@0.5, mAP@0.5:0.95, FPS
- Quantization-ready: ResNet50 can be exported to float16 TFLite for deployment.

**Tech Stack:**
- Python, TensorFlow / Keras, Ultralytics YOLOv8
- Streamlit for interactive dashboard
- NumPy, Pandas, Pillow, Matplotlib


"""
    )
