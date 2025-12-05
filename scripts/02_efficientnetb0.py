# ============================================================
# SMARTVISION AI - MODEL 4: EfficientNetB0 (FINE-TUNING)
# Target: High-accuracy 25-class classifier
# ============================================================

import os
import time
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

print("TensorFlow version:", tf.__version__)

from tensorflow.keras.applications.efficientnet import (
    EfficientNetB0,
    preprocess_input,
)

# ------------------------------------------------------------
# 1. CONFIGURATION
# ------------------------------------------------------------

BASE_DIR = "smartvision_dataset"
CLASS_DIR = os.path.join(BASE_DIR, "classification")
TRAIN_DIR = os.path.join(CLASS_DIR, "train")
VAL_DIR = os.path.join(CLASS_DIR, "val")
TEST_DIR = os.path.join(CLASS_DIR, "test")

IMG_SIZE = (224, 224)  # EfficientNetB0 default
BATCH_SIZE = 32
NUM_CLASSES = 25

MODELS_DIR = "saved_models"
METRICS_DIR = "smartvision_metrics"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

print("Train dir:", TRAIN_DIR)
print("Val dir  :", VAL_DIR)
print("Test dir :", TEST_DIR)

# ------------------------------------------------------------
# 2. LOAD DATASETS
# ------------------------------------------------------------

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

class_names = train_ds.class_names
print("Detected classes:", class_names)
print("Number of classes:", len(class_names))

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

# ------------------------------------------------------------
# 3. ADVANCED DATA AUGMENTATION
# ------------------------------------------------------------

def bright_jitter(x):
    x_f32 = tf.cast(x, tf.float32)
    x_f32 = tf.image.random_brightness(x_f32, max_delta=0.25)
    return tf.cast(x_f32, x.dtype)

def sat_jitter(x):
    x_f32 = tf.cast(x, tf.float32)
    x_f32 = tf.image.random_saturation(x_f32, lower=0.7, upper=1.3)
    return tf.cast(x_f32, x.dtype)

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),    # ≈ ±30 degrees
        layers.RandomZoom(0.15),
        layers.RandomContrast(0.3),
        layers.RandomTranslation(0.1, 0.1),
        layers.Lambda(bright_jitter),
        layers.Lambda(sat_jitter),
    ],
    name="advanced_data_augmentation",
)

# ------------------------------------------------------------
# 4. BUILD EfficientNetB0 MODEL (TWO-STAGE FINE-TUNING)
# ------------------------------------------------------------

def build_efficientnetb0_model():
    inputs = keras.Input(shape=(*IMG_SIZE, 3), name="input_layer")

    # 1. Data augmentation (training only)
    x = data_augmentation(inputs)

    # 2. EfficientNetB0 preprocess_input
    x = layers.Lambda(
        lambda z: preprocess_input(tf.cast(z, tf.float32)),
        name="effnet_preprocess",
    )(x)

    # 3. EfficientNetB0 base model (ImageNet)
    base_model = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMG_SIZE, 3),
        name="efficientnetb0",
    )

    base_model.trainable = False  # Stage 1: frozen

    x = base_model(x, training=False)

    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization(name="head_bn_1")(x)
    x = layers.Dense(256, activation="relu", name="head_dense_1")(x)
    x = layers.BatchNormalization(name="head_bn_2")(x)
    x = layers.Dropout(0.4, name="head_dropout")(x)

    outputs = layers.Dense(
        NUM_CLASSES,
        activation="softmax",
        name="predictions",
    )(x)

    model = keras.Model(inputs, outputs, name="EfficientNetB0_smartvision")
    return model

effnet_model = build_efficientnetb0_model()
effnet_model.summary()

# ------------------------------------------------------------
# 5. TRAINING UTILITY (WEIGHTS-ONLY .weights.h5)
# ------------------------------------------------------------

def compile_and_train(
    model,
    save_name: str,
    train_ds,
    val_ds,
    epochs: int,
    lr: float,
    initial_epoch: int = 0,
    patience_es: int = 5,
    patience_rlr: int = 2,
):
    optimizer = keras.optimizers.Adam(learning_rate=lr)

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    best_weights_path = os.path.join(
        MODELS_DIR, f"{save_name}.weights.h5"
    )

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=best_weights_path,
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=True,
            mode="max",
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=patience_es,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=patience_rlr,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        initial_epoch=initial_epoch,
        callbacks=callbacks,
    )

    return history, best_weights_path

# ------------------------------------------------------------
# 6. TWO-STAGE TRAINING
# ------------------------------------------------------------

MODEL_NAME = "efficientnetb0"

print("\n========== STAGE 1: TRAIN HEAD ONLY ==========\n")

history_stage1, effnet_stage1_best = compile_and_train(
    effnet_model,
    save_name=f"{MODEL_NAME}_stage1_best",
    train_ds=train_ds,
    val_ds=val_ds,
    epochs=10,
    lr=1e-3,
    initial_epoch=0,
    patience_es=5,
    patience_rlr=2,
)

print("Stage 1 best weights saved at:", effnet_stage1_best)

print("\n========== STAGE 2: FINE-TUNE TOP LAYERS ==========\n")

# Get the EfficientNet base from the combined model
base_model = effnet_model.get_layer("efficientnetb0")

# Unfreeze top N layers
num_unfreeze = 80
for layer in base_model.layers[:-num_unfreeze]:
    layer.trainable = False
for layer in base_model.layers[-num_unfreeze:]:
    layer.trainable = True
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False  # keep BN frozen

initial_epoch_stage2 = len(history_stage1.history["accuracy"])

history_stage2, effnet_stage2_best = compile_and_train(
    effnet_model,
    save_name=f"{MODEL_NAME}_stage2_best",
    train_ds=train_ds,
    val_ds=val_ds,
    epochs=30,          # total (Stage1 + Stage2)
    lr=5e-5,
    initial_epoch=initial_epoch_stage2,
    patience_es=5,
    patience_rlr=2,
)

print("Stage 2 best weights saved at:", effnet_stage2_best)
print("👉 Use this file in Streamlit app:", effnet_stage2_best)

# ------------------------------------------------------------
# 7. EVALUATION + SAVE METRICS & CONFUSION MATRIX
# ------------------------------------------------------------

def evaluate_and_save(model, model_name, best_weights_path, test_ds, class_names):
    print(f"\n===== EVALUATING {model_name.upper()} ON TEST SET =====")

    model.load_weights(best_weights_path)
    print(f"Loaded best weights from {best_weights_path}")

    y_true = []
    y_pred = []
    all_probs = []

    total_time = 0.0
    total_images = 0

    for images, labels in test_ds:
        images_np = images.numpy()
        bs = images_np.shape[0]

        start = time.perf_counter()
        probs = model.predict(images_np, verbose=0)
        end = time.perf_counter()

        total_time += (end - start)
        total_images += bs

        preds = np.argmax(probs, axis=1)

        y_true.extend(labels.numpy())
        y_pred.extend(preds)
        all_probs.append(probs)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    all_probs = np.concatenate(all_probs, axis=0)

    accuracy = float((y_true == y_pred).mean())
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    top5_correct = 0
    for i, label in enumerate(y_true):
        if label in np.argsort(all_probs[i])[-5:]:
            top5_correct += 1
    top5_acc = top5_correct / len(y_true)

    time_per_image = total_time / total_images
    images_per_second = 1.0 / time_per_image

    temp_w = os.path.join(MODELS_DIR, f"{model_name}_temp_for_size.weights.h5")
    model.save_weights(temp_w)
    size_mb = os.path.getsize(temp_w) / (1024 * 1024)
    os.remove(temp_w)

    cm = confusion_matrix(y_true, y_pred)

    print("\nClassification Report:")
    print(
        classification_report(
            y_true, y_pred, target_names=class_names, zero_division=0
        )
    )

    print(f"Test Accuracy        : {accuracy:.4f}")
    print(f"Weighted Precision   : {precision:.4f}")
    print(f"Weighted Recall      : {recall:.4f}")
    print(f"Weighted F1-score    : {f1:.4f}")
    print(f"Top-5 Accuracy       : {top5_acc:.4f}")
    print(f"Avg time per image   : {time_per_image*1000:.2f} ms")
    print(f"Images per second    : {images_per_second:.2f}")
    print(f"Model size (weights) : {size_mb:.2f} MB")
    print(f"Num parameters       : {model.count_params()}")

    save_dir = os.path.join(METRICS_DIR, model_name)
    os.makedirs(save_dir, exist_ok=True)

    metrics = {
        "model_name": model_name,
        "accuracy": accuracy,
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(f1),
        "top5_accuracy": float(top5_acc),
        "avg_inference_time_sec": float(time_per_image),
        "images_per_second": float(images_per_second),
        "model_size_mb": float(size_mb),
        "num_parameters": int(model.count_params()),
    }

    metrics_path = os.path.join(save_dir, "metrics.json")
    cm_path = os.path.join(save_dir, "confusion_matrix.npy")

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    np.save(cm_path, cm)

    print(f"\nSaved metrics to        : {metrics_path}")
    print(f"Saved confusion matrix to: {cm_path}")

    return metrics, cm

effnet_metrics, effnet_cm = evaluate_and_save(
    effnet_model,
    model_name="efficientnetb0_stage2",
    best_weights_path=effnet_stage2_best,
    test_ds=test_ds,
    class_names=class_names,
)

print("\n✅ EfficientNetB0 Model 4 pipeline complete.")
print("✅ Use weights file in app:", effnet_stage2_best)
