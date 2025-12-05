# ============================================================
# SMARTVISION AI - MODEL 1 (v2): VGG16 (TRANSFER LEARNING + FT)
# with proper preprocess_input + label smoothing + deeper FT
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

from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input

print("TensorFlow version:", tf.__version__)

# ------------------------------------------------------------
# 1. CONFIGURATION
# ------------------------------------------------------------

BASE_DIR      = "smartvision_dataset"  # your dataset root
CLASS_DIR     = os.path.join(BASE_DIR, "classification")
TRAIN_DIR     = os.path.join(CLASS_DIR, "train")
VAL_DIR       = os.path.join(CLASS_DIR, "val")
TEST_DIR      = os.path.join(CLASS_DIR, "test")

IMG_SIZE      = (224, 224)
BATCH_SIZE    = 32
NUM_CLASSES   = 25

MODELS_DIR    = "saved_models"
METRICS_DIR   = "smartvision_metrics"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

print("Train dir:", TRAIN_DIR)
print("Val dir  :", VAL_DIR)
print("Test dir :", TEST_DIR)

# ------------------------------------------------------------
# 2. LOAD DATASETS (FROM CROPPED SINGLE-OBJECT IMAGES)
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
val_ds   = val_ds.prefetch(AUTOTUNE)
test_ds  = test_ds.prefetch(AUTOTUNE)

# ------------------------------------------------------------
# 3. DATA AUGMENTATION (APPLIED ONLY DURING TRAINING)
# ------------------------------------------------------------

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),                 # random horizontal flips
        layers.RandomRotation(0.04),                     # ≈ ±15 degrees
        layers.RandomZoom(0.1),                          # random zoom
        layers.RandomContrast(0.2),                      # ±20% contrast
        layers.Lambda(lambda x: tf.image.random_brightness(x, max_delta=0.2)),
        layers.Lambda(lambda x: tf.image.random_saturation(x, 0.8, 1.2)),
    ],
    name="data_augmentation",
)

# NOTE:
# We DO NOT use Rescaling(1./255) here.
# Instead, we use VGG16's preprocess_input which subtracts ImageNet means
# and expects BGR ordering. This matches the pretrained weights.

# ------------------------------------------------------------
# 4. BUILD VGG16 MODEL (FROZEN BASE + CUSTOM HEAD)
# ------------------------------------------------------------

def build_vgg16_model_v2():
    inputs = keras.Input(shape=(*IMG_SIZE, 3), name="input_layer")

    # 1. Augmentation (only active during training)
    x = data_augmentation(inputs)

    # 2. VGG16-specific preprocessing
    x = layers.Lambda(
        lambda z: preprocess_input(tf.cast(z, tf.float32)),
        name="vgg16_preprocess"
    )(x)

    # 3. Pre-trained VGG16 backbone (no top classification head)
    base_model = VGG16(
        include_top=False,
        weights="imagenet",
        input_tensor=x,
    )

    # Freeze backbone initially (Stage 1)
    base_model.trainable = False

    # 4. Custom classification head for 25 classes
    x = layers.GlobalAveragePooling2D(name="global_average_pooling2d")(base_model.output)
    x = layers.Dense(256, activation="relu", name="dense_256")(x)
    x = layers.Dropout(0.5, name="dropout_0_5")(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="VGG16_smartvision_v2")
    return model

vgg16_model = build_vgg16_model_v2()
vgg16_model.summary()

# ------------------------------------------------------------
# 5. CUSTOM LOSS WITH LABEL SMOOTHING
# ------------------------------------------------------------

def make_sparse_ce_with_label_smoothing(num_classes, label_smoothing=0.05):
    """
    Implements sparse categorical crossentropy with manual label smoothing.
    Works even if your Keras version doesn't support `label_smoothing` in
    SparseCategoricalCrossentropy.__init__.
    """
    ls = float(label_smoothing)
    nc = int(num_classes)

    def loss_fn(y_true, y_pred):
        # y_true: integer labels, shape (batch,)
        y_true = tf.cast(y_true, tf.int32)
        y_true_oh = tf.one_hot(y_true, depth=nc)

        if ls > 0.0:
            smooth = ls
            y_true_oh = (1.0 - smooth) * y_true_oh + smooth / tf.cast(nc, tf.float32)

        # y_pred is softmax probabilities
        return tf.keras.losses.categorical_crossentropy(
            y_true_oh, y_pred, from_logits=False
        )

    return loss_fn

# ------------------------------------------------------------
# 6. TRAINING UTILITY (COMMON FOR STAGE 1 & 2)
# ------------------------------------------------------------

def compile_and_train(
    model,
    model_name,
    train_ds,
    val_ds,
    epochs,
    lr,
    model_tag,
    patience_es=5,
    patience_rlr=2,
):
    """
    Compile and train model, saving the best weights by val_accuracy.
    model_name: base name ("vgg16_v2")
    model_tag : "stage1" or "stage2" etc.
    """
    print(f"\n===== TRAINING {model_name} ({model_tag}) =====")

    optimizer = keras.optimizers.Adam(learning_rate=lr)

    # Use our custom loss with label smoothing
    loss_fn = make_sparse_ce_with_label_smoothing(
        num_classes=NUM_CLASSES,
        label_smoothing=0.05,
    )

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=["accuracy"],
    )

    best_weights_path = os.path.join(MODELS_DIR, f"{model_name}_{model_tag}_best.h5")

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=best_weights_path,
            monitor="val_accuracy",
            save_best_only=True,
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
        callbacks=callbacks,
    )

    return history, best_weights_path

# ------------------------------------------------------------
# 7. STAGE 1: TRAIN HEAD WITH FROZEN VGG16 BASE
# ------------------------------------------------------------

print("\n===== STAGE 1: Training head with frozen VGG16 base =====")

# Safety: ensure all VGG16 conv blocks are frozen
for layer in vgg16_model.layers:
    if layer.name.startswith("block"):
        layer.trainable = False

epochs_stage1 = 20
lr_stage1     = 1e-4

history_stage1, vgg16_stage1_best = compile_and_train(
    vgg16_model,
    model_name="vgg16_v2",
    train_ds=train_ds,
    val_ds=val_ds,
    epochs=epochs_stage1,
    lr=lr_stage1,
    model_tag="stage1",
    patience_es=5,
    patience_rlr=2,
)

print("Stage 1 best weights saved at:", vgg16_stage1_best)

# ------------------------------------------------------------
# 8. STAGE 2: FINE-TUNE BLOCK4 + BLOCK5 OF VGG16
# ------------------------------------------------------------

print("\n===== STAGE 2: Fine-tuning VGG16 block4 + block5 =====")

# Load best Stage 1 weights before fine-tuning
vgg16_model.load_weights(vgg16_stage1_best)

# Unfreeze only block4_* and block5_* layers for controlled fine-tuning
for layer in vgg16_model.layers:
    if layer.name.startswith("block5") :
        layer.trainable = True      # fine-tune top two blocks
    elif layer.name.startswith("block"):
        layer.trainable = False     # keep lower blocks frozen (block1–3)

# Head layers (GAP + Dense + Dropout + output) remain trainable

epochs_stage2 = 15
lr_stage2     = 1e-5   # slightly higher than 1e-5 but still safe for FT

history_stage2, vgg16_stage2_best = compile_and_train(
    vgg16_model,
    model_name="vgg16_v2",
    train_ds=train_ds,
    val_ds=val_ds,
    epochs=epochs_stage2,
    lr=lr_stage2,
    model_tag="stage2",
    patience_es=6,
    patience_rlr=3,
)

print("Stage 2 best weights saved at:", vgg16_stage2_best)

# ------------------------------------------------------------
# 9. EVALUATION + SAVE METRICS & CONFUSION MATRIX
# ------------------------------------------------------------

def evaluate_and_save(model, model_name, best_weights_path, test_ds, class_names):
    print(f"\n===== EVALUATING {model_name.upper()} ON TEST SET =====")

    # Load best weights
    model.load_weights(best_weights_path)
    print(f"Loaded best weights from {best_weights_path}")

    y_true = []
    y_pred = []
    all_probs = []

    total_time = 0.0
    total_images = 0

    # Predict over test dataset
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

    # Basic metrics
    accuracy = float((y_true == y_pred).mean())
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    # Top-5 accuracy
    top5_correct = 0
    for i, label in enumerate(y_true):
        if label in np.argsort(all_probs[i])[-5:]:
            top5_correct += 1
    top5_acc = top5_correct / len(y_true)

    # Inference time
    time_per_image = total_time / total_images
    images_per_second = 1.0 / time_per_image

    # Model size (weights only)
    temp_w = os.path.join(MODELS_DIR, f"{model_name}_temp_for_size.weights.h5")
    model.save_weights(temp_w)
    size_mb = os.path.getsize(temp_w) / (1024 * 1024)
    os.remove(temp_w)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    print(f"Test Accuracy        : {accuracy:.4f}")
    print(f"Weighted Precision   : {precision:.4f}")
    print(f"Weighted Recall      : {recall:.4f}")
    print(f"Weighted F1-score    : {f1:.4f}")
    print(f"Top-5 Accuracy       : {top5_acc:.4f}")
    print(f"Avg time per image   : {time_per_image*1000:.2f} ms")
    print(f"Images per second    : {images_per_second:.2f}")
    print(f"Model size (weights) : {size_mb:.2f} MB")
    print(f"Num parameters       : {model.count_params()}")

    # Save metrics + confusion matrix in dedicated folder
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


# Evaluate FINAL (fine-tuned) model on test set
vgg16_metrics, vgg16_cm = evaluate_and_save(
    vgg16_model,
    model_name="vgg16_v2_stage2",
    best_weights_path=vgg16_stage2_best,
    test_ds=test_ds,
    class_names=class_names,
)

print("\n✅ VGG16 v2 (2-stage, improved) pipeline complete.")
