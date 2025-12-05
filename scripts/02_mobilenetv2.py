# ============================================================
# SMARTVISION AI - MODEL 3 (v3): MobileNetV2 (FAST + ACCURATE)
# with manual label smoothing + deeper fine-tuning
# ============================================================

import os
import time
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

print("TensorFlow version:", tf.__version__)

# ------------------------------------------------------------
# 1. CONFIGURATION
# ------------------------------------------------------------

BASE_DIR      = "smartvision_dataset"
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
# 2. LOAD DATASETS (CROPPED SINGLE-OBJECT IMAGES)
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
# 3. DATA AUGMENTATION (STANDARD, TRAIN-ONLY)
# ------------------------------------------------------------

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.04),                   # ~±15°
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.15),
        layers.Lambda(lambda x: tf.image.random_brightness(x, max_delta=0.15)),
        layers.Lambda(lambda x: tf.image.random_saturation(x, 0.85, 1.15)),
    ],
    name="data_augmentation",
)

# ------------------------------------------------------------
# 4. BUILD MobileNetV2 MODEL (2-STAGE TRAINING)
# ------------------------------------------------------------

def build_mobilenetv2_model_v2():
    """
    Returns:
        model      : full MobileNetV2 classification model
        base_model : the MobileNetV2 backbone (for freezing/unfreezing)
    """
    inputs = keras.Input(shape=(*IMG_SIZE, 3), name="input_layer")

    # Apply augmentation only during training
    x = data_augmentation(inputs)

    # MobileNetV2 expects [-1, 1] normalized inputs via preprocess_input
    x = layers.Lambda(
        keras.applications.mobilenet_v2.preprocess_input,
        name="mobilenetv2_preprocess",
    )(x)

    # Pretrained MobileNetV2 backbone
    base_model = keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMG_SIZE, 3),
    )

    # Run backbone
    x = base_model(x)

    # Global pooling + custom classification head
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

    outputs = layers.Dense(
        NUM_CLASSES, activation="softmax", name="predictions"
    )(x)

    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="MobileNetV2_smartvision_v2",
    )
    return model, base_model

mobilenet_model, base_model = build_mobilenetv2_model_v2()
mobilenet_model.summary()

# ------------------------------------------------------------
# 5. MANUAL LABEL-SMOOTHED LOSS
# ------------------------------------------------------------

def make_sparse_ce_with_label_smoothing(num_classes, label_smoothing=0.05):
    ls = float(label_smoothing)
    nc = int(num_classes)

    def loss_fn(y_true, y_pred):
        # y_true: integer labels, shape (batch,)
        y_true = tf.cast(y_true, tf.int32)
        y_true_oh = tf.one_hot(y_true, depth=nc)

        if ls > 0.0:
            smooth = ls
            y_true_oh = (1.0 - smooth) * y_true_oh + smooth / tf.cast(
                nc, tf.float32
            )

        # y_pred is softmax probabilities
        return tf.keras.losses.categorical_crossentropy(
            y_true_oh, y_pred, from_logits=False
        )

    return loss_fn

# ------------------------------------------------------------
# 6. TRAINING UTILITY (SAVES WEIGHTS-ONLY .weights.h5)
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
    """Compile and train model, saving the best weights by val_accuracy."""
    print(f"\n===== TRAINING {model_name} ({model_tag}) =====")

    optimizer = keras.optimizers.Adam(learning_rate=lr)

    loss_fn = make_sparse_ce_with_label_smoothing(
        num_classes=NUM_CLASSES,
        label_smoothing=0.05,
    )

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=["accuracy"],
    )

    # Keras 3 requirement: weights-only must end with ".weights.h5"
    best_weights_path = os.path.join(
        MODELS_DIR, f"{model_name}_{model_tag}_best.weights.h5"
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
        callbacks=callbacks,
    )

    return history, best_weights_path

# ------------------------------------------------------------
# 7. STAGE 1: TRAIN HEAD WITH FROZEN BASE
# ------------------------------------------------------------

print("\n===== STAGE 1: Training head with frozen MobileNetV2 base =====")

for layer in base_model.layers:
    layer.trainable = False

epochs_stage1 = 12
lr_stage1     = 1e-3

history_stage1, mobilenet_stage1_best = compile_and_train(
    mobilenet_model,
    model_name="mobilenetv2_v2",
    train_ds=train_ds,
    val_ds=val_ds,
    epochs=epochs_stage1,
    lr=lr_stage1,
    model_tag="stage1",
    patience_es=4,
    patience_rlr=2,
)

print("Stage 1 best weights saved at:", mobilenet_stage1_best)

# ------------------------------------------------------------
# 8. STAGE 2: DEEPER FINE-TUNE LAST LAYERS OF BASE MODEL
# ------------------------------------------------------------

print("\n===== STAGE 2: Fine-tuning last layers of MobileNetV2 base =====")

mobilenet_model.load_weights(mobilenet_stage1_best)

base_model.trainable = True
num_unfreeze = 25

print(f"Base model has {len(base_model.layers)} layers.")
print(f"Unfrozen layers in base model: {num_unfreeze}")

for layer in base_model.layers[:-num_unfreeze]:
    layer.trainable = False

for layer in base_model.layers[-num_unfreeze:]:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

epochs_stage2 = 25
lr_stage2     = 3e-5

history_stage2, mobilenet_stage2_best = compile_and_train(
    mobilenet_model,
    model_name="mobilenetv2_v2",
    train_ds=train_ds,
    val_ds=val_ds,
    epochs=epochs_stage2,
    lr=lr_stage2,
    model_tag="stage2",
    patience_es=8,
    patience_rlr=3,
)

print("Stage 2 best weights saved at:", mobilenet_stage2_best)
print("👉 Use this file in Streamlit app:", mobilenet_stage2_best)

# ------------------------------------------------------------
# 9. EVALUATION + SAVE METRICS & CONFUSION MATRIX
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

mobilenet_metrics, mobilenet_cm = evaluate_and_save(
    mobilenet_model,
    model_name="mobilenetv2_v2_stage2",
    best_weights_path=mobilenet_stage2_best,
    test_ds=test_ds,
    class_names=class_names,
)

print("\n✅ MobileNetV2 v3 (label-smoothed + deeper FT) pipeline complete.")
print("✅ Use weights file in app:", mobilenet_stage2_best)
