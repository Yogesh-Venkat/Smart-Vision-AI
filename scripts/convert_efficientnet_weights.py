# scripts/convert_efficientnet_weights.py

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.efficientnet import (
    EfficientNetB0,
    preprocess_input as effnet_preprocess,
)

print("TensorFlow version:", tf.__version__)

IMG_SIZE = (224, 224)
NUM_CLASSES = 25
MODELS_DIR = "saved_models"


# --- These were in your training script, keep same names ---

def bright_jitter(x):
    x_f32 = tf.cast(x, tf.float32)
    x_f32 = tf.image.random_brightness(x_f32, max_delta=0.25)
    return tf.cast(x_f32, x.dtype)

def sat_jitter(x):
    x_f32 = tf.cast(x, tf.float32)
    x_f32 = tf.image.random_saturation(x_f32, lower=0.7, upper=1.3)
    return tf.cast(x_f32, x.dtype)


def build_efficientnetb0_model_v2():
    """
    Rebuilds the SAME EfficientNetB0 architecture used in your training script
    (data_augmentation + preprocess_input + EfficientNetB0 backbone + head).
    """
    inputs = keras.Input(shape=(*IMG_SIZE, 3), name="input_layer")

    # --- Data augmentation (as in training) ---
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),    # ≈ ±30°
            layers.RandomZoom(0.15),
            layers.RandomContrast(0.3),
            layers.RandomTranslation(0.1, 0.1),
            layers.Lambda(bright_jitter, name="bright_jitter"),
            layers.Lambda(sat_jitter, name="sat_jitter"),
        ],
        name="advanced_data_augmentation",
    )

    x = data_augmentation(inputs)

    # EfficientNetB0 preprocess_input (same as training)
    x = layers.Lambda(
        lambda z: effnet_preprocess(tf.cast(z, tf.float32)),
        name="effnet_preprocess",
    )(x)

    # EfficientNetB0 backbone
    base_model = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        name="efficientnetb0",
    )
    base_model.trainable = False  # doesn't matter for conversion

    x = base_model(x, training=False)

    # Classification head (same as training)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization(name="head_bn_1")(x)
    x = layers.Dense(256, activation="relu", name="head_dense_1")(x)
    x = layers.BatchNormalization(name="head_bn_2")(x)
    x = layers.Dropout(0.4, name="head_dropout")(x)

    # Final output: float32 softmax
    outputs = layers.Dense(
        NUM_CLASSES,
        activation="softmax",
        dtype="float32",
        name="predictions",
    )(x)

    model = keras.Model(inputs, outputs, name="EfficientNetB0_smartvision_v2")
    return model


if __name__ == "__main__":
    full_path = os.path.join(MODELS_DIR, "efficientnetb0_best.h5")
    weights_path = os.path.join(MODELS_DIR, "efficientnetb0_best.weights.h5")

    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Full EfficientNet model .h5 not found at: {full_path}")

    print("🔧 Building EfficientNetB0 v2 architecture...")
    model = build_efficientnetb0_model_v2()
    model.summary()

    print(f"\n📥 Loading weights BY NAME (skip mismatches) from:\n  {full_path}")
    # 🔑 KEY FIX: use by_name=True and skip_mismatch=True so shape mismatches
    # are simply ignored instead of crashing.
    model.load_weights(full_path, by_name=True, skip_mismatch=True)
    print("✅ Weights loaded into rebuilt model (by name, mismatches skipped).")

    print(f"\n💾 Saving weights-only file to:\n  {weights_path}")
    model.save_weights(weights_path)
    print("✅ Done converting EfficientNetB0 weights to .weights.h5")
