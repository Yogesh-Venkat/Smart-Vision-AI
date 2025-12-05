# scripts/convert_vgg16_weights.py

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input

print("TensorFlow version:", tf.__version__)

IMG_SIZE = (224, 224)
NUM_CLASSES = 25
MODELS_DIR = "saved_models"

# --- SAME AUGMENTATION AS IN TRAINING (ok for building, problem was only deserializing old model) ---

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.04),                     # ≈ ±15°
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.2),
        layers.Lambda(lambda x: tf.image.random_brightness(x, max_delta=0.2)),
        layers.Lambda(lambda x: tf.image.random_saturation(x, 0.8, 1.2)),
    ],
    name="data_augmentation",
)


def build_vgg16_model_v2():
    """
    EXACTLY the same architecture as your VGG16 training code.
    """
    inputs = keras.Input(shape=(*IMG_SIZE, 3), name="input_layer")

    # 1. Augmentation
    x = data_augmentation(inputs)

    # 2. VGG16-specific preprocessing
    x = layers.Lambda(
        lambda z: preprocess_input(tf.cast(z, tf.float32)),
        name="vgg16_preprocess",
    )(x)

    # 3. Pre-trained VGG16 backbone
    base_model = VGG16(
        include_top=False,
        weights="imagenet",
        input_tensor=x,
    )

    # 4. Custom head
    x = layers.GlobalAveragePooling2D(name="global_average_pooling2d")(base_model.output)
    x = layers.Dense(256, activation="relu", name="dense_256")(x)
    x = layers.Dropout(0.5, name="dropout_0_5")(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="VGG16_smartvision_v2")
    return model


if __name__ == "__main__":
    full_path = os.path.join(MODELS_DIR, "vgg16_v2_stage2_best.h5")
    weights_path = os.path.join(MODELS_DIR, "vgg16_v2_stage2_best.weights.h5")

    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Full VGG16 model .h5 not found at: {full_path}")

    print("🧱 Rebuilding VGG16 v2 architecture...")
    model = build_vgg16_model_v2()
    model.summary()

    print(f"📥 Loading weights from legacy full model file (by_name, skip_mismatch): {full_path}")
    # NOTE: this reads the HDF5 weights **without** trying to deserialize the old Lambda graph
    model.load_weights(full_path, by_name=True, skip_mismatch=True)

    print(f"💾 Saving clean weights-only file to: {weights_path}")
    model.save_weights(weights_path)
    print("✅ Done: vgg16_v2_stage2_best.weights.h5 created.")
