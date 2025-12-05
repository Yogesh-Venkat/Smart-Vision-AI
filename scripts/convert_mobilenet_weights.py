import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

IMG_SIZE = (224, 224)
NUM_CLASSES = 25

# ---- this MUST match your training build_mobilenetv2_model_v2 ----
def build_mobilenetv2_model_v2():
    inputs = keras.Input(shape=(*IMG_SIZE, 3), name="input_layer")

    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.04),                   # ~±15°
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.15),
            layers.Lambda(
                lambda x: tf.image.random_brightness(x, max_delta=0.15)
            ),
            layers.Lambda(
                lambda x: tf.image.random_saturation(x, 0.85, 1.15)
            ),
        ],
        name="data_augmentation",   # 👈 same name as training
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

    outputs = layers.Dense(
        NUM_CLASSES, activation="softmax", name="predictions"
    )(x)

    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="MobileNetV2_smartvision_v2",
    )
    return model


if __name__ == "__main__":
    old_path = os.path.join("saved_models", "mobilenetv2_v2_stage2_best.h5")
    new_path = os.path.join("saved_models", "mobilenetv2_v2_stage2_best.weights.h5")

    print("Building MobileNetV2 architecture...")
    model = build_mobilenetv2_model_v2()

    print("Loading weights from full .h5 (by_name, skip_mismatch)...")
    model.load_weights(old_path, by_name=True, skip_mismatch=True)

    print("Saving clean weights-only file...")
    model.save_weights(new_path)

    print("✅ Done. Saved weights-only file to:", new_path)
