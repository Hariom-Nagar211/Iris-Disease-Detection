"""
model_utils.py
==============
Handles model creation, weight loading (if available), and inference.

Architecture: EfficientNetB0 pre-trained on ImageNet
  - EfficientNetB0 outperforms MobileNetV2 in accuracy while staying lightweight
  - Top layers are replaced with a custom classification head
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0

# ── Constants ──────────────────────────────────────────────────────────────────
IMG_SIZE = 224
CLASSES = [
    "Central Serous Chorioretinopathy",
    "Diabetic Retinopathy",
    "Disc Edema",
    "Glaucoma",
    "Healthy",
    "Macular Scar",
    "Myopia",
    "Pterygium",
    "Retinal Detachment",
    "Retinitis Pigmentosa",
]
NUM_CLASSES = len(CLASSES)
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "best_model.keras")


# ── Build model ────────────────────────────────────────────────────────────────
def build_model(num_classes: int = NUM_CLASSES) -> tf.keras.Model:
    """
    EfficientNetB0 backbone + custom classification head.

    Design choices:
    ---------------
    • include_top=False  → remove ImageNet softmax layer
    • GlobalAveragePooling2D → compact feature vector
    • BatchNormalization + Dropout → regularisation
    • Softmax output for multi-class probability
    """
    base = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    # Freeze base initially; fine-tune later if desired
    base.trainable = False

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    # EfficientNetB0 has its own preprocessing built-in (rescaling 0-255)
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="retinal_classifier")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ── Load model ─────────────────────────────────────────────────────────────────
def load_model() -> tf.keras.Model:
    """
    Load model weights if a saved checkpoint exists,
    otherwise return the freshly built architecture.
    (Users should train and save weights using train.py)
    """
    model = build_model()

    if os.path.exists(WEIGHTS_PATH):
        try:
            model.load_weights(WEIGHTS_PATH)
            print(f"[INFO] Loaded weights from {WEIGHTS_PATH}")
        except Exception as e:
            print(f"[WARN] Could not load weights: {e}. Running with random weights.")
    else:
        print("[INFO] No saved weights found — running with ImageNet-initialised base + random head.")

    return model


# ── Preprocessing ──────────────────────────────────────────────────────────────
def preprocess_image(img_array: np.ndarray) -> np.ndarray:
    """
    Resize and prepare a uint8 HxWx3 image for model inference.
    EfficientNetB0 expects pixel values in [0, 255] — no manual rescaling needed.
    """
    img = tf.image.resize(img_array, (IMG_SIZE, IMG_SIZE))
    img = tf.cast(img, tf.float32)          # keep 0-255 range
    img = tf.expand_dims(img, axis=0)       # add batch dim → (1, 224, 224, 3)
    return img.numpy()


# ── Inference ──────────────────────────────────────────────────────────────────
def predict_disease(
    model: tf.keras.Model,
    img_array: np.ndarray,
    top_k: int = 3,
):
    """
    Run inference and return:
      - predictions : 1-D array of class probabilities (length = NUM_CLASSES)
      - top_k_results: list of (class_name, probability) tuples, descending
    """
    preprocessed = preprocess_image(img_array)
    raw_preds = model.predict(preprocessed, verbose=0)[0]   # shape (NUM_CLASSES,)

    top_k_idx = np.argsort(raw_preds)[::-1][:top_k]
    top_k_results = [(CLASSES[i], float(raw_preds[i])) for i in top_k_idx]

    return raw_preds, top_k_results
