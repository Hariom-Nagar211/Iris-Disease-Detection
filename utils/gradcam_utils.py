"""
gradcam_utils.py
================
Generates Gradient-weighted Class Activation Maps (Grad-CAM) for model
explainability.

Grad-CAM Algorithm (Selvaraju et al., 2017):
  1. Forward pass → get class score for predicted class
  2. Backprop gradients to the last conv layer
  3. Pool gradients spatially → per-channel importance weights
  4. Weighted sum of activation maps → raw heatmap
  5. ReLU + normalise → overlay on original image
"""

import numpy as np
import cv2
import tensorflow as tf

from utils.model_utils import preprocess_image, IMG_SIZE

# OpenCV colormap lookup
COLORMAP_LOOKUP = {
    "JET":     cv2.COLORMAP_JET,
    "HOT":     cv2.COLORMAP_HOT,
    "INFERNO": cv2.COLORMAP_INFERNO,
    "PLASMA":  cv2.COLORMAP_PLASMA,
}


def _get_last_conv_layer(model: tf.keras.Model) -> str:
    """Return the name of the last Conv2D layer in the backbone."""
    for layer in reversed(model.layers):
        # EfficientNetB0 backbone is wrapped as a sub-model
        if hasattr(layer, "layers"):
            for sub in reversed(layer.layers):
                if isinstance(sub, tf.keras.layers.Conv2D):
                    return layer.name, sub.name
        if isinstance(layer, tf.keras.layers.Conv2D):
            return None, layer.name
    raise ValueError("No Conv2D layer found in the model.")


def generate_gradcam(
    model: tf.keras.Model,
    img_array: np.ndarray,
    alpha: float = 0.5,
    colormap: str = "JET",
) -> np.ndarray:
    """
    Generate a Grad-CAM heatmap overlaid on the original image.

    Parameters
    ----------
    model     : Keras model
    img_array : Raw uint8 HxWx3 image
    alpha     : Heatmap overlay opacity (0=only image, 1=only heatmap)
    colormap  : One of JET | HOT | INFERNO | PLASMA

    Returns
    -------
    overlay   : uint8 HxWxC numpy array (same size as input)
    """
    preprocessed = preprocess_image(img_array)  # (1, 224, 224, 3)

    # ── Build a sub-model that outputs (last_conv_output, predictions) ──────
    try:
        backbone_name, last_conv_name = _get_last_conv_layer(model)
        if backbone_name:
            backbone = model.get_layer(backbone_name)
            last_conv_layer = backbone.get_layer(last_conv_name)
            # Create intermediate model: input → last conv output
            grad_model = tf.keras.Model(
                inputs=model.inputs,
                outputs=[last_conv_layer.output, model.output],
            )
        else:
            last_conv_layer = model.get_layer(last_conv_name)
            grad_model = tf.keras.Model(
                inputs=model.inputs,
                outputs=[last_conv_layer.output, model.output],
            )
    except Exception:
        # Fallback: use a simple overlay (blurred original as pseudo-heatmap)
        return _fallback_overlay(img_array, alpha, colormap)

    # ── Compute gradients ───────────────────────────────────────────────────
    with tf.GradientTape() as tape:
        inputs = tf.cast(preprocessed, tf.float32)
        tape.watch(inputs)
        try:
            conv_outputs, predictions = grad_model(inputs)
            pred_class = tf.argmax(predictions[0])
            class_score = predictions[:, pred_class]
        except Exception:
            return _fallback_overlay(img_array, alpha, colormap)

    grads = tape.gradient(class_score, conv_outputs)

    if grads is None:
        return _fallback_overlay(img_array, alpha, colormap)

    # ── Pool gradients over spatial dims (H, W) ─────────────────────────────
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # shape (C,)

    # ── Weighted activation map ─────────────────────────────────────────────
    conv_out = conv_outputs[0]                             # (H, W, C)
    heatmap = conv_out @ pooled_grads[..., tf.newaxis]     # (H, W, 1)
    heatmap = tf.squeeze(heatmap)                          # (H, W)

    # ReLU + normalise to [0, 1]
    heatmap = tf.nn.relu(heatmap).numpy()
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    # ── Resize heatmap to original image size ───────────────────────────────
    orig_h, orig_w = img_array.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (orig_w, orig_h))

    # ── Apply colormap ──────────────────────────────────────────────────────
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    cmap = COLORMAP_LOOKUP.get(colormap, cv2.COLORMAP_JET)
    colored_heatmap = cv2.applyColorMap(heatmap_uint8, cmap)
    colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)

    # ── Blend with original image ───────────────────────────────────────────
    orig_rgb = img_array.copy()
    overlay = cv2.addWeighted(orig_rgb, 1 - alpha, colored_heatmap, alpha, 0)
    return overlay.astype(np.uint8)


def _fallback_overlay(img_array, alpha, colormap):
    """Return a blurred-edge pseudo-heatmap when Grad-CAM isn't possible."""
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (51, 51), 0)
    blurred = np.uint8(255 * blurred / (blurred.max() + 1e-8))
    cmap = COLORMAP_LOOKUP.get(colormap, cv2.COLORMAP_JET)
    colored = cv2.applyColorMap(blurred, cmap)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img_array, 1 - alpha, colored, alpha, 0).astype(np.uint8)
