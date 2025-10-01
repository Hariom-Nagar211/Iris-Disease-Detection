import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

# Load model
model = tf.keras.models.load_model("iris_disease_model.h5")
categories = ["Healthy", "Unhealthy"]

# Page Config
st.set_page_config(page_title="Iris Disorder Detection", page_icon="🧿", layout="wide")

# Title & Description
st.markdown(
    """
    <h1 style='text-align: center; color: #4CAF50;'>🧿 Iris Disorder Detection</h1>
    <p style='text-align: center;'>Upload an iris image to classify it as <b>Healthy</b> or <b>Unhealthy</b> and visualize the model’s focus with <b>Grad-CAM</b>.</p>
    <hr>
    """,
    unsafe_allow_html=True
)

# Sidebar
st.sidebar.header("ℹ️ About")
st.sidebar.info(
    """
    This app uses a deep learning model (TensorFlow/Keras)  
    trained to detect iris disorders from eye images.  

    - Input size: **224x224**  
    - Output: **Healthy / Unhealthy**  
    - Transparency: **Grad-CAM heatmap** shows focus area  
    """
)


# ---------- Grad-CAM Function ----------
def get_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-8  # avoid division by zero

    # Ensure NumPy
    if isinstance(heatmap, tf.Tensor):
        heatmap = heatmap.numpy()
    return heatmap


def overlay_gradcam(img, heatmap, alpha=0.4):
    img = np.array(img)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)
    return superimposed_img


# ---------- File uploader ----------
uploaded_file = st.file_uploader("📤 Upload an iris image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype('float32')

    # Prediction
    with st.spinner("🔎 Analyzing image..."):
        pred = model.predict(img_array)[0][0]
        probability_unhealthy = float(pred)
        probability_healthy = 1 - probability_unhealthy
        label = categories[int(pred > 0.5)]

        # Grad-CAM Heatmap
        last_conv_layer_name = [layer.name for layer in model.layers if 'conv' in layer.name][-1]
        heatmap = get_gradcam_heatmap(img_array, model, last_conv_layer_name)
        gradcam_img = overlay_gradcam(img_resized, heatmap)

    # Layout in two columns
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🖼️ Uploaded Image")
        st.image(img, use_container_width=True)

        if label == "Healthy":
            st.success(f"✅ Prediction: **Healthy** ({probability_healthy:.2%} confidence)")
        else:
            st.error(f"⚠️ Prediction: **Unhealthy** ({probability_unhealthy:.2%} confidence)")

    with col2:
        st.subheader("🔥 Grad-CAM Focus Area")
        st.image(gradcam_img, caption="Model Focus (Grad-CAM)", use_container_width=True)

    # Confidence bars
    st.markdown("### 📊 Confidence Levels")
    st.progress(int(probability_healthy * 100))
    st.write(f"Healthy: **{probability_healthy:.2%}**")
    st.progress(int(probability_unhealthy * 100))
    st.write(f"Unhealthy: **{probability_unhealthy:.2%}**")
