import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io
import time

from utils.model_utils import load_model, predict_disease
from utils.gradcam_utils import generate_gradcam
from utils.display_utils import (
    render_prediction_card,
    render_gradcam_overlay,
    render_disease_info,
    CLASS_INFO,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Iris Disease Classifier",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        color: white;
        border-radius: 12px;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: #f0f4ff;
        border-left: 5px solid #4361ee;
        border-radius: 8px;
        padding: 1.2rem;
        margin: 0.5rem 0;
    }
    .healthy-box  { border-left-color: #2dc653; background: #edfff2; }
    .warning-box  { border-left-color: #f4a261; background: #fff7ed; }
    .danger-box   { border-left-color: #e63946; background: #fff0f0; }
    .info-card {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin-bottom: 1rem;
    }
    .metric-value { font-size: 2rem; font-weight: 700; }
    .stProgress > div > div { background-color: #4361ee; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/eye.png", width=80)
    st.title("⚙️ Settings")

    st.markdown("### Model")
    confidence_threshold = st.slider(
        "Confidence threshold", 0.0, 1.0, 0.5, 0.05,
        help="Predictions below this threshold are marked as uncertain",
    )
    show_top_k = st.number_input("Show top-k predictions", 1, 10, 3)

    st.markdown("### Grad-CAM")
    gradcam_alpha = st.slider("Overlay opacity", 0.1, 1.0, 0.5, 0.05)
    colormap_choice = st.selectbox(
        "Heatmap colormap",
        ["JET", "HOT", "INFERNO", "PLASMA"],
        index=0,
    )

    st.markdown("---")
    st.markdown("**Supported conditions:**")
    for cls in CLASS_INFO:
        st.markdown(f"- {CLASS_INFO[cls]['icon']} {cls}")


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>👁️ Iris Disease Classifier</h1>
    <p style="opacity:0.85">AI-powered iris image analysis using Transfer Learning + Grad-CAM</p>
</div>
""", unsafe_allow_html=True)


# ── Load model ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def get_model():
    return load_model()

model = get_model()


# ── Upload ─────────────────────────────────────────────────────────────────────
col_upload, col_info = st.columns([1, 1])

with col_upload:
    st.markdown("### 📤 Upload Fundus Image")
    uploaded_file = st.file_uploader(
        "Choose a retinal fundus image",
        type=["jpg", "jpeg", "png", "bmp"],
        help="Upload a colour fundus photograph for analysis",
    )
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded image", use_container_width=True)

with col_info:
    st.markdown("### ℹ️ How it works")
    st.markdown("""
    1. **Upload** a retinal fundus image  
    2. The model (EfficientNetB0 pre-trained on ImageNet) analyses it  
    3. **Top-k predictions** are shown with confidence scores  
    4. **Grad-CAM heatmap** highlights the regions that influenced the prediction  
    5. A brief **disease overview** is provided for the top prediction  
    """)
    st.info("⚠️ This tool is for educational purposes only. Always consult a qualified ophthalmologist for diagnosis.")


# ── Analyse ────────────────────────────────────────────────────────────────────
if uploaded_file is not None:
    st.markdown("---")

    if st.button("🔍 Analyse Image", type="primary", use_container_width=True):
        with st.spinner("Analysing…"):
            progress = st.progress(0)
            time.sleep(0.2); progress.progress(20)

            # Convert to numpy
            img_array = np.array(image)

            # Predict
            predictions, top_k_results = predict_disease(
                model, img_array, top_k=int(show_top_k)
            )
            progress.progress(60)

            # Grad-CAM
            gradcam_img = generate_gradcam(
                model, img_array,
                alpha=gradcam_alpha,
                colormap=colormap_choice,
            )
            progress.progress(90)
            time.sleep(0.1); progress.progress(100)

        # ── Results layout ────────────────────────────────────────────────
        st.markdown("## 📊 Results")

        res_col1, res_col2 = st.columns([1, 1])

        with res_col1:
            st.markdown("### 🎯 Predictions")
            top_class, top_conf = top_k_results[0]

            # Severity colour
            severity = CLASS_INFO.get(top_class, {}).get("severity", "warning")
            box_cls = "healthy-box" if severity == "healthy" else (
                "danger-box" if severity == "high" else "warning-box"
            )

            st.markdown(f"""
            <div class="prediction-box {box_cls}">
                <h3>{CLASS_INFO.get(top_class, {}).get('icon','🔵')} {top_class}</h3>
                <p style="font-size:1.1rem">Confidence: <strong>{top_conf*100:.1f}%</strong></p>
                {'<p style="color:#e63946">⚠️ Below confidence threshold — result uncertain</p>' if top_conf < confidence_threshold else ''}
            </div>
            """, unsafe_allow_html=True)

            st.markdown("#### All predictions")
            for cls_name, conf in top_k_results:
                icon = CLASS_INFO.get(cls_name, {}).get("icon", "🔵")
                st.markdown(f"**{icon} {cls_name}**")
                st.progress(float(conf))
                st.caption(f"{conf*100:.2f}%")

        with res_col2:
            st.markdown("### 🔥 Grad-CAM Heatmap")
            st.image(
                gradcam_img,
                caption="Model attention map — warmer colours = higher attention",
                use_container_width=True,
            )
            st.caption(
                "Grad-CAM highlights the retinal regions that most influenced "
                "the model's prediction."
            )

        # ── Disease info ──────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 📚 Disease Information")
        render_disease_info(top_class)

        # ── Raw probabilities table ───────────────────────────────────────
        with st.expander("📈 Full probability distribution"):
            import pandas as pd
            df = pd.DataFrame(
                [(k, f"{v*100:.2f}%") for k, v in
                 sorted(zip(
                     [r[0] for r in top_k_results],
                     [r[1] for r in top_k_results]
                 ), key=lambda x: -x[1])],
                columns=["Class", "Confidence"],
            )
            # Show all predictions
            all_probs = sorted(
                zip(list(CLASS_INFO.keys()), predictions.tolist()),
                key=lambda x: -x[1],
            )
            df_full = pd.DataFrame(
                [(cls, f"{p*100:.2f}%") for cls, p in all_probs],
                columns=["Class", "Confidence"],
            )
            st.dataframe(df_full, use_container_width=True)
