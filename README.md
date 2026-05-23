# 👁️ Retinal Disease Classifier

AI-powered retinal fundus image analysis using **EfficientNetB0** (Transfer Learning) with **Grad-CAM** visualisation, built with Streamlit.

---

## 🗂️ Project Structure

```
retinal_disease_classifier/
├── app.py                  # Streamlit UI 
├── train.py                # Training script 
├── requirements.txt
├── models/
│   └── best_model.keras    # Saved after training 
└── utils/
    ├── __init__.py
    ├── model_utils.py      # Model architecture, loading, inference
    ├── gradcam_utils.py    # Grad-CAM heatmap generation
    └── display_utils.py    # Streamlit UI helpers + disease metadata
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. (Optional) Train the model

#### Before traning download the dataset and organise it in the following structure:
dataset link : https://www.kaggle.com/datasets/jeandedieunyandwi/lending-club-dataset

```
dataset/
├── Central Serous Chorioretinopathy/
├── Diabetic Retinopathy/
├── Disc Edema/
├── Glaucoma/
├── Healthy/
├── Macular Scar/
├── Myopia/
├── Pterygium/
├── Retinal Detachment/
└── Retinitis Pigmentosa/
```

Then run:

```bash
python train.py --data_dir /path/to/dataset --epochs 30 --batch 32
```

The best weights are automatically saved to `models/best_model.keras`.

### 3. Run the app

```bash
streamlit run app.py
```

---

## 🧠 Model Architecture

| Component | Details |
|-----------|---------|
| Backbone | EfficientNetB0 (ImageNet pre-trained) |
| Input size | 224 × 224 × 3 |
| Head | GAP → BN → Dropout(0.3) → Dense(256) → Dropout(0.2) → Softmax(10) |
| Optimizer | Adam (1e-4 → 1e-5 for fine-tuning) |
| Loss | Categorical cross-entropy |
| Fine-tuning | Top 50 backbone layers unfrozen in Phase 2 |

**Why EfficientNetB0 over MobileNetV2?**
- ~1.5–2% higher ImageNet top-1 accuracy at similar latency
- Better compound scaling (depth + width + resolution simultaneously)
- More robust feature extraction for medical images

---

## 🔥 Grad-CAM Explainability

Grad-CAM (Gradient-weighted Class Activation Mapping) shows *which retinal regions* the model focused on for its prediction:

1. Forward pass → record last conv layer activations
2. Backpropagate class score → get gradients at last conv layer
3. Pool gradients spatially → per-channel importance weights
4. Weight activations → raw heatmap
5. ReLU + normalise → overlay on original image

Colour scale: **blue → green → yellow → red** (low → high attention)

---

## 🩺 Supported Conditions

| Condition | Severity |
|-----------|----------|
| ✅ Healthy | — |
| 👓 Myopia | Low |
| 🟤 Pterygium | Low |
| 🔵 Central Serous Chorioretinopathy | Medium |
| ⚫ Macular Scar | Medium |
| 🟡 Diabetic Retinopathy | High |
| 🟠 Disc Edema | High |
| 🔴 Glaucoma | High |
| 🚨 Retinal Detachment | High |
| 🌑 Retinitis Pigmentosa | High |

---

## 🔮 Future Improvements

- **Larger backbone options** — plug-in support for EfficientNetB3/B5 or
  Vision Transformers (ViT) for higher accuracy on larger datasets
- **DICOM support** — accept raw clinical DICOM files directly, not just
  exported JPG/PNG images
- **Patient history integration** — incorporate age, diabetes duration, and
  IOP readings as auxiliary inputs alongside the image
- **Uncertainty quantification** — Monte Carlo Dropout at inference time to
  produce calibrated confidence intervals, not just point estimates
- **REST API + Docker** — expose the model as a FastAPI microservice
  containerised with Docker for easy hospital system integration
- **Federated learning** — train across multiple hospital datasets without
  centralising sensitive patient images
- **Automated report generation** — export a PDF clinical summary with the
  Grad-CAM overlay, top predictions, and disease overview for each patient scan

---

## ⚠️ Disclaimer

This tool is for **educational and research purposes only**. It is not a certified medical device. Always consult a qualified ophthalmologist for clinical diagnosis and treatment decisions.
