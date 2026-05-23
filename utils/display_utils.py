"""
display_utils.py
================
UI helper functions and disease-information metadata for the Streamlit app.
"""

import streamlit as st
import numpy as np

# ── Disease metadata ───────────────────────────────────────────────────────────
CLASS_INFO = {
    "Central Serous Chorioretinopathy": {
        "icon": "🔵",
        "severity": "medium",
        "description": (
            "Fluid accumulates beneath the retina, typically near the macula. "
            "Usually self-resolving, but chronic cases can cause permanent vision loss."
        ),
        "symptoms": ["Blurred central vision", "Distorted images", "Dimming of colours"],
        "treatment": "Observation, laser therapy, or photodynamic therapy for chronic cases.",
        "prevalence": "Most common in men aged 30–50.",
    },
    "Diabetic Retinopathy": {
        "icon": "🟡",
        "severity": "high",
        "description": (
            "Damage to retinal blood vessels caused by long-term diabetes. "
            "A leading cause of blindness in working-age adults worldwide."
        ),
        "symptoms": ["Floaters", "Blurred vision", "Dark areas", "Vision loss"],
        "treatment": "Blood sugar control, laser photocoagulation, anti-VEGF injections, vitrectomy.",
        "prevalence": "Affects ~1 in 3 people with diabetes.",
    },
    "Disc Edema": {
        "icon": "🟠",
        "severity": "high",
        "description": (
            "Swelling of the optic disc, often due to raised intracranial pressure "
            "(papilloedema) or local inflammation (papillitis)."
        ),
        "symptoms": ["Headache", "Nausea", "Transient visual obscurations", "Double vision"],
        "treatment": "Treat underlying cause; may require lumbar puncture, medication, or surgery.",
        "prevalence": "Rare but can indicate serious systemic disease.",
    },
    "Glaucoma": {
        "icon": "🔴",
        "severity": "high",
        "description": (
            "Group of conditions causing optic nerve damage, usually linked to elevated "
            "intraocular pressure. Progressive and irreversible if untreated."
        ),
        "symptoms": ["Gradual peripheral vision loss", "Tunnel vision (late stage)", "Eye pain (acute)"],
        "treatment": "Eye drops, laser therapy (SLT), or filtering surgery (trabeculectomy).",
        "prevalence": "Second leading cause of blindness globally; ~80 million affected.",
    },
    "Healthy": {
        "icon": "✅",
        "severity": "healthy",
        "description": (
            "No significant retinal pathology detected. The optic disc, macula, vessels "
            "and peripheral retina appear within normal limits."
        ),
        "symptoms": ["None"],
        "treatment": "Routine annual eye examinations recommended.",
        "prevalence": "N/A",
    },
    "Macular Scar": {
        "icon": "⚫",
        "severity": "medium",
        "description": (
            "Fibrous tissue replacing normal retinal cells at the macula, often following "
            "trauma, inflammation, or wet age-related macular degeneration."
        ),
        "symptoms": ["Central vision loss", "Distorted/missing central vision"],
        "treatment": "Limited; low-vision rehabilitation aids can help. Prevention of further damage.",
        "prevalence": "Common sequela of treated wet AMD.",
    },
    "Myopia": {
        "icon": "👓",
        "severity": "low",
        "description": (
            "Nearsightedness caused by an elongated eyeball or steep corneal curvature. "
            "High myopia (>−6 D) is associated with retinal complications."
        ),
        "symptoms": ["Difficulty seeing distant objects", "Squinting", "Eye strain"],
        "treatment": "Corrective lenses, contact lenses, or refractive surgery (LASIK/PRK).",
        "prevalence": "Affects ~2.6 billion people globally.",
    },
    "Pterygium": {
        "icon": "🟤",
        "severity": "low",
        "description": (
            "Fleshy overgrowth of conjunctival tissue onto the cornea, linked to UV exposure. "
            "Though superficial, it can distort vision when it encroaches on the visual axis."
        ),
        "symptoms": ["Redness", "Foreign-body sensation", "Blurred vision (if large)"],
        "treatment": "Eye drops for mild cases; surgical excision for significant pterygium.",
        "prevalence": "Common in tropical/outdoor populations.",
    },
    "Retinal Detachment": {
        "icon": "🚨",
        "severity": "high",
        "description": (
            "Separation of the neurosensory retina from the underlying retinal pigment epithelium — "
            "an ophthalmic emergency requiring immediate intervention."
        ),
        "symptoms": ["Sudden flashes", "Floaters", "Shadow/curtain across vision"],
        "treatment": "Urgent surgery: pneumatic retinopexy, scleral buckle, or vitrectomy.",
        "prevalence": "~1 in 10,000 per year; higher risk with myopia or trauma.",
    },
    "Retinitis Pigmentosa": {
        "icon": "🌑",
        "severity": "high",
        "description": (
            "Inherited degenerative disease causing progressive rod and cone photoreceptor loss, "
            "leading to tunnel vision and eventual blindness."
        ),
        "symptoms": ["Night blindness", "Tunnel vision", "Loss of peripheral vision"],
        "treatment": "No cure; Vitamin A supplementation, gene therapy (Luxturna for RPE65 mutations).",
        "prevalence": "Affects ~1.5 million people worldwide.",
    },
}

SEVERITY_COLORS = {
    "healthy": "#2dc653",
    "low":     "#74b9ff",
    "medium":  "#f4a261",
    "high":    "#e63946",
}


# ── UI helpers ─────────────────────────────────────────────────────────────────
def render_prediction_card(class_name: str, confidence: float, rank: int = 1):
    info = CLASS_INFO.get(class_name, {})
    severity = info.get("severity", "medium")
    color = SEVERITY_COLORS.get(severity, "#888")
    icon = info.get("icon", "🔵")

    st.markdown(f"""
    <div style="border-left:5px solid {color}; padding:0.8rem 1rem;
                background:#f9f9f9; border-radius:6px; margin-bottom:0.5rem;">
        <strong>#{rank} {icon} {class_name}</strong>
        <span style="float:right; color:{color}; font-weight:700;">
            {confidence*100:.1f}%
        </span>
    </div>
    """, unsafe_allow_html=True)


def render_gradcam_overlay(overlay_img: np.ndarray):
    st.image(overlay_img, use_container_width=True,
             caption="Grad-CAM — warmer regions = higher model attention")


def render_disease_info(class_name: str):
    info = CLASS_INFO.get(class_name)
    if not info:
        st.warning("No information available for this class.")
        return

    severity = info["severity"]
    color = SEVERITY_COLORS.get(severity, "#888")
    icon = info["icon"]

    cols = st.columns([1, 1])

    with cols[0]:
        st.markdown(f"""
        <div style="background:white; border-radius:10px; padding:1.2rem;
                    box-shadow:0 2px 8px rgba(0,0,0,0.08);">
            <h4 style="color:{color}">{icon} {class_name}</h4>
            <p>{info['description']}</p>
            <p><strong>Prevalence:</strong> {info['prevalence']}</p>
        </div>
        """, unsafe_allow_html=True)

    with cols[1]:
        st.markdown(f"""
        <div style="background:white; border-radius:10px; padding:1.2rem;
                    box-shadow:0 2px 8px rgba(0,0,0,0.08);">
            <h4>🩺 Clinical Details</h4>
            <p><strong>Symptoms:</strong> {', '.join(info['symptoms'])}</p>
            <p><strong>Treatment:</strong> {info['treatment']}</p>
            <p style="color:#888; font-size:0.85rem;">
                ⚠️ For informational purposes only. Consult a qualified ophthalmologist.
            </p>
        </div>
        """, unsafe_allow_html=True)
