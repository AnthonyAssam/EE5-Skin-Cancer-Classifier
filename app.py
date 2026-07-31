import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Skin Lesion Classifier | GET 324 EE5",
    page_icon="🩺",
    layout="centered"
)

# Styling for visual hierarchy and clean UI
st.markdown("""
    <style>
    .main {
        background-color: #fafafa;
    }
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .guidance-box {
        background-color: #eff6ff;
        border: 1px solid #bfdbfe;
        padding: 14px 18px;
        border-radius: 8px;
        color: #1e40af;
        font-size: 0.9rem;
        margin-bottom: 1.5rem;
    }
    .result-card-benign {
        background-color: #f0fdf4;
        border: 1px solid #bbf7d0;
        padding: 16px;
        border-radius: 8px;
        color: #166534;
    }
    .result-card-malignant {
        background-color: #fef2f2;
        border: 1px solid #fecaca;
        padding: 16px;
        border-radius: 8px;
        color: #991b1b;
    }
    .footer-note {
        margin-top: 2rem;
        padding: 12px;
        border-top: 1px solid #e2e8f0;
        font-size: 0.8rem;
        color: #64748b;
    }
    </style>
""", unsafe_allow_html=True)

# Application Title
st.markdown("<div class='main-title'>🔬 Skin Cancer vs. Benign Tumor Classifier</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>GET 324 EE5 Mini-Project</div>", unsafe_allow_html=True)

# Sidebar setup
st.sidebar.header("System Specifications")
st.sidebar.write("**Group:** EE5")
st.sidebar.write("**Model:** MobileNetV2 (Transfer Learning)")
st.sidebar.write("**Target Classes:** Benign vs. Malignant")
st.sidebar.write("**Input Size:** 224 x 224 pixels")

# Load trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("skin_cancer_model.h5")

try:
    model = load_model()
except Exception as err:
    st.error("Model failure: Could not load skin_cancer_model.h5.")
    st.stop()

# Usage Notice / Image Scope Guidance
st.markdown("""
    <div class='guidance-box'>
        <b>Supported Image Types:</b> This model is specifically trained on close-up <b>dermatoscopic skin lesion images</b> (such as moles, spots, or cutaneous growths). It is not designed to analyze regular photos, documents, or non-dermatological subjects.
    </div>
""", unsafe_allow_html=True)

# File Uploader
uploaded_file = st.file_uploader("Upload a close-up skin lesion image (JPG, JPEG, or PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
    with col2:
        st.subheader("Analysis Output")
        
        with st.spinner("Processing image..."):
            # Preprocessing input image to 224x224 and normalizing
            target_size = (224, 224)
            resized_img = ImageOps.fit(image, target_size, Image.Resampling.LANCZOS)
            img_array = np.asarray(resized_img) / 255.0
            input_tensor = np.expand_dims(img_array, axis=0)
            
            # Prediction logic (0: benign, 1: malignant)
            prediction = model.predict(input_tensor, verbose=0)[0][0]
            
            if prediction > 0.5:
                confidence = prediction * 100
                st.markdown(f"""
                    <div class='result-card-malignant'>
                        <h4 style='margin:0 0 4px 0;'>Classification: Malignant</h4>
                        <p style='margin:0;'>Confidence Score: <b>{confidence:.2f}%</b></p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                confidence = (1 - prediction) * 100
                st.markdown(f"""
                    <div class='result-card-benign'>
                        <h4 style='margin:0 0 4px 0;'>Classification: Benign</h4>
                        <p style='margin:0;'>Confidence Score: <b>{confidence:.2f}%</b></p>
                    </div>
                """, unsafe_allow_html=True)
                
            st.write("")
            st.caption(f"Raw sigmoid activation: {prediction:.4f}")

# Project Note
st.markdown("""
    <div class='footer-note'>
        <b>Note:</b> Developed for academic demonstration in GET 324. While trained on verified ISIC clinical dataset samples, this system is an engineering prototype and should be interpreted as an academic demonstration rather than a certified medical diagnostic tool.
    </div>
""", unsafe_allow_html=True)
