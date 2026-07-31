import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# Page Configuration
st.set_page_config(
    page_title="DermAI | Skin Lesion Classifier",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom Styling for a polished frontend UI
st.markdown("""
    <style>
    /* Main container styling */
    .main {
        background-color: #f8f9fa;
    }
    
    /* Title and Header customization */
    .title-text {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        color: #1e293b;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0px;
    }
    .subtitle-text {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        color: #64748b;
        text-align: center;
        font-size: 1.05rem;
        margin-bottom: 25px;
    }

    /* Result Card Styling */
    .result-card-malignant {
        background-color: #fef2f2;
        border-left: 6px solid #ef4444;
        padding: 20px;
        border-radius: 8px;
        margin-top: 20px;
    }
    .result-card-benign {
        background-color: #f0fdf4;
        border-left: 6px solid #22c55e;
        padding: 20px;
        border-radius: 8px;
        margin-top: 20px;
    }
    .result-title {
        font-size: 1.25rem;
        font-weight: 700;
        margin-bottom: 5px;
    }
    .result-title-malignant { color: #991b1b; }
    .result-title-benign { color: #166534; }
    
    /* Footer Disclaimer */
    .disclaimer-box {
        background-color: #f1f5f9;
        border: 1px solid #e2e8f0;
        padding: 12px 16px;
        border-radius: 6px;
        font-size: 0.85rem;
        color: #475569;
        margin-top: 30px;
    }
    </style>
""", unsafe_allow_syntax_gradient=True, unsafe_allow_html=True)

# Application Header
st.markdown("<h1 class='title-text'>DermAI Vision</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle-text'>Automated Dermatoscopic Image Analysis (EE5 Task Group)</p>", unsafe_allow_html=True)

st.markdown("---")

# Sidebar Details
with st.sidebar:
    st.header("📌 Project Details")
    st.markdown("**Course:** GET 324 Mini-Project")
    st.markdown("**Group Assignment:** EE5")
    st.markdown("**Scope:** Binary Classification")
    st.markdown("---")
    st.markdown("**Model Specs:**")
    st.markdown("* Base Architecture: MobileNetV2")
    st.markdown("* Input Resolution: 224 × 224 px")
    st.markdown("* Target Classes: Benign vs Malignant")

# Model Loading Strategy
@st.cache_resource
def load_trained_model():
    return tf.keras.models.load_model("skin_cancer_model.h5")

try:
    model = load_trained_model()
except Exception as e:
    st.error("⚠️ Model file 'skin_cancer_model.h5' could not be loaded. Ensure the file is present in the repository root directory.")
    st.stop()

# Main Workspace
st.subheader("1. Upload Dermatoscopic Image")
uploaded_file = st.file_uploader("Choose a clear skin lesion photograph (PNG, JPG, or JPEG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(image, caption='Uploaded Image Preview', use_column_width=True)
        
    with col2:
        st.subheader("2. Image Analysis")
        with st.spinner('Preprocessing tensor and executing inference...'):
            # Preprocessing input to match training configuration
            target_size = (224, 224)
            resized_img = ImageOps.fit(image, target_size, Image.Resampling.LANCZOS)
            img_array = np.asarray(resized_img) / 255.0
            input_tensor = np.expand_dims(img_array, axis=0)
            
            # Predict
            raw_prediction = model.predict(input_tensor, verbose=0)[0][0]
            
            # Thresholding logic: class_indices -> {'benign': 0, 'malignant': 1}
            is_malignant = raw_prediction > 0.5
            
            if is_malignant:
                confidence = raw_prediction * 100
                st.markdown(f"""
                    <div class='result-card-malignant'>
                        <div class='result-title result-title-malignant'>Malignant (Skin Cancer)</div>
                        <p style='margin:0; color:#7f1d1d;'>Model Confidence Score: <b>{confidence:.2f}%</b></p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                confidence = (1 - raw_prediction) * 100
                st.markdown(f"""
                    <div class='result-card-benign'>
                        <div class='result-title result-title-benign'>Benign Lesion</div>
                        <p style='margin:0; color:#14532d;'>Model Confidence Score: <b>{confidence:.2f}%</b></p>
                    </div>
                """, unsafe_allow_html=True)
                
            st.write("")
            st.metric(label="Raw Prediction Score", value=f"{raw_prediction:.4f}")

# Medical & Academic Disclaimer
st.markdown("""
    <div class='disclaimer-box'>
        <b>Academic Disclaimer:</b> This software application was developed for the GET 324 laboratory mini-project. It is intended solely for academic demonstration and system evaluation. It is not designed or certified for clinical diagnosis.
    </div>
""", unsafe_allow_html=True)
