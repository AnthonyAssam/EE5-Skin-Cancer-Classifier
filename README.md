# EE5: Skin Cancer vs. Benign Tumor Classification Application

An AI-powered web application for binary dermatoscopic image classification, built using MobileNetV2 Transfer Learning and deployed via Streamlit Community Cloud.

Developed for the GET 324: Cloud Computing and AI Model Deployment for Engineering Applications Laboratory Mini-Project.

---

## Project Overview
* Group Code: EE5 (Electrical & Electronics Engineering)
* Target Task: Skin Cancer (Malignant) vs. Benign Tumors
* Model Architecture: MobileNetV2 (Fine-Tuned Transfer Learning)
* Frameworks: TensorFlow / Keras, Streamlit, Pillow
* Dataset Source: ISIC / HAM10000 Dermatoscopic Imagery

---

## Repository Structure
```text
├── .streamlit/
│   └── config.toml          # Streamlit server configurations
├── README.md                # Project documentation
├── app.py                   # Streamlit web application source code
├── requirements.txt         # Runtime python dependencies
├── runtime.txt              # Python runtime version override (3.11)
└── skin_cancer_model.h5     # Trained Keras model binary

---

## Live Web Application
https://ee5-skin-cancer-classifier.streamlit.app/
