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

## Group Members

1. UKERE, SAMUEL ABASI-UBONG  22/EG/EE/2042

2. ESSIEN, SAMUEL MICHAEL 22/EG/EE/2012

3. OLUPINLA, RICHARD AYOKANMI 22/EG/EE/1972

4. UDOKANG, GOSPEL UWEM  22/EG/EE/2072
 
5. ASSAM, ISRAEL ANTHONY 22/EG/EE/2002

6. AYANG GREATMAN OKPOKAM  22/EG/EE/1992

7. ⁠OKON UDUAKOBONG UKEME 22/EG/EE/2062

8. ARCHIBONG, IRENE ANTHONY 22/EG/EE/1962

9.  OMONRIBHOR, PRAISE OSAGIE 22/EG/EE/2082

10. IDARAOBONG EMMANUEL EDET 22/EG/EE/1982

11. EKEREKE IDONGESIT SUNDAY  22/EG/EE/2092

12. UDOKA MFONISO NSENAM 22/EG/EE/2102
    

---

## Repository Structure
```text
├── .streamlit/
│   └── config.toml          # Streamlit server configurations
├── CONTRIBUTORS.md          # Group member names and contribution logs
├── README.md                # Project documentation
├── app.py                   # Streamlit web application source code
├── requirements.txt         # Runtime python dependencies
├── runtime.txt              # Python runtime version override (3.11)
└── skin_cancer_model.h5     # Trained Keras model binary
```
---

## Live Web Application
https://ee5-skin-cancer-classifier.streamlit.app/
