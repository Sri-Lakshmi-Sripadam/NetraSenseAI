import streamlit as st
from PIL import Image
import os
import tempfile
import numpy as np

from models.predict import predict_eye

# Class names - Nee dataset prakaram
CLASS_NAMES = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]

st.set_page_config(page_title="NetraSenseAI", page_icon="👁️", layout="centered")
st.title("👁️ NetraSenseAI - AI Based Eye Screening")
st.markdown("Early detection of Diabetic Retinopathy using AI")
st.divider()

uploaded_file = st.file_uploader("Eye Image Upload Chey", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("🔍 Analyze Now", type="primary"):
        with st.spinner("AI Analyzing..."):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                    image.save(tmp.name)
                    temp_path = tmp.name
                
                raw_pred = predict_eye(temp_path)
                
                # Convert [[0.005, 0.43...]] -> proper label
                probs = np.array(raw_pred[0]) if isinstance(raw_pred, (list, np.ndarray)) and len(np.array(raw_pred).shape) > 1 else np.array(raw_pred)
                # Handle [[...]] case
                if probs.ndim > 1:
                    probs = probs[0]
                
                idx = int(np.argmax(probs))
                label = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else f"Class {idx}"
                confidence = float(probs[idx] * 100)
                
                if idx == 0:
                    st.success(f"### ✅ Result: {label}")
                    st.write(f"Confidence: {confidence:.2f}% - Healthy Eye")
                else:
                    st.warning(f"### ⚠️ Result: {label}")
                    st.write(f"Confidence: {confidence:.2f}% - Please consult a doctor")
                
                # Show all probabilities for transparency
                with st.expander("See Detailed Probabilities"):
                    for i, p in enumerate(probs):
                        st.write(f"{CLASS_NAMES[i]}: {p*100:.2f}%")
                
                st.balloons()
                os.unlink(temp_path)
                
            except Exception as e:
                st.error(f"Error: {e}")

st.divider()
st.markdown("Built by **Sripadam Lakshmi** | [GitHub](https://github.com/Sri-Lakshmi-Sripadam/NetraSenseAI)")
st.caption("Disclaimer: For screening purpose only, not a medical diagnosis.")