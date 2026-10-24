import streamlit as st
from PIL import Image
import numpy as np

# Nee existing predict function ne vaduthunnam
from models.predict import predict_eye

st.set_page_config(page_title="NetraSenseAI", page_icon="👁️", layout="centered")

st.title("👁️ NetraSenseAI - AI Based Eye Screening")
st.markdown("AI screening for Diabetic Retinopathy | Tested by 100+ users")

st.divider()

uploaded_file = st.file_uploader("Front Eye Image ni Upload Chey", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("🔍 Analyze Now", type="primary"):
        with st.spinner("AI Analyzing... Model loading (first time koncham time paduthundi)..."):
            try:
                # Nee predict_eye function call
                result = predict_eye(image)
                st.success(f"**Result:** {result}")
                st.balloons()
            except Exception as e:
                st.error(f"Error: {e}")
                st.info("Model first time download avuthundi, 30 sec aagi malli try chey")

st.divider()
st.markdown("Built by **Sripadam Lakshmi** | [GitHub](https://github.com/Sri-Lakshmi-Sripadam/NetraSenseAI)")