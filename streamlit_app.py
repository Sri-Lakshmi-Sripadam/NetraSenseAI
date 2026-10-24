import streamlit as st
from PIL import Image
import os
import tempfile

from models.predict import predict_eye

st.set_page_config(page_title="NetraSenseAI", page_icon="👁️", layout="centered")

st.title("👁️ NetraSenseAI - AI Based Eye Screening")
st.markdown("AI screening for Diabetic Retinopathy")

st.divider()

uploaded_file = st.file_uploader("Front Eye Image ni Upload Chey", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("🔍 Analyze Now", type="primary"):
        with st.spinner("AI Analyzing..."):
            try:
                # FIX: Image ni temp file ga save chesi path pampadam
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                    image.save(tmp.name)
                    temp_path = tmp.name
                
                result = predict_eye(temp_path)  # ippudu path pampisthunnam
                st.success(f"**Result:** {result}")
                st.balloons()
                
                # temp file delete
                os.unlink(temp_path)
                
            except Exception as e:
                st.error(f"Error: {e}")

st.divider()
st.markdown("Built by **Sripadam Lakshmi** | [GitHub](https://github.com/Sri-Lakshmi-Sripadam/NetraSenseAI)")