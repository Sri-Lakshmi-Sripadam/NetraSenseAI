import streamlit as st
from PIL import Image
import os
import tempfile
from models.predict import predict_eye

st.set_page_config(page_title="NetraSenseAI", page_icon="👁️", layout="wide")

st.title("👁️ NetraSenseAI - AI Based Eye Screening")
st.markdown("Your original project logic")
st.divider()

uploaded_file = st.file_uploader("Eye Image Upload Chey (JPG/PNG)", type=["jpg","jpeg","png","webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=350)
    
    if st.button("🔍 Analyze Now", type="primary"):
        with st.spinner("AI Analyzing... Please wait..."):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                    image.save(tmp.name)
                    temp_path = tmp.name

                # CORRECT CALL - Nee Flask app lo la 6 values
                prediction, confidence, description, symptoms, precautions, consult = predict_eye(temp_path)
                
                try:
                    conf_val = float(str(confidence).replace('%','').strip())
                except:
                    conf_val = 90.0

                st.divider()
                st.subheader("🔬 AI Detection Result")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Prediction", prediction)
                with col2:
                    st.metric("Confidence", confidence)

                st.progress(int(conf_val) if conf_val <=100 else 90)

                st.write(f"**Description:** {description}")
                
                with st.expander("🩺 Symptoms", expanded=True):
                    st.write(symptoms)
                with st.expander("🛡️ Precautions", expanded=True):
                    st.write(precautions)
                with st.expander("👨‍⚕️ Doctor Consultation", expanded=True):
                    st.write(consult)

                st.balloons()
                os.unlink(temp_path)

            except Exception as e:
                st.error(f"Prediction Error: {e}")
                st.info("models/predict.py lo emundo check chey")

st.divider()
st.caption("Built by Sripadam Lakshmi | Same logic as Flask app")