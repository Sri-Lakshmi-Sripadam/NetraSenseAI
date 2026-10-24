import os
import gdown
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_eye_model.keras")
FILE_ID = "1WEv9x7V-Mzkzr4QtXlm7W5oRcTH6CjgG"

if not os.path.exists(MODEL_PATH):
    print("Downloading model from Drive...")
    gdown.download(id=FILE_ID, output=MODEL_PATH, quiet=False, fuzzy=True)
    print("Download done!")

from tensorflow.keras.models import load_model
print(f"Loading model: {MODEL_PATH}")
# KERAS 3 FIX
model = load_model(MODEL_PATH, safe_mode=False, compile=False)
print("Model loaded!")

CLASS_DETAILS = {
    0: {"name": "Normal Eye", "desc": "Your eye appears healthy.", "symptoms": "No symptoms. Clear vision.", "precautions": "Regular checkup every 6 months, Vitamin A foods.", "consult": "No immediate consultation needed."},
    1: {"name": "Cataract", "desc": "Clouding of lens causing blurred vision.", "symptoms": "Blurred Vision, Sensitivity to Light.", "precautions": "Wear sunglasses, Quit smoking, Regular exams.", "consult": "Consult Ophthalmologist."},
    2: {"name": "Diabetic Retinopathy", "desc": "Retina damage due to diabetes.", "symptoms": "Blurred Vision, Floaters.", "precautions": "Control blood sugar, Regular retina screening.", "consult": "Consult Retina Specialist urgently."},
    3: {"name": "Glaucoma", "desc": "Increased pressure damaging optic nerve.", "symptoms": "Eye Pain, Redness, Headache.", "precautions": "Regular pressure check, Avoid eye strain.", "consult": "Consult Glaucoma Specialist immediately."},
}

def predict_eye(img_path):
    from tensorflow.keras.preprocessing import image
    img = image.load_img(img_path, target_size=(224, 224))
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0) / 255.0
    preds = model.predict(arr)
    probs = preds[0]
    idx = int(np.argmax(probs))
    conf = float(np.max(probs) * 100)
    if idx >= len(CLASS_DETAILS):
        idx = idx % len(CLASS_DETAILS)
    info = CLASS_DETAILS.get(idx, CLASS_DETAILS[0])
    return info["name"], f"{conf:.2f}%", info["desc"], info["symptoms"], info["precautions"], info["consult"]