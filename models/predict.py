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
model = load_model(MODEL_PATH)
print("Model loaded!")

# --- NEE CLASSES - Nee model output batti ---
CLASS_DETAILS = {
    0: {
        "name": "Normal Eye",
        "desc": "Your eye appears healthy with no signs of disease.",
        "symptoms": "No symptoms. Clear vision.",
        "precautions": "Regular eye checkup every 6 months, Eat Vitamin A rich foods, Avoid excessive screen time.",
        "consult": "No immediate consultation needed. Maintain regular checkups."
    },
    1: {
        "name": "Cataract",
        "desc": "Clouding of the eye lens causing blurred vision.",
        "symptoms": "Blurred Vision, Sensitivity to Light, Difficulty in Night Vision.",
        "precautions": "Wear sunglasses, Quit smoking, Manage diabetes, Regular eye exams.",
        "consult": "Consult Ophthalmologist immediately for cataract evaluation."
    },
    2: {
        "name": "Diabetic Retinopathy",
        "desc": "Damage to retina due to diabetes, can lead to blindness.",
        "symptoms": "Blurred Vision, Floaters, Dark areas in vision.",
        "precautions": "Control blood sugar, Control BP, Regular retina screening, Healthy diet.",
        "consult": "Consult Retina Specialist urgently. Diabetic eye checkup needed."
    },
    3: {
        "name": "Glaucoma",
        "desc": "Increased pressure in eye damaging optic nerve.",
        "symptoms": "Eye Pain, Redness, Headache, Blurred Vision.",
        "precautions": "Regular eye pressure check, Avoid eye strain, Take prescribed drops regularly.",
        "consult": "Consult Glaucoma Specialist immediately."
    },
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
    
    # Handle if model has more classes than our dict
    if idx >= len(CLASS_DETAILS):
        idx = idx % len(CLASS_DETAILS)
        
    info = CLASS_DETAILS.get(idx, CLASS_DETAILS[0])
    
    prediction = info["name"]
    confidence = f"{conf:.2f}%"
    
    return prediction, confidence, info["desc"], info["symptoms"], info["precautions"], info["consult"]