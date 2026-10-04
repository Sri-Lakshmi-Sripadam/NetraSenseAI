import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image

# --- IMPORT FIX ---
try:
    from disease_info import eye_info
except ImportError:
    from models.disease_info import eye_info

# --- MODEL PATH FIX - Works from both locations ---
# Check 2 possible locations
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # project root
POSSIBLE_PATHS = [
    os.path.join(BASE_DIR, "models", "best_eye_model.keras"),
    os.path.join(BASE_DIR, "best_eye_model.keras"),
    os.path.join(os.path.dirname(__file__), "best_eye_model.keras"),
]

MODEL_PATH = None
for p in POSSIBLE_PATHS:
    if os.path.exists(p):
        MODEL_PATH = p
        break

if MODEL_PATH is None:
    print(f"ERROR: Model not found! Checked: {POSSIBLE_PATHS}")
    raise FileNotFoundError("best_eye_model.keras not found - Train first!")

print(f"[VisionSenseAI] Loading model from: {MODEL_PATH} - 93.57% Model")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

class_names = ["Cataract", "Conjunctivitis", "Eyelid", "Normal", "Uveitis"]

def predict_eye(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    preds = model.predict(img_array, verbose=0)[0]
    predicted_index = int(np.argmax(preds))
    disease = class_names[predicted_index]
    confidence = float(np.max(preds) * 100)

    print(f"\n--- PREDICTION ---")
    for i, c in enumerate(class_names):
        print(f" {c}: {preds[i]*100:.1f}%")
    print(f"-> {disease} {confidence:.1f}%\n")

    # Low confidence check
    if confidence < 55:
        sorted_idx = np.argsort(preds)[::-1]
        top3 = [(class_names[i], float(preds[i]*100)) for i in sorted_idx[:3]]
        info = {
            "description": f"AI uncertain. Confused between {top3[0][0]} ({top3[0][1]:.1f}%), {top3[1][0]} ({top3[1][1]:.1f}%). Upload close-up eye image.",
            "symptoms": [f"Top predictions: {', '.join([f'{n} {c:.1f}%' for n,c in top3])}"],
            "precautions": ["Retake close-up eye photo", "Good lighting", "No blur"],
            "consult": f"Low confidence {confidence:.1f}%. Retake photo."
        }
        return (f"Uncertain - Possible {disease}", f"{confidence:.1f}%", info["description"], info["symptoms"], info["precautions"], info["consult"])

    info = eye_info.get(disease, {"description": "No info", "symptoms": [], "precautions": [], "consult": "Consult doctor"})
    return (disease, f"{confidence:.1f}%", info["description"], info["symptoms"], info["precautions"], info["consult"])