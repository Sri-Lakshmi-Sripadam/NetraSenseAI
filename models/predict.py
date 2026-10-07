import os
import gdown
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_eye_model.keras")
FILE_ID = "1WEv9x7V-Mzkzr4QtXlm7W5oRcTH6CjgG"

if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)

model = load_model(MODEL_PATH)

def predict_eye(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return model.predict(img_array)