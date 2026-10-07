import os
import gdown

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_eye_model.keras")
FILE_ID = "1WEv9x7V-Mzkzr4QtXlm7W5oRcTH6CjgG"

if not os.path.exists(MODEL_PATH):
    print("Downloading model from Drive...")
    gdown.download(id=FILE_ID, output=MODEL_PATH, quiet=False, fuzzy=True)
    print("Download done!")

# Import after download
from tensorflow.keras.models import load_model
print(f"Loading model: {MODEL_PATH}")
model = load_model(MODEL_PATH)
print("Model loaded!")

def predict_eye(img_path):
    from tensorflow.keras.preprocessing import image
    import numpy as np
    img = image.load_img(img_path, target_size=(224, 224))
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0) / 255.0
    return model.predict(arr)