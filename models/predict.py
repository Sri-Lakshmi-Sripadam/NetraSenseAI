import os
import gdown

# Model path
MODEL_PATH = "models/best_eye_model.keras"
# Nee Drive File ID ikkada pettu
FILE_ID = "1AbCdEfGhIJKL... <- ikkada nee ID pettu"

# Model lekapothe download chey
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Drive...")
    os.makedirs("models", exist_ok=True)
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)
    print("Model downloaded!")

# Tarvata nee normal load code
# model = load_model(MODEL_PATH)