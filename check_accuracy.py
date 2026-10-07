from tensorflow.keras.models import load_model
import os

model_path = "models/eye_model.keras"
if not os.path.exists(model_path):
    model_path = "models/best_eye_model.keras"

print(f"Checking {model_path}...")
model = load_model(model_path)
model.summary()

# Try to get accuracy from model history if saved
print("\n--- Model Loaded Successfully ---")
print("To get real accuracy, run Option 2")