from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os


MODEL_PATH = r"C:\Users\Zoya\Desktop\realorfake detection\ai_detector_1100.h5"
IMG_PATH = r"C:\Users\Zoya\Desktop\realorfake detection\dataset\ai\01TLT6JPSK.jpg"
TARGET_SIZE = (128, 128)  


if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
model = load_model(MODEL_PATH)
print("Model loaded successfully.")

 
if not os.path.exists(IMG_PATH):
    raise FileNotFoundError(f"Image file not found: {IMG_PATH}")

img = image.load_img(IMG_PATH, target_size=TARGET_SIZE)
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0) / 255.0  


prediction = model.predict(x)
prob = prediction[0][0]  


result = "Real" if prob >= 0.5 else "Fake"
print(f"Prediction: {result} (Probability: {prob:.4f})")
