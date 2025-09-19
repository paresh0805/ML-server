import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Flask app
app = Flask(__name__)
CORS(app)

# Load MobileNet model
MODEL_PATH = "mobilenet_model.h5"
model = load_model(MODEL_PATH)

# Create uploads folder
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "🚀 MobileNet Image Detector API is running!"})

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # Load and preprocess image
    img = Image.open(filepath).resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    preds = model.predict(img_array)
    pred_class = np.argmax(preds, axis=1)

    return jsonify({"prediction": int(pred_class[0])})
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

