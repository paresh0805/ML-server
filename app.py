
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import json
from PIL import Image
import io

app = Flask(__name__)
CORS(app)  # Allow requests from frontend

# === Configuration ===
MODEL_PATH = "road_classifier.h5"
CLASS_MAP_PATH = "class_indices.json"
IMG_SIZE = (224, 224)

# === Load model ===
model = load_model(MODEL_PATH)

# === Load class indices ===
with open(CLASS_MAP_PATH, "r") as f:
    class_indices = json.load(f)

# Invert class_indices to get index -> class name mapping
index_to_class = {v: k for k, v in class_indices.items()}

# === Prediction route ===
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"success": False, "message": "No image file provided"}), 400

    img_file = request.files["image"]

    try:
        # Read image into PIL format
        img = Image.open(img_file).convert("RGB")
        img = img.resize(IMG_SIZE)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Make prediction
        preds = model.predict(img_array)
        pred_index = np.argmax(preds)
        label = index_to_class[pred_index]
        confidence = float(preds[0][pred_index])

        return jsonify({
            "success": True,
            "label": label,
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Prediction error: {str(e)}"
        }), 500

# === Run the app ===
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
