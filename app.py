from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import json
from PIL import Image
import os

app = Flask(__name__)
CORS(app)

MODEL_PATH = "road_classifier.h5"  # or folder if SavedModel
CLASS_MAP_PATH = "class_indices.json"
IMG_SIZE = (224, 224)

# Safe model loading
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = load_model(MODEL_PATH)
        print(f"Model loaded from {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")
else:
    print(f"Model file not found at {MODEL_PATH}")

# Load class mapping safely
index_to_class = {}
if os.path.exists(CLASS_MAP_PATH):
    with open(CLASS_MAP_PATH, "r") as f:
        class_indices = json.load(f)
    index_to_class = {v: k for k, v in class_indices.items()}
else:
    print(f"Class mapping file not found at {CLASS_MAP_PATH}")

# Dummy users
users = [
    {"email": "test@example.com", "password": "123456", "department": "water"},
    {"phone": "9876543210", "password": "123456", "department": "electricity"},
]

@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    email = data.get("email")
    phone = data.get("phone")
    password = data.get("password")

    if not password:
        return jsonify({"success": False, "message": "Password is required"}), 400

    if email:
        user = next((u for u in users if u.get("email") == email and u["password"] == password), None)
        if user:
            return jsonify({"success": True, "message": "Login successful"})

    if phone:
        user = next((u for u in users if u.get("phone") == phone and u["password"] == password), None)
        if user:
            return jsonify({"success": True, "message": "Login successful"})

    return jsonify({"success": False, "message": "Invalid credentials"}), 401

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"success": False, "message": "Model not loaded"}), 500

    if "image" not in request.files:
        return jsonify({"success": False, "message": "No image file provided"}), 400
    
    img_file = request.files["image"]
    
    try:
        img = Image.open(img_file).convert("RGB")
        img = img.resize(IMG_SIZE)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        preds = model.predict(img_array)
        pred_index = np.argmax(preds)
        label = index_to_class.get(pred_index, "Unknown")
        confidence = float(preds[0][pred_index])

        return jsonify({
            "success": True,
            "label": label,
            "confidence": confidence
        })
    except Exception as e:
        return jsonify({"success": False, "message": f"Prediction error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
