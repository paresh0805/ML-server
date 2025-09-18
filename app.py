from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import json
import io
from PIL import Image

app = Flask(__name__)
CORS(app)  # allow requests from frontend

# Load ML model and class mapping once at startup
MODEL_PATH = "road_classifier.h5"  # rename your file accordingly
CLASS_MAP_PATH = "class_indices.json"

model = load_model(MODEL_PATH)

with open(CLASS_MAP_PATH, "r") as f:
    class_indices = json.load(f)
index_to_class = {v: k for k, v in class_indices.items()}  # safer mapping

IMG_SIZE = (224, 224)

# Dummy users for testing login
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
        label = index_to_class[pred_index]
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
