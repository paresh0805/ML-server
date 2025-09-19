import os
import json
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

# ✅ Load and flip class indices (from label → index to index → label)
with open("class_indices.json", "r") as f:
    original_indices = json.load(f)
    class_indices = {str(v): k for k, v in original_indices.items()}  # Flip mapping

# ✅ Load trained MobileNet model
MODEL_PATH = "road_classifier.h5"
model = load_model(MODEL_PATH)

# ✅ Ensure upload folder exists
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

    # ✅ Preprocess image
    img = Image.open(filepath).resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # ✅ Predict
    preds = model.predict(img_array)
    pred_class_index = int(np.argmax(preds, axis=1)[0])
    predicted_label = class_indices.get(str(pred_class_index), "Unknown")

    return jsonify({
        "prediction_index": pred_class_index,
        "prediction_label": predicted_label
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
