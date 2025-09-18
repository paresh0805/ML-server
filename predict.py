# predict.py
import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import sys

MODEL_PATH = "road_classifier.h5"
CLASS_MAP_PATH = "class_indices.json"
IMG_SIZE = (224, 224)

# Load model
model = load_model(MODEL_PATH)

# Load class mapping
with open(CLASS_MAP_PATH, "r") as f:
    class_indices = json.load(f)
classes = list(class_indices.keys())

def predict_image(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)
    label = classes[np.argmax(pred)]
    return label, pred[0]

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ Usage: python predict.py <image_path>")
        sys.exit(1)

    img_path = sys.argv[1]
    label, probs = predict_image(img_path)

    print("Prediction:", label)
    print("Probabilities:", probs)
