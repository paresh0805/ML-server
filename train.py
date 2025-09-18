# train.py
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

DATA_DIR = "data/"             
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5
MODEL_PATH = "road_classifier.h5"
CLASS_MAP_PATH = "class_indices.json"


datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)
train_data = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    subset="training"
)
val_data = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    subset="validation"
)


base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224,224,3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
predictions = Dense(train_data.num_classes, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])


model.fit(train_data, validation_data=val_data, epochs=EPOCHS)

# Save model and class mapping
model.save(MODEL_PATH)
print(f"✅ Model saved as {MODEL_PATH}")

with open(CLASS_MAP_PATH, "w") as f:
    json.dump(train_data.class_indices, f)
print(f"✅ Class mapping saved as {CLASS_MAP_PATH}")
