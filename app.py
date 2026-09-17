from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# -----------------------------
# Configuration
# -----------------------------

MODEL_PATH = "mango_final_model_87_21.keras"
UPLOAD_FOLDER = "uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -----------------------------
# Load Model
# -----------------------------

print("Loading model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("✅ Model loaded successfully")

# -----------------------------
# Class Names
# -----------------------------

class_names = [
    "Alternaria",
    "Anthracnose",
    "Bacterial Canker",
    "Black Mould Rot",
    "Healthy",
    "Scab",
    "Stem End Rot"
]

IMG_SIZE = (256, 256)

# -----------------------------
# Home Page
# -----------------------------

@app.route("/")
def home():
    return render_template("index.html")


# -----------------------------
# Prediction API
# -----------------------------

@app.route("/predict", methods=["POST"])
def predict():

    # Check image
    if "image" not in request.files:
        return jsonify({
            "error": "No image uploaded."
        }), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({
            "error": "No image selected."
        }), 400

    try:

        # Open image
        image = Image.open(file).convert("RGB")

        # Resize to model input size
        image = image.resize(IMG_SIZE)

        # Convert image to NumPy
        image_array = np.array(image, dtype=np.float32)

        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)

        # -----------------------------
        # Model Prediction
        # -----------------------------

        predictions = model.predict(
            image_array,
            verbose=0
        )[0]

        # Predicted class
        predicted_index = int(np.argmax(predictions))

        predicted_class = class_names[predicted_index]

        # Confidence
        confidence = float(
            predictions[predicted_index] * 100
        )

        # All class probabilities
        probabilities = {}

        for i, class_name in enumerate(class_names):
            probabilities[class_name] = round(
                float(predictions[i] * 100),
                2
            )

        # -----------------------------
        # Response
        # -----------------------------

        return jsonify({
            "prediction": predicted_class,
            "confidence": round(confidence, 2),
            "probabilities": probabilities
        })

    except Exception as e:

        print("Prediction error:", e)

        return jsonify({
            "error": "Unable to process the image."
        }), 500


# -----------------------------
# Run Flask
# -----------------------------

if __name__ == "__main__":

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )