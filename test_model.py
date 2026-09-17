import tensorflow as tf
import numpy as np
from PIL import Image


# Load the trained model
model = tf.keras.models.load_model("mango_final_model.keras")


# Class names must match the order used during training
class_names = [
    "Anthracnose",
    "Bacterial Canker",
    "Healthy",
    "Scab",
    "Stem End Rot"
]


def predict_image(image_path):

    # Open the image
    image = Image.open(image_path).convert("RGB")

    # Resize image to the model's required input size
    image = image.resize((224, 224))

    # Convert image into NumPy array
    image_array = np.array(image)

    # Normalize pixel values from 0-255 to 0-1
    image_array = image_array / 255.0

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # Make prediction
    predictions = model.predict(image_array, verbose=0)

    # Find class with highest probability
    predicted_index = np.argmax(predictions[0])

    # Get predicted class
    predicted_class = class_names[predicted_index]

    # Get confidence
    confidence = predictions[0][predicted_index] * 100

    return predicted_class, confidence