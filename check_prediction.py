import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model("mango_final_model.keras")

class_names = [
    "Anthracnose",
    "Bacterial Canker",
    "Healthy",
    "Scab",
    "Stem End Rot"
]

image_path = "mango-fruit-veggipedia.png"

image = Image.open(image_path).convert("RGB")
image = image.resize((224, 224))

image_array = np.array(image) / 255.0
image_array = np.expand_dims(image_array, axis=0)

predictions = model.predict(image_array, verbose=0)[0]

print("\nPrediction Probabilities:\n")

for class_name, probability in zip(class_names, predictions):
    print(f"{class_name}: {probability * 100:.2f}%")

print("\nPredicted:", class_names[np.argmax(predictions)])
print(f"Confidence: {np.max(predictions) * 100:.2f}%")