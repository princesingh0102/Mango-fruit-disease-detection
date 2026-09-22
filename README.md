# 🥭 Mango Fruit Disease Detection

A deep-learning-based web application that detects diseases in mango fruits from images using a trained **TensorFlow/Keras CNN model**.

The application allows users to upload a mango fruit image, processes the image using a Flask backend, and predicts the most likely disease along with its confidence score.

---

## 📌 Project Overview

Mango fruit diseases can significantly affect fruit quality and agricultural productivity. Manual disease identification can be time-consuming and requires domain expertise.

This project uses **Computer Vision and Deep Learning** to automatically classify mango fruit images into different disease categories.

### Workflow

```text
Mango Fruit Image
        ↓
Image Upload
        ↓
Flask Backend
        ↓
Image Preprocessing
        ↓
Trained CNN Model
        ↓
Disease Classification
        ↓
Prediction + Confidence
        ↓
Web Interface
```

---

## ✨ Features

* 🥭 Mango fruit disease detection from images
* 🤖 Deep-learning-based image classification
* 🧠 TensorFlow/Keras trained model
* 🌐 Flask web application
* 📤 Image upload functionality
* 🔍 Automatic image preprocessing
* 📊 Disease prediction with confidence score
* 🎨 Simple and responsive web interface

---

## 🦠 Disease Classes

The application is configured to classify mango fruit images into the following categories:

1. **Alternaria**
2. **Anthracnose**
3. **Bacterial Canker**
4. **Black Mould Rot**
5. **Healthy**
6. **Scab**
7. **Stem End Rot**

---

## 🛠️ Technologies Used

### Machine Learning / Deep Learning

* Python
* TensorFlow
* Keras
* NumPy
* Pillow (PIL)

### Backend

* Flask

### Frontend

* HTML
* CSS
* JavaScript

### Model

* Convolutional Neural Network (CNN)
* Keras model format (`.keras`)

---

## 📂 Project Structure

```text
Mango-fruit-disease-detection/
│
├── static/
│   ├── css/
│   │   └── style.css
│   │
│   └── js/
│       └── script.js
│
├── templates/
│   └── index.html
│
├── app.py
├── check_prediction.py
├── test_model.py
├── mango_final_model_87_21.keras
└── README.md
```

---

## 🧠 How the Model Works

The project uses a CNN-based image classification model.

A simplified CNN workflow is:

```text
Input Image
     ↓
Convolution
     ↓
Activation
     ↓
Pooling
     ↓
Convolution
     ↓
Pooling
     ↓
Feature Extraction
     ↓
Flatten
     ↓
Dense Layer
     ↓
Output Classes
```

The CNN learns visual patterns such as:

* edges
* colors
* textures
* spots
* lesions
* disease-specific patterns

and uses these learned features to classify the mango fruit.

---

## 🔄 Image Processing Pipeline

When a user uploads an image, the application processes it as follows:

### 1. Image Upload

The user selects a mango fruit image through the web interface.

### 2. Image Conversion

The image is opened using Pillow and converted to RGB format.

```python
image = Image.open(file).convert("RGB")
```

### 3. Image Resizing

The image is resized to the model's expected input size.

```python
image = image.resize((256, 256))
```

### 4. NumPy Conversion

The image is converted into a numerical array.

```python
image_array = np.array(
    image,
    dtype=np.float32
)
```

### 5. Batch Dimension

A batch dimension is added before sending the image to the model.

```python
image_array = np.expand_dims(
    image_array,
    axis=0
)
```

The resulting shape is conceptually:

```text
1 × 256 × 256 × 3
```

where:

* `1` = number of images in the batch
* `256 × 256` = image size
* `3` = RGB channels

---

## 🤖 Model Prediction

The trained Keras model is loaded using:

```python
model = tf.keras.models.load_model(
    "mango_final_model_87_21.keras"
)
```

The image is then passed to the model:

```python
predictions = model.predict(
    image_array,
    verbose=0
)[0]
```

The model produces probabilities for the disease classes.

For example:

```text
Alternaria       → 0.02
Anthracnose      → 0.87
Bacterial Canker → 0.03
Black Mould Rot  → 0.01
Healthy          → 0.04
Scab             → 0.02
Stem End Rot     → 0.01
```

The highest probability is selected using:

```python
predicted_index = int(
    np.argmax(predictions)
)
```

The corresponding class name is then obtained:

```python
predicted_class = class_names[predicted_index]
```

The confidence is calculated as:

```python
confidence = float(
    predictions[predicted_index] * 100
)
```

---

## 🌐 Flask Backend

The Flask backend provides two major routes.

### Home Route

```python
@app.route("/")
def home():
    return render_template("index.html")
```

This displays the web interface.

### Prediction Route

```python
@app.route("/predict", methods=["POST"])
def predict():
```

This receives the uploaded image, processes it, sends it to the trained model, and returns the prediction.

The response is returned as JSON:

```json
{
  "prediction": "Anthracnose",
  "confidence": 87.21
}
```

---

## 💻 Frontend Workflow

The frontend is built using HTML, CSS and JavaScript.

The user selects an image:

```html
<input
    type="file"
    id="imageInput"
    accept="image/*"
>
```

JavaScript collects the selected image:

```javascript
const file = imageInput.files[0];
```

The image is added to `FormData`:

```javascript
const formData = new FormData();

formData.append("image", file);
```

The image is then sent to Flask:

```javascript
const response = await fetch("/predict", {
    method: "POST",
    body: formData
});
```

The prediction returned by Flask is displayed on the webpage.

---

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/princesingh0102/Mango-fruit-disease-detection.git
```

### 2. Enter the Project Directory

```bash
cd Mango-fruit-disease-detection
```

### 3. Create a Virtual Environment

```bash
python3 -m venv .venv
```

### 4. Activate the Virtual Environment

#### macOS / Linux

```bash
source .venv/bin/activate
```

#### Windows

```bash
.venv\Scripts\activate
```

### 5. Install Dependencies

```bash
pip install tensorflow flask numpy pillow
```

---

## ▶️ Run the Application

Start the Flask application:

```bash
python3 app.py
```

You should see a local Flask server address in the terminal.

Open the address in your browser, for example:

```text
http://127.0.0.1:5000
```

Then:

```text
1. Upload a mango image
2. Click "Detect Disease"
3. Wait for the model prediction
4. View the predicted disease
5. View the confidence score
```

---

## 🧪 Testing the Model

The repository also contains scripts for testing model predictions.

### `test_model.py`

Used to load the trained model and test predictions.

### `check_prediction.py`

Used to check the model's prediction on an individual mango image.

---

## 📊 Model

The trained model is stored in Keras format:

```text
mango_final_model_87_21.keras
```

Keras model files can contain the trained neural-network architecture and learned weights required to perform inference.

The model can be loaded with:

```python
model = tf.keras.models.load_model(
    "mango_final_model_87_21.keras"
)
```

---

## 🔮 Future Improvements

Possible future improvements include:

* Increasing the size and diversity of the dataset
* Improving model accuracy
* Adding more mango disease categories
* Using transfer learning with architectures such as MobileNet, EfficientNet or ResNet
* Adding model performance visualizations
* Adding prediction history
* Deploying the application to a cloud platform
* Adding an API for mobile applications
* Improving UI/UX
* Adding treatment or prevention information for detected diseases

---

## ⚠️ Disclaimer

This project is intended for **educational and research purposes**. Model predictions should not be considered a replacement for professional agricultural diagnosis.

---

## 👨‍💻 Author

**Prince Singh**

B.Tech — Computer Science & Engineering

GitHub:
https://github.com/princesingh0102

---

## ⭐ If You Like This Project

If this project helped you or you found it interesting, consider giving the repository a ⭐ on GitHub.

---

## 📜 License

This project is intended for educational purposes. Add an appropriate open-source license if you plan to distribute or modify the project publicly.
