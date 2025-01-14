from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

# Initialize the Flask app
app = Flask(__name__)
CORS(app)

# Load models
cnn_model = tf.keras.models.load_model("../models/sign_language_cnn_model.h5")
efficientnet_model = tf.keras.models.load_model("../models/model_keras.h5")
# Placeholder for YOLO model
yolo_model = None  # Replace with actual YOLO model loading when available

# Compile TensorFlow models
cnn_model.compile(optimizer=tf.keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
efficientnet_model.compile(optimizer=tf.keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Define the class labels
classes = [
    "fight", "give", "hats_off", "heart", "hello",
    "help", "i_love_you", "namaste", "no", "perfect",
    "please", "silent", "sorry", "stop", "thank_you",
    "water", "yes"
]


# Preprocessing function for images

def preprocess_image(file, model_type="cnn", size=(300, 300)):
    # Open image and convert to RGB
    img = Image.open(file).convert('RGB')

    # Resize image according to the model type
    if model_type == "cnn":
        img = img.resize((224, 224), Image.LANCZOS)  # Resize for CNN model (224x224)
    elif model_type == "efficientnet":
        img = img.resize((size[0], size[1]), Image.LANCZOS)  # Resize for EfficientNet model (300x300)

    # Convert image to numpy array and normalize
    inp_numpy = np.array(img)[None, ...]  # Add batch dimension
    inp_numpy = inp_numpy / 255.0  # Normalize pixel values to [0, 1]

    return inp_numpy


# API: Static Image Classification
@app.route('/classify-image', methods=['POST'])
def classify_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    model_type = request.form.get('model_type', 'cnn')  # Get model type (default is cnn)

    # Preprocess image based on the model type
    image = preprocess_image(file, model_type=model_type)

    # Predict using the chosen model
    if model_type == "cnn":
        predictions = cnn_model.predict(image)[0]
    elif model_type == "efficientnet":
        predictions = efficientnet_model.predict(image)[0]

    class_idx = np.argmax(predictions)
    confidence = predictions[class_idx]

    return jsonify({
        'model': model_type,
        'class': classes[class_idx],
        'confidence': float(confidence)
    })


# API: Real-Time Gesture Detection (YOLO Placeholder)
@app.route('/detect-gesture', methods=['GET'])
def detect_gesture():
    def generate_frames():
        cap = cv2.VideoCapture(0)  # Open webcam
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # TODO: Add YOLO inference logic here
            # For now, just display raw frames
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        cap.release()

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# API: Gesture Information
@app.route('/gesture-info/<gesture_id>', methods=['GET'])
def gesture_info(gesture_id):
    gesture_details = {
        "hello": "A common greeting gesture.",
        "thank_you": "A sign of gratitude.",
        "i_love_you": "A gesture to express love.",
        # Add details for other gestures...
    }

    info = gesture_details.get(gesture_id.lower(), "Gesture information not found.")
    return jsonify({'gesture': gesture_id, 'info': info})


# API: Compare Models
@app.route('/compare-models', methods=['POST'])
def compare_models():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    model_type = request.form.get('model_type', 'cnn')  # Get model type (default is cnn)

    # Preprocess image based on the model type
    image = preprocess_image(file, model_type=model_type)

    # Predict using CNN
    cnn_predictions = cnn_model.predict(image)[0]
    cnn_class_idx = np.argmax(cnn_predictions)
    cnn_confidence = cnn_predictions[cnn_class_idx]

    # Predict using EfficientNet
    efficientnet_predictions = efficientnet_model.predict(image)[0]
    efficientnet_class_idx = np.argmax(efficientnet_predictions)
    efficientnet_confidence = efficientnet_predictions[efficientnet_class_idx]

    return jsonify({
        "model_comparison": {
            "cnn": {
                "class": classes[cnn_class_idx],
                "confidence": float(cnn_confidence)
            },
            "efficientnet": {
                "class": classes[efficientnet_class_idx],
                "confidence": float(efficientnet_confidence)
            }
        }
    })


# API: Test Endpoint
@app.route('/test', methods=['GET'])
def test():
    return jsonify({"status": "OK", "message": "Backend is running!"})


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
