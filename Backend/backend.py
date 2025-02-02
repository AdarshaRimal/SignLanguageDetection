from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
from ultralytics import YOLO

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Allow cross-origin requests
# Load CNN and EfficientNet models
cnn_model = tf.keras.models.load_model("../models/sign_language_cnn_model.h5")
efficientnet_model = tf.keras.models.load_model("../models/model_keras.h5")

# Compile TensorFlow models
cnn_model.compile(optimizer=tf.keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
efficientnet_model.compile(optimizer=tf.keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Load YOLO model for real-time gesture detection
yolo_model = YOLO("../models/best.pt")

# Define class labels
classes = [
    "fight", "give", "hats_off", "heart", "hello",
    "help", "i_love_you", "namaste", "no", "perfect",
    "please", "silent", "sorry", "stop", "thank_you",
    "water", "yes"
]

# ========================= IMAGE PREPROCESSING FUNCTION =========================
def preprocess_image(file, model_type="cnn", size=(300, 300)):
    """Preprocess an image for CNN or EfficientNet models."""
    img = Image.open(file).convert('RGB')

    if model_type == "cnn":
        img = img.resize((224, 224), Image.LANCZOS)
    elif model_type == "efficientnet":
        img = img.resize((size[0], size[1]), Image.LANCZOS)

    inp_numpy = np.array(img)[None, ...] / 255.0  # Normalize image
    return inp_numpy


# ========================= STATIC IMAGE CLASSIFICATION API =========================
@app.route('/classify-image', methods=['POST'])
def classify_image():
    """Classifies a static image using CNN or EfficientNet."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    model_type = request.form.get('model_type', 'cnn')

    image = preprocess_image(file, model_type=model_type)

    if model_type == "cnn":
        predictions = cnn_model.predict(image)[0]
    elif model_type == "efficientnet":
        predictions = efficientnet_model.predict(image)[0]
    else:
        return jsonify({'error': 'Invalid model type'}), 400

    class_idx = np.argmax(predictions)
    confidence = predictions[class_idx]

    return jsonify({
        'model': model_type,
        'class': classes[class_idx],
        'confidence': float(confidence)
    })


# ========================= REAL-TIME GESTURE DETECTION API (YOLO) =========================
@app.route('/detect-gesture', methods=['GET'])
def detect_gesture():
    """Streams real-time gesture detection using YOLOv8."""
    def generate_frames():
        cap = cv2.VideoCapture(0)  # Open webcam
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Run YOLO model for gesture detection
            results = yolo_model(frame, conf=0.5, iou=0.5)
            for result in results:
                frame_with_boxes = result.plot()  # Draw detections

            _, buffer = cv2.imencode('.jpg', frame_with_boxes)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        cap.release()

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# ========================= GESTURE INFORMATION API =========================
@app.route('/gesture-info/<gesture_id>', methods=['GET'])
def gesture_info(gesture_id):
    """Returns information about a specific sign gesture."""
    gesture_details = {
        "hello": "A common greeting gesture.",
        "thank_you": "A sign of gratitude.",
        "i_love_you": "A gesture to express love.",
        "fight": "Indicates a conflict or struggle.",
        "heart": "Represents love and affection.",
        "water": "Sign for requesting water.",
        # Add details for all gestures...
    }

    info = gesture_details.get(gesture_id.lower(), "Gesture information not found.")
    return jsonify({'gesture': gesture_id, 'info': info})


# ========================= MODEL COMPARISON API =========================
@app.route('/compare-models', methods=['POST'])
def compare_models():
    """Compares CNN and EfficientNet predictions on the same image."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']

    # Preprocess images for both models
    cnn_image = preprocess_image(file, model_type="cnn")
    efficientnet_image = preprocess_image(file, model_type="efficientnet")

    # Predict using CNN
    cnn_predictions = cnn_model.predict(cnn_image)[0]
    cnn_class_idx = np.argmax(cnn_predictions)
    cnn_confidence = cnn_predictions[cnn_class_idx]

    # Predict using EfficientNet
    efficientnet_predictions = efficientnet_model.predict(efficientnet_image)[0]
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


# ========================= API STATUS CHECK =========================
@app.route('/test', methods=['GET'])
def test():
    """API health check."""
    return jsonify({"status": "OK", "message": "Backend is running!"})


# ========================= RUN FLASK APP =========================
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
