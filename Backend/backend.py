from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
from ultralytics import YOLO
import cv2
import os

# =============================================
# Custom Layer Definitions for Model Compatibility
# =============================================
from tensorflow.keras.layers import InputLayer, BatchNormalization
from tensorflow.keras.layers.experimental import SyncBatchNormalization
from tensorflow.keras.mixed_precision import Policy


class CustomInputLayer(InputLayer):
    @classmethod
    def from_config(cls, config):
        if 'batch_shape' in config:
            config['batch_input_shape'] = config.pop('batch_shape')
        return super().from_config(config)


class CustomBatchNormalization(BatchNormalization):
    @classmethod
    def from_config(cls, config):
        config.pop('synchronized', None)
        return super().from_config(config)


class CustomSyncBatchNormalization(SyncBatchNormalization):
    @classmethod
    def from_config(cls, config):
        config.pop('synchronized', None)
        return super().from_config(config)


CUSTOM_OBJECTS = {
    'InputLayer': CustomInputLayer,
    'BatchNormalization': CustomBatchNormalization,
    'SyncBatchNormalization': CustomSyncBatchNormalization,
    'DTypePolicy': Policy,
}

# =============================================
# Flask Application Setup
# =============================================
app = Flask(__name__)
CORS(app)

# =============================================
# Model Loading with Compatibility Fixes
# =============================================
try:
    # Load Custom CNN Model
    cnn_model = tf.keras.models.load_model(
        "C:\\Users\\Adarsha Rimal\\Desktop\\sign_language_detection\\others_models\\scratch_cnn.h5",
        custom_objects=CUSTOM_OBJECTS,
        compile=False
    )
    cnn_model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Load EfficientNet Model
    efficientnet_model = tf.keras.models.load_model("../models/model_keras.h5")
    efficientnet_model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Load YOLO Model
    yolo_model = YOLO("../models/best.pt")

except Exception as e:
    print(f"Error loading models: {str(e)}")
    exit(1)

# =============================================
# Constants and Configuration
# =============================================
CLASS_LABELS = [
    "fight", "give", "hats_off", "heart", "hello",
    "help", "i_love_you", "namaste", "no", "perfect",
    "please", "silent", "sorry", "stop", "thank_you",
    "water", "yes"
]


# =============================================
# Image Preprocessing Functions
# =============================================
def preprocess_image(file, model_type="cnn"):
    """Preprocess image based on model requirements"""
    img = Image.open(file).convert('RGB')

    if model_type == "cnn":
        # For custom CNN (480 height, 640 width)
        img = img.resize((640, 480), Image.LANCZOS)  # Width, Height order
        img_array = np.array(img)[None, ...] / 255.0
    elif model_type == "efficientnet":
        # For EfficientNetB0
        img = img.resize((300, 300), Image.LANCZOS)
        img_array = np.array(img)[None, ...]

    return img_array


# =============================================
# API Endpoints
# =============================================
@app.route('/classify-image', methods=['POST'])
def classify_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    model_type = request.form.get('model_type', 'cnn').lower()

    try:
        processed_img = preprocess_image(file, model_type)

        if model_type == "cnn":
            predictions = cnn_model.predict(processed_img)[0]
        elif model_type == "efficientnet":
            predictions = efficientnet_model.predict(processed_img)[0]
        else:
            return jsonify({'error': 'Invalid model type'}), 400

        class_idx = np.argmax(predictions)
        confidence = np.max(predictions)

        return jsonify({
            'model': model_type,
            'class': CLASS_LABELS[class_idx],
            'confidence': float(confidence)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/detect-gesture', methods=['GET'])
def detect_gesture():
    def generate_frames():
        cap = cv2.VideoCapture(0)
        while True:
            success, frame = cap.read()
            if not success:
                break

            results = yolo_model(frame, conf=0.5, iou=0.5)
            annotated_frame = results[0].plot()

            _, buffer = cv2.imencode('.jpg', annotated_frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

        cap.release()

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# =============================================
# Utility Endpoints
# =============================================
@app.route('/gesture-info/<gesture_id>', methods=['GET'])
def gesture_info(gesture_id):
    gesture_details = {
        "hello": "A common greeting gesture using an open palm wave.",
        "thank_you": "Hand movement from chin outward to express gratitude.",
        # Add more gesture descriptions
    }
    return jsonify(gesture_details.get(gesture_id.lower(), "Gesture information not available."))


@app.route('/test', methods=['GET'])
def health_check():
    return jsonify({
        "status": "active",
        "models_loaded": {
            "cnn": cnn_model is not None,
            "efficientnet": efficientnet_model is not None,
            "yolo": yolo_model is not None
        }
    })


# =============================================
# Security Headers
# =============================================
@app.after_request
def add_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['Cache-Control'] = 'no-store, max-age=0'
    return response


# =============================================
# Application Entry Point
# =============================================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)