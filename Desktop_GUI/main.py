import cv2
import threading
import tkinter as tk
from tkinter import Label, Button, filedialog, ttk, messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO
import pyttsx3
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input

# Load models
yolo_model = YOLO("../models/best.pt")

#to resolve cnn model incompatability issue
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
efficientnet_model = tf.keras.models.load_model("../models/model_keras.h5")

# Compile CNN models

efficientnet_model.compile(optimizer=tf.keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Debug: Print model architectures
print("Scratch CNN Model Summary:")
cnn_model.summary()

print("\nEfficientNet Model Summary:")
efficientnet_model.summary()

# Class labels
classes = [
    "fight", "give", "hats_off", "heart", "hello",
    "help", "i_love_you", "namaste", "no", "perfect",
    "please", "silent", "sorry", "stop", "thank_you",
    "water", "yes"
]
print("\nClass Indices:", classes)

# Initialize Tkinter Window
root = tk.Tk()
root.title("GestureSpeak - Sign Language Detection")
root.geometry("800x600")
root.configure(bg="black")

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Global Variables
cap = None
running = False
last_spoken = ""
selected_model = tk.StringVar(value="YOLO")

# Webcam Label
video_label = Label(root, bg="black")
video_label.pack(pady=10)

# Prediction Label
prediction_label = Label(root, text="", bg="black", fg="white", font=("Arial", 20))
prediction_label.pack(pady=10)


# Function to speak detected gesture
def speak(text):
    engine.say(text)
    engine.runAndWait()


# Start detection for YOLO
def start_detection():
    global cap, running
    if selected_model.get() == "YOLO":
        running = True
        cap = cv2.VideoCapture(1)
        detect_gestures()
    else:
        messagebox.showinfo("Info", "Real-time detection is only available for YOLO model.")


# Stop detection
def stop_detection():
    global running, cap
    running = False
    if cap:
        cap.release()
        cap = None


# YOLO real-time detection
def detect_gestures():
    global cap, running, last_spoken

    if not running or cap is None:
        return

    ret, frame = cap.read()
    if not ret:
        return

    results = yolo_model(frame, conf=0.3, iou=0.5)
    for result in results:
        frame_with_boxes = result.plot()
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = yolo_model.names[class_id]
            if class_name != last_spoken:
                threading.Thread(target=speak, args=(class_name,)).start()
                last_spoken = class_name
                prediction_label.config(text=f"Detected: {class_name}")

    frame_rgb = cv2.cvtColor(frame_with_boxes, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb).resize((640, 480))
    imgtk = ImageTk.PhotoImage(image=img)

    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    if running:
        root.after(10, detect_gestures)


# Load image for CNN models
def load_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path).convert('RGB')
        predict_image(img)


# Capture image from webcam for CNN models
def capture_image():
    cap = cv2.VideoCapture(1)
    ret, frame = cap.read()
    cap.release()
    if ret:
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        predict_image(img)


# Predict using selected CNN model
def predict_image(img):
    try:
        # Select the model based on the user's choice
        model = cnn_model if selected_model.get() == "Scratch CNN" else efficientnet_model

        # Resize the image based on the selected model's input size
        if selected_model.get() == "Scratch CNN":
            img_resized = img.resize((640, 480), Image.LANCZOS)
            img_array = np.array(img_resized)[None, ...]
            img_array = img_array / 255.0  # Normalize for Scratch CNN
        else:
            img_resized = img.resize((300, 300), Image.LANCZOS)
            img_array = np.array(img_resized)[None, ...]
            img_array = preprocess_input(img_array)  # EfficientNet-specific preprocessing

        # Debug: Print raw model outputs
        class_scores = model.predict(img_array)
        print("Raw Model Outputs:", class_scores)

        # Get the predicted class
        predicted_class_idx = class_scores.argmax()
        predicted_class = classes[predicted_class_idx]

        # Debug: Print confidence
        confidence = class_scores[0][predicted_class_idx]
        print(f"Predicted Class: {predicted_class} (Index: {predicted_class_idx}, Confidence: {confidence:.2f})")

        # Display the image
        img_display = img.resize((640, 480))
        imgtk = ImageTk.PhotoImage(image=img_display)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)

        # Update the prediction label
        prediction_label.config(text=f"Predicted: {predicted_class}")
        threading.Thread(target=speak, args=(predicted_class,)).start()

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred during prediction: {str(e)}")


# UI Elements
model_options = ttk.Combobox(root, textvariable=selected_model, values=["YOLO", "Scratch CNN", "EfficientNet CNN"],
                             font=("Arial", 14))
model_options.pack(pady=10)

btn_start = Button(root, text="Start Detection (YOLO)", command=start_detection, bg="green", fg="white",
                   font=("Arial", 14))
btn_start.pack(pady=5)

btn_stop = Button(root, text="Stop Detection", command=stop_detection, bg="red", fg="white", font=("Arial", 14))
btn_stop.pack(pady=5)

btn_upload = Button(root, text="Upload Image (CNN)", command=load_image, bg="blue", fg="white", font=("Arial", 14))
btn_upload.pack(pady=5)

btn_capture = Button(root, text="Capture Image (CNN)", command=capture_image, bg="purple", fg="white",
                     font=("Arial", 14))
btn_capture.pack(pady=5)

# Run Tkinter main loop
root.mainloop()