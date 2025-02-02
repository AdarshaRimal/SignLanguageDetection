import cv2
import threading
import tkinter as tk
from tkinter import Label, Button
from PIL import Image, ImageTk
from ultralytics import YOLO

# Load the trained YOLO model
model = YOLO("../models/best.pt")

# Initialize Tkinter Window
root = tk.Tk()
root.title("GestureSpeak - Real-Time Sign Language Detection")
root.geometry("800x600")
root.configure(bg="black")

# Webcam Label
video_label = Label(root, bg="black")
video_label.pack(pady=10)

# Global Variables
cap = None
running = False

def start_detection():
    """Starts the real-time detection."""
    global cap, running
    running = True
    cap = cv2.VideoCapture(1)  # Use 0 for the default webcam
    detect_gestures()

def stop_detection():
    """Stops the real-time detection."""
    global running, cap
    running = False
    if cap:
        cap.release()
        cap = None

def detect_gestures():
    """Performs real-time gesture detection and updates Tkinter window."""
    global cap, running

    if not running or cap is None:
        return

    ret, frame = cap.read()
    if not ret:
        return

    # Run YOLO inference
    results = model(frame, conf=0.5, iou=0.5)

    # Draw bounding boxes and labels
    for result in results:
        frame_with_boxes = result.plot()

    # Convert OpenCV image to Tkinter format
    frame_rgb = cv2.cvtColor(frame_with_boxes, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    img = img.resize((640, 480))  # Resize for display
    imgtk = ImageTk.PhotoImage(image=img)

    # Update Tkinter label with new frame
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    # Schedule next frame update
    if running:
        root.after(10, detect_gestures)

# Buttons for Start & Stop
btn_start = Button(root, text="Start Detection", command=start_detection, bg="green", fg="white", font=("Arial", 14))
btn_start.pack(pady=10)

btn_stop = Button(root, text="Stop Detection", command=stop_detection, bg="red", fg="white", font=("Arial", 14))
btn_stop.pack(pady=10)

# Run Tkinter main loop
root.mainloop()
