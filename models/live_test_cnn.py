import cv2
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('mobilenetv2_sign_language_model.h5')

# Define the class labels
classes = ["hello", "i_love_you", "namaste", "silent", "thank_you", "fight", "heart", "please", "perfect", "give", "help", "sorry", "water"]

# Function to resize image while maintaining aspect ratio
def resize_with_aspect_ratio(image, target_size):
    h, w = image.shape[:2]
    scale = min(target_size[0] / h, target_size[1] / w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized_image = cv2.resize(image, (new_w, new_h))

    # Add padding to make it target size
    pad_top = (target_size[0] - new_h) // 2
    pad_bottom = target_size[0] - new_h - pad_top
    pad_left = (target_size[1] - new_w) // 2
    pad_right = target_size[1] - new_w - pad_left

    padded_image = cv2.copyMakeBorder(
        resized_image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )
    return padded_image

# Start webcam capture
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        break

    # Preprocess the frame
    frame_resized = resize_with_aspect_ratio(frame, (150, 150))  # Resize to 224x224
    frame_normalized = frame_resized / 255.0  # Normalize
    frame_input = np.expand_dims(frame_normalized, axis=0)  # Add batch dimension

    # Make predictions
    predictions = model.predict(frame_input)
    predicted_class = np.argmax(predictions)  # Get class with highest probability
    predicted_label = classes[predicted_class]  # Get the sign language label

    # Display the frame with predicted label
    cv2.putText(frame, predicted_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow('Live Sign Detection', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
