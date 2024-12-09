import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Enable memory growth for GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Function to resize images while maintaining aspect ratio
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

# Load dataset (replace with actual dataset path)
image_dir = "C:\\Users\\Adarsha Rimal\\Desktop\\sign_language_detection\\dataset"
classes = ["hello", "i_love_you", "namaste", "silent", "thank_you", "fight", "heart", "please", "perfect", "give", "help", "sorry", "water"]
data, labels = [], []

# Load images and labels
for label, class_name in enumerate(classes):
    class_dir = os.path.join(image_dir, class_name)
    for img_name in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            resized_img = resize_with_aspect_ratio(img, (150, 150))
            data.append(resized_img)
            labels.append(label)

# Convert to numpy arrays and normalize
X = np.array(data, dtype="float32") / 255.0
y = np.array(labels)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# One-hot encoding
num_classes = len(classes)
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

# Load pre-trained MobileNetV2 model
base_model = tf.keras.applications.MobileNetV2(input_shape=(150, 150, 3),
                                               include_top=False,  # Exclude the top layer
                                               weights='imagenet')  # Load ImageNet weights

# Freeze the base model
base_model.trainable = False

# Build the custom model
model = models.Sequential([
    base_model,  # Add the pre-trained MobileNetV2 model
    layers.GlobalAveragePooling2D(),  # Global Average Pooling
    layers.Dense(128, activation='relu'),  # Custom dense layer
    layers.Dense(num_classes, activation='softmax')  # Output layer with softmax for multi-class classification
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)

# Save the model
model.save("mobilenetv2_sign_language_model.h5")

# Plot training & validation accuracy
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plot training & validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Evaluate the model
eval_results = model.evaluate(X_test, y_test)
print(f"Test Loss: {eval_results[0]}\nTest Accuracy: {eval_results[1]}")
