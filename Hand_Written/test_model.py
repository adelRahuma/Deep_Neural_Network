import tensorflow as tf
import numpy as np
import cv2  # OpenCV for image processing
import matplotlib.pyplot as plt

# Load the trained model
model = tf.keras.models.load_model("mymodel.h5")  # Ensure the correct path

# Load the new image
image_path = "number.jpg"  # Replace with your image filename
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale
image = cv2.resize(image, (28, 28))  # Resize to 28x28 pixels

# Normalize pixel values
image = image / 255.0

# Expand dimensions to match model input (1, 28, 28)
image = np.expand_dims(image, axis=0)

# Show the processed image
plt.imshow(image[0], cmap='gray')
plt.title("Processed Image")
plt.axis("off")
plt.show()

# Make a prediction
prediction = model.predict(image)  # Get model predictions
# print(prediction)
predicted_digit = np.argmax(prediction)  # Get the digit with the highest probability

print(f"Predicted Digit: {predicted_digit}")
