import tensorflow as tf
import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize data

# Build a simple neural network model
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# Function to make predictions
def predict_digit(img):
    img = img.convert('L')  # Convert to grayscale
    img = img.resize((28, 28))  # Resize to 28x28
    img = np.array(img) / 255.0  # Normalize pixel values (0-1)
    img = 1 - img  # Invert colors (MNIST is white on black)

    if img.shape != (28, 28):
        return "Error: Image shape mismatch!"

    img = img.reshape(1, 28, 28)  # Reshape for the model
    prediction = model.predict(img)

    return {str(i): float(prediction[0][i]) for i in range(10)}


# Create Gradio interface
iface = gr.Interface(fn=predict_digit,
                     inputs=gr.Image(image_mode='L'),
                     outputs=gr.Label())

# Launch the interface
iface.launch()
