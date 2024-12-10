# app.py
import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load model
model = load_model('cnn_model.h5')

# Function to predict class
def predict_image(image):
    image = image.resize((32, 32))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    predictions = model.predict(image_array)
    return np.argmax(predictions)

# Streamlit app
st.title("Image Classifier")
uploaded_file = st.file_uploader("Upload an Image", type=['png', 'jpg', 'jpeg'])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    label = predict_image(image)
    st.write(f"Predicted Class: {label}")
