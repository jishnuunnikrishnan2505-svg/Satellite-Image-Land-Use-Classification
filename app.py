import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.title("Satellite Land Use Classification")

try:
    model = tf.keras.models.load_model("eurosat_cnn_model.h5")
    st.success("Model Loaded Successfully")
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

class_names = [
    'AnnualCrop','Forest','HerbaceousVegetation','Highway',
    'Industrial','Pasture','PermanentCrop',
    'Residential','River','SeaLake'
]

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((64,64))

    st.image(image, caption="Uploaded Image")

    img_array = np.array(image)   # NO /255
    img_array = np.expand_dims(img_array, axis=0)

    # st.write("Shape:", img_array.shape)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = float(np.max(prediction))

    st.success(f"Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}")

