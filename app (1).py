import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model("diabetic_retinopathy_model_final.h5")

classes = [
    "No Diabetic Retinopathy",
    "Mild DR",
    "Moderate DR",
    "Severe DR",
    "Proliferative DR"
]

st.title("Diabetic Retinopathy Detection")
st.write("Upload a retinal fundus image to predict the DR stage.")

uploaded_file = st.file_uploader(
    "Choose a retinal image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_id = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    st.success(f"Prediction: {classes[class_id]}")
    st.info(f"Confidence: {confidence:.2f}%")
