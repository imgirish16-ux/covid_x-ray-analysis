
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("covid_xray_resnet50.h5")

model = load_model()

classes = ["Covid", "Normal", "Viral Pneumonia"]

st.title("COVID-19 Chest X-Ray Classification")

uploaded_file = st.file_uploader(
    "Upload Chest X-Ray",
    type=["jpg","jpeg","png"]
)

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image")

    img = image.resize((224,224))

    img = np.array(img)/255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)

    predicted_class = classes[np.argmax(pred)]

    st.success(f"Prediction: {predicted_class}")

    st.write("Probabilities:")

    for i, c in enumerate(classes):
        st.write(f"{c}: {pred[0][i]*100:.2f}%")
