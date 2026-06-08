import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import gdown

MODEL_PATH = "covid_xray_resnet50.h5"

if not os.path.exists(MODEL_PATH):
    FILE_ID = "102Zke6ozKIhAflbD72F3B5XTHTzB8Cf7"
    url = f"https://drive.google.com/uc?id={FILE_ID}"

    with st.spinner("Downloading model..."):
        gdown.download(url, MODEL_PATH, quiet=False)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

classes = ["Covid", "Normal", "Viral Pneumonia"]

st.title("COVID-19 Chest X-Ray Classification")

uploaded_file = st.file_uploader(
    "Upload Chest X-Ray",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    img = image.resize((224, 224))

    img = np.array(img) / 255.0

    img = np.expand_dims(
        img,
        axis=0
    )

    pred = model.predict(img)

    predicted_class = classes[
        np.argmax(pred)
    ]

    st.success(
        f"Prediction: {predicted_class}"
    )

    st.write("### Prediction Probabilities")

    for i, c in enumerate(classes):
        st.write(
            f"{c}: {pred[0][i] * 100:.2f}%"
        )
