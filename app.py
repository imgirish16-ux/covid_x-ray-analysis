
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model(
    "covid_xray_resnet50.h5"
)

classes = [
    "Covid",
    "Normal",
    "Viral Pneumonia"
]

st.title("COVID-19 X-Ray Detection")

uploaded_file = st.file_uploader(
    "Upload Chest X-Ray",
    type=["jpg","jpeg","png"]
)

if uploaded_file:

    image = Image.open(uploaded_file)

    image = image.convert("RGB")

    image = image.resize((224,224))

    img = np.array(image)/255.0

    img = np.expand_dims(
        img,
        axis=0
    )

    prediction = model.predict(img)

    class_id = np.argmax(prediction)

    confidence = np.max(prediction)

    st.image(image)

    st.success(
        f"Prediction: {classes[class_id]}"
    )

    st.write(
        f"Confidence: {confidence:.2%}"
    )
