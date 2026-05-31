import gradio as gr
import tensorflow as tf
import numpy as np
import cv2
import pydicom

# Load model
model = tf.keras.models.load_model(
    "best_pneumonia_model.h5"
)

def predict(file):

    # Read DICOM
    ds = pydicom.dcmread(file.name)
    img = ds.pixel_array

    # Resize
    img = cv2.resize(img, (224,224))

    # Normalize
    img = img / 255.0

    # Convert grayscale → RGB
    img = cv2.cvtColor(
        img.astype(np.float32),
        cv2.COLOR_GRAY2RGB
    )

    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)[0][0]

    pneumonia_prob = float(pred)
    normal_prob = 1 - pneumonia_prob

    return {
        "Pneumonia": pneumonia_prob,
        "Normal": normal_prob
    }

demo = gr.Interface(
    fn=predict,
    inputs=gr.File(
        file_types=[".dcm"],
        label="Upload DICOM Chest X-ray (.dcm)"
    ),
    outputs=gr.Label(
        num_top_classes=2
    ),
    title="Pneumonia Detection from DICOM X-ray",
    description="Upload a DICOM (.dcm) chest X-ray image."
)

demo.launch()
