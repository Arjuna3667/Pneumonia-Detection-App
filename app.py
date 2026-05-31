import gradio as gr
import tensorflow as tf
import numpy as np
import cv2
import pydicom
from pydicom.uid import ImplicitVRLittleEndian

# Load model
model = tf.keras.models.load_model(
    "best_pneumonia_model.h5"
)

def predict(file):

    try:

        # Force-read DICOM
        ds = pydicom.dcmread(
            file.name,
            force=True
        )

        # Fix missing Transfer Syntax UID
        if not hasattr(ds.file_meta, "TransferSyntaxUID"):
            ds.file_meta.TransferSyntaxUID = (
                ImplicitVRLittleEndian
            )

        # Read pixel array
        img = ds.pixel_array.astype(np.float32)

        # Resize to model input size
        img = cv2.resize(
            img,
            (224,224)
        )

        # Normalize
        img = img / np.max(img)

        # Convert grayscale → RGB
        img = cv2.cvtColor(
            img,
            cv2.COLOR_GRAY2RGB
        )

        # Add batch dimension
        img = np.expand_dims(
            img,
            axis=0
        )

        # Prediction
        pred = model.predict(img)[0][0]

        return {
            "Pneumonia": float(pred),
            "Normal": float(1-pred)
        }

    except Exception as e:
        return str(e)

# Gradio UI
demo = gr.Interface(
    fn=predict,
    inputs=gr.File(
        file_types=[".dcm"],
        label="Upload DICOM Chest X-ray (.dcm)"
    ),
    outputs=gr.Label(num_top_classes=2),
    title="Pneumonia Detection from DICOM X-ray",
    description="Upload a DICOM (.dcm) chest X-ray image."
)

demo.launch()
