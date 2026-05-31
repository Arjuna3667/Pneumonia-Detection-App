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

        # Read DICOM
        ds = pydicom.dcmread(
            file.name,
            force=True
        )

        # Fix missing transfer syntax
        if not hasattr(ds.file_meta, "TransferSyntaxUID"):
            ds.file_meta.TransferSyntaxUID = (
                ImplicitVRLittleEndian
            )

        # Check pixel data exists
        if 'PixelData' not in ds:
            return {
                "Error": 1.0
            }

        # Read image
        img = ds.pixel_array.astype(np.float32)

        # Resize
        img = cv2.resize(
            img,
            (224,224)
        )

        # Safe normalization
        if np.max(img) > 0:
            img = img / np.max(img)

        # Gray → RGB
        if len(img.shape) == 2:
            img = cv2.cvtColor(
                img,
                cv2.COLOR_GRAY2RGB
            )

        # Batch dimension
        img = np.expand_dims(
            img,
            axis=0
        )

        # Prediction
        pred = model.predict(
            img,
            verbose=0
        )[0][0]

        return {
            "Pneumonia": float(pred),
            "Normal": float(1-pred)
        }

    except Exception as e:
        return {
            "Error": str(e)
        }

demo = gr.Interface(
    fn=predict,
    inputs=gr.File(
        file_types=[".dcm"],
        label="Upload Chest X-ray DICOM (.dcm)"
    ),
    outputs=gr.Label(),
    title="Pneumonia Detection from DICOM X-ray",
    description="Upload a chest X-ray DICOM image (.dcm)."
)

demo.launch()
