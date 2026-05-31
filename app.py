
import gradio as gr
import tensorflow as tf
import numpy as np
import cv2

# Load model
model = tf.keras.models.load_model(
    'best_pneumonia_model.h5'
)

def predict(img):

    # Resize
    img = cv2.resize(img, (224,224))

    # Normalize
    img = img / 255.0

    # Convert grayscale → RGB
    if len(img.shape)==2:
        img = cv2.cvtColor(
            img,
            cv2.COLOR_GRAY2RGB
        )

    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)[0][0]

    label = (
        "Pneumonia"
        if pred > 0.5
        else "Normal"
    )

    return {
        label: float(pred)
    }

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(),
    outputs=gr.Label(),
    title="Pneumonia Detection App"
)

demo.launch()
