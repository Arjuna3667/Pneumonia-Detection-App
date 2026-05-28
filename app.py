
import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model("best_pneumonia_model.h5")
IMG_SIZE=(224,224)
classes=["Normal","Pneumonia"]

def predict(img):
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img)/255.0
    arr = np.expand_dims(arr,0)
    pred = model.predict(arr)[0][0]
    label = classes[int(pred>0.5)]
    prob = float(pred if pred>0.5 else 1-pred)
    return {"Predicted Class": label, "Probability": prob}

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="label",
    title="Pneumonia Detection"
)
demo.launch()
