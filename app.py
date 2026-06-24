import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import ImageOps

model = tf.keras.models.load_model("mnist_model.keras")


def predict_digit(sketch):
    if sketch is None:
        return None
    if isinstance(sketch, dict):
        img = sketch["composite"]
    else:
        img = sketch
    img = img.convert("L")
    img = ImageOps.invert(img)
    img = img.resize((28, 28))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    prediction = model.predict(img_array)[0]
    confidences = {str(i): float(prediction[i]) for i in range(10)}
    return confidences


demo = gr.Interface(
    fn=predict_digit,
    inputs=gr.Sketchpad(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="Live Handwritten Digit Recognizer",
    description="Draw a single digit (0-9) on the canvas below and watch the CNN predict it in real-time.",
)

if __name__ == "__main__":
    demo.launch(inbrowser=True)
