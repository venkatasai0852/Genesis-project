'''
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from flask import Flask, request, render_template
from inference.predict import predict

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None

    if request.method == "POST":
        image = request.files["image"]
        image_path = "static/upload.jpg"
        image.save(image_path)

        result, confidence = predict(image_path)

    return render_template("index.html",
                           result=result,
                           confidence=confidence)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
'''

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import gradio as gr

from model_def import RGBFFT_Attention_EfficientNetB3

# Load model
model = RGBFFT_Attention_EfficientNetB3()
model.load_state_dict(torch.load("best_model.pth", map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

LABELS = {0: "Real", 1: "AI Generated"}

def predict(image):
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        out = model(image)
        probs = F.softmax(out, dim=1)

    conf, pred = torch.max(probs, dim=1)
    confidence = conf.item()

    if confidence < 0.6:
        return "Uncertain (low confidence)"

    return f"{LABELS[pred.item()]} (Confidence: {confidence:.2f})"

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Venkata Sai – Genesis Project",
    description="AI Generated Image Detector using Deep Learning"
)

demo.launch()
