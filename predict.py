import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model.model_def import RGBFFT_Attention_EfficientNetB3
    

# Load model
model = RGBFFT_Attention_EfficientNetB3()
model.load_state_dict(torch.load("model/best_model.pth", map_location="cpu"))
model.eval()

# Image transform (same size as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

LABELS = {0: "Real", 1: "AI Generated"}

def predict(image_path):
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img)
        probs = F.softmax(output, dim=1)

    confidence, pred = torch.max(probs, dim=1)
    confidence = confidence.item()
    label = LABELS[pred.item()]
    if confidence < 0.6:
        return "Uncertain", confidence
    return label, confidence



# Test directly
if __name__ == "__main__":
    label, conf = predict("test.jpg")
    print("Prediction:", label)
    print("Confidence:", conf)

