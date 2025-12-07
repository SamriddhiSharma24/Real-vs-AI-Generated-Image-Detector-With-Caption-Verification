from flask import Flask, render_template, request
from PIL import Image, ImageOps
import torch
import torch.nn as nn
from torchvision import models, transforms
import clip
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch.nn.functional as F


# Flask Setup

app = Flask(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"


# Load Models

print("Loading BLIP (captioning)...")
caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
caption_model.eval()

print("Loading CLIP...")
clip_model, preprocess = clip.load("ViT-B/32", device=device)
clip_model.eval()

print("Loading Fake Detector (ResNet18)...")
fake_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
fake_model.fc = nn.Linear(fake_model.fc.in_features, 2)

try:
    fake_model.load_state_dict(torch.load("fake_detector.pth", map_location=device))
    print("Fake detector loaded.")
except Exception as e:
    print("Could not load fake_detector.pth:", e)

fake_model = fake_model.to(device)
fake_model.eval()


# Preprocessing

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def normalize_image(image):
    image = image.convert("RGB")
    image = ImageOps.exif_transpose(image)
    return image

# Caption Generator
def generate_caption(image):
    inputs = caption_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        out = caption_model.generate(**inputs, max_length=40)
    return caption_processor.decode(out[0], skip_special_tokens=True)


# Fake Detection + Accuracy Matrix
def detect_fake(image):
    image = normalize_image(image)
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = fake_model(img_tensor)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()

    ai_conf = float(probs[0] * 100)    # Class 0 = AI-generated
    real_conf = float(probs[1] * 100)  # Class 1 = Real

    predicted = torch.argmax(logits, dim=1).item()
    label = "Real" if predicted == 1 else "AI-generated"

    return label, real_conf, ai_conf


# CLIP Caption Match

def check_caption_match(image, caption):
    image_input = preprocess(image).unsqueeze(0).to(device)
    text_input = clip.tokenize([caption]).to(device)
    with torch.no_grad():
        image_features = clip_model.encode_image(image_input)
        text_features = clip_model.encode_text(text_input)
        similarity = torch.cosine_similarity(image_features, text_features)
    return similarity.item() > 0.25


# Routes

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files or request.files["image"].filename == "":
        return render_template("index.html", result="⚠️ Please upload an image.")

    image_file = request.files["image"]
    user_caption = request.form.get("caption", "")
    image = Image.open(image_file).convert("RGB")

    generated_caption = generate_caption(image)

    authenticity, real_conf, ai_conf = detect_fake(image)

    caption_match = "Match" if check_caption_match(image, user_caption) else "Mismatch"

    # Build Accuracy Matrix Table
    accuracy_table = f"""
    <table class="table table-bordered text-white">
        <tr>
            <th>Class</th>
            <th>Confidence</th>
        </tr>
        <tr>
            <td><b>Real</b></td>
            <td>{real_conf:.2f}%</td>
        </tr>
        <tr>
            <td><b>AI-generated</b></td>
            <td>{ai_conf:.2f}%</td>
        </tr>
    </table>
    """

    result_html = f"""
    <h3><b>Generated Caption:</b> {generated_caption}</h3>
    <h3><b>Your Caption:</b> {user_caption}</h3>
    <h4><b>Caption Match:</b> {caption_match}</h4>
    <h3><b>Image Authenticity:</b> {authenticity}</h3>
    <h4><b>Accuracy Matrix</b></h4>
    {accuracy_table}
    """

    return render_template("index.html", result=result_html)

# Run Flask
if __name__ == "__main__":
    app.run(debug=True)
