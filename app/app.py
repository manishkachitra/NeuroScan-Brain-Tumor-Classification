import os
import io
import pickle
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder='.')
CORS(app)

# -------------------------------------------------------------
# Configuration & Metadata
# -------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PKL_PATH = os.path.join(BASE_DIR, "neuroscan_model.pkl")
WEIGHTS_PTH_PATH = os.path.join(BASE_DIR, "neuroscan_best_mobilenetv2.pth")

CLASS_NAMES = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]
DISPLAY_NAMES = {
    "glioma_tumor": "Glioma Tumor",
    "meningioma_tumor": "Meningioma Tumor",
    "no_tumor": "Healthy / No Tumor",
    "pituitary_tumor": "Pituitary Tumor"
}

# -------------------------------------------------------------
# Load Model Metadata from .pkl
# -------------------------------------------------------------
model_metadata = {}
if os.path.exists(MODEL_PKL_PATH):
    try:
        with open(MODEL_PKL_PATH, "rb") as f:
            model_metadata = pickle.load(f)
        print("Loaded model metadata from neuroscan_model.pkl")
    except Exception as e:
        print(f"Warning: Could not read neuroscan_model.pkl: {e}")

# -------------------------------------------------------------
# Optional PyTorch Model Setup (if torch & weights are installed)
# -------------------------------------------------------------
torch_model = None
try:
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
    from PIL import Image

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def build_model(num_classes=4):
        model = models.mobilenet_v2(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes)
        )
        return model

    if os.path.exists(WEIGHTS_PTH_PATH):
        torch_model = build_model(len(CLASS_NAMES))
        torch_model.load_state_dict(torch.load(WEIGHTS_PTH_PATH, map_location=device))
        torch_model.to(device)
        torch_model.eval()
        print("PyTorch model loaded with weights from neuroscan_best_mobilenetv2.pth")
    else:
        print("Note: neuroscan_best_mobilenetv2.pth not found locally.")

    eval_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
except Exception as e:
    print(f"PyTorch runtime not fully initialized: {e}")
    eval_transforms = None

# -------------------------------------------------------------
# Routes
# -------------------------------------------------------------
@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/<path:path>")
def static_proxy(path):
    return send_from_directory(".", path)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "service": "NeuroScan Brain Tumor Classifier",
        "model_loaded": torch_model is not None,
        "classes": CLASS_NAMES
    })

@app.route("/model-info", methods=["GET"])
def model_info():
    if model_metadata:
        # Return a sanitized copy suitable for JSON response
        info = {k: v for k, v in model_metadata.items() if isinstance(v, (str, int, float, list, dict))}
        return jsonify(info)
    return jsonify({
        "model_name": "NeuroScan MobileNetV2",
        "classes": CLASS_NAMES,
        "display_names": DISPLAY_NAMES
    })

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files and "image" not in request.files:
        return jsonify({"error": "No image file provided. Send file with key 'file' or 'image'."}), 400

    file = request.files.get("file") or request.files.get("image")
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    try:
        from PIL import Image
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        if torch_model is not None and eval_transforms is not None:
            tensor = eval_transforms(img).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = torch_model(tensor)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

            probabilities = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
            top_idx = int(probs.argmax())
            top_class = CLASS_NAMES[top_idx]
            confidence = float(probs[top_idx])

            return jsonify({
                "top_class": top_class,
                "display_name": DISPLAY_NAMES.get(top_class, top_class),
                "confidence": confidence,
                "probabilities": probabilities,
                "engine": "PyTorch MobileNetV2"
            })
        else:
            return jsonify({
                "message": "Model weights (.pth) not currently loaded on host. Place neuroscan_best_mobilenetv2.pth in root or run with PyTorch environment.",
                "classes": CLASS_NAMES,
                "display_names": DISPLAY_NAMES
            }), 200

    except Exception as e:
        return jsonify({"error": f"Failed to process image: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

