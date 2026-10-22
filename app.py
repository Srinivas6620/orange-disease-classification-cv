"""
app.py
======
Flask web application for Orange Disease Classification.
Serves a beautiful UI that classifies uploaded orange images
using BOTH trained models simultaneously:
  - Approach 1: Classical CV (SVM + handcrafted features)
  - Approach 2: Deep Learning (MobileNetV2 / TensorFlow-Keras)

Usage:
    python app.py
Then open: http://localhost:5000
"""

import os
import sys
import time
import io
import base64
import numpy as np
import cv2
import joblib
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

# ─────────────────────────────────────────────────────────────────────────────
# App Configuration
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent
MODELS_DIR   = BASE_DIR / "models"
UPLOAD_DIR   = BASE_DIR / "static" / "uploads"
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR   = BASE_DIR / "static"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB

CLASS_NAMES  = ["blackspot", "canker", "fresh", "grenning"]
CLASS_LABELS = {
    "blackspot": "Black Spot Disease",
    "canker":    "Canker Disease",
    "fresh":     "Fresh / Healthy",
    "grenning":  "Greening (HLB)",
}
CLASS_COLORS = {
    "blackspot": "#1565C0",
    "canker":    "#E65100",
    "fresh":     "#2E7D32",
    "grenning":  "#F9A825",
}
CLASS_ICONS = {
    "blackspot": "🔵",
    "canker":    "🟠",
    "fresh":     "🟢",
    "grenning":  "🟡",
}
IMG_SIZE = 224

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
app.secret_key = "greenpods-orange-cv-2024"

# ─────────────────────────────────────────────────────────────────────────────
# Load Models at Startup
# ─────────────────────────────────────────────────────────────────────────────

scaler_model = None
svm_model    = None
dl_model     = None


def load_models():
    global scaler_model, svm_model, dl_model

    # Classical models
    scaler_path = MODELS_DIR / "scaler.pkl"
    svm_path    = MODELS_DIR / "svm_model.pkl"
    if scaler_path.exists() and svm_path.exists():
        try:
            scaler_model = joblib.load(scaler_path)
            svm_model    = joblib.load(svm_path)
            print(f"[OK] Classical SVM model loaded from {svm_path.name}")
        except Exception as e:
            print(f"[WARNING] Could not load SVM model: {e}")
    else:
        print(f"[WARNING] Classical models not found in {MODELS_DIR}. "
              "Run Notebook 02 first to train and save them.")

    # Deep Learning model
    dl_path = MODELS_DIR / "mobilenetv2_best.h5"
    if dl_path.exists():
        try:
            import tensorflow as tf
            dl_model = tf.keras.models.load_model(str(dl_path))
            print(f"[OK] MobileNetV2 model loaded from {dl_path.name}")
        except ImportError:
            print("[WARNING] TensorFlow not installed. DL model skipped.")
        except Exception as e:
            print(f"[WARNING] Could not load DL model: {e}")
    else:
        print(f"[WARNING] DL model not found at {dl_path}. "
              "Run Notebook 03 first to train and save it.")


# ─────────────────────────────────────────────────────────────────────────────
# Feature Extraction (mirrors src/features.py)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_features(bgr_img: np.ndarray) -> np.ndarray:
    from skimage.feature import local_binary_pattern, graycomatrix, graycoprops

    lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_feats = []
    for ch in range(3):
        c = lab[:, :, ch]
        lab_feats.extend([c.mean(), c.std()])

    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv_feats = []
    for ch, (lo, hi) in enumerate([(0, 180), (0, 256), (0, 256)]):
        hist = cv2.calcHist([hsv], [ch], None, [16], [lo, hi])
        hist = hist.flatten() / (hist.sum() + 1e-7)
        hsv_feats.extend(hist.tolist())

    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    lbp_feats = []
    for P, R in [(8, 1), (16, 2)]:
        n_bins = P + 2
        lbp = local_binary_pattern(gray, P, R, method="uniform")
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins,
                               range=(0, n_bins), density=True)
        lbp_feats.extend(hist.tolist())

    gray_r = (gray // 4).astype(np.uint8)
    glcm = graycomatrix(gray_r, distances=[1], angles=[0, np.pi / 2],
                        levels=64, symmetric=True, normed=True)
    glcm_feats = []
    for prop in ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]:
        glcm_feats.extend(graycoprops(glcm, prop).flatten().tolist())

    return np.concatenate([lab_feats, hsv_feats, lbp_feats, glcm_feats])


# ─────────────────────────────────────────────────────────────────────────────
# Inference Functions
# ─────────────────────────────────────────────────────────────────────────────

def predict_classical(bgr_img: np.ndarray):
    if scaler_model is None or svm_model is None:
        return None
    bgr_resized = cv2.resize(bgr_img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    t0 = time.perf_counter()
    feat_vec    = _extract_features(bgr_resized).reshape(1, -1)
    feat_scaled = scaler_model.transform(feat_vec)
    proba       = svm_model.predict_proba(feat_scaled)[0]
    elapsed_ms  = (time.perf_counter() - t0) * 1000
    pred_idx    = int(np.argmax(proba))
    return {
        "predicted":    CLASS_NAMES[pred_idx],
        "confidence":   float(proba[pred_idx]),
        "probabilities": {CLASS_NAMES[i]: round(float(p), 4) for i, p in enumerate(proba)},
        "inference_ms": round(elapsed_ms, 1),
    }


def predict_dl(bgr_img: np.ndarray):
    if dl_model is None:
        return None
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mv2_preprocess
    rgb         = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    rgb_resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    x           = mv2_preprocess(rgb_resized.astype(np.float32))
    x           = np.expand_dims(x, axis=0)
    t0          = time.perf_counter()
    proba       = dl_model.predict(x, verbose=0)[0]
    elapsed_ms  = (time.perf_counter() - t0) * 1000
    pred_idx    = int(np.argmax(proba))
    return {
        "predicted":    CLASS_NAMES[pred_idx],
        "confidence":   float(proba[pred_idx]),
        "probabilities": {CLASS_NAMES[i]: round(float(p), 4) for i, p in enumerate(proba)},
        "inference_ms": round(elapsed_ms, 1),
    }


def image_to_base64(bgr_img: np.ndarray, max_size: int = 400) -> str:
    h, w = bgr_img.shape[:2]
    scale = min(max_size / h, max_size / w, 1.0)
    display = cv2.resize(bgr_img, (int(w * scale), int(h * scale)))
    display = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
    from PIL import Image
    pil_img = Image.fromarray(display)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.before_request
def startup():
    if svm_model is None or dl_model is None:
        load_models()


@app.route("/")
def index():
    classical_ready = svm_model is not None
    dl_ready        = dl_model is not None
    return render_template(
        "index.html",
        class_names=CLASS_NAMES,
        class_labels=CLASS_LABELS,
        class_colors=CLASS_COLORS,
        class_icons=CLASS_ICONS,
        classical_ready=classical_ready,
        dl_ready=dl_ready,
    )


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"error": f"File type '{ext}' not allowed. Use: jpg, png, bmp"}), 400

    # Read image
    file_bytes = np.frombuffer(file.read(), np.uint8)
    bgr_img    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if bgr_img is None:
        try:
            from PIL import Image
            pil_img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
            rgb_arr = np.array(pil_img)
            bgr_img = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
        except Exception as e:
            return jsonify({"error": f"Could not decode image: {e}"}), 400

    # Run inference
    classical_result = predict_classical(bgr_img)
    dl_result        = predict_dl(bgr_img)

    # Image thumbnail
    img_b64 = image_to_base64(bgr_img)

    # Agreement
    agreement = None
    if classical_result and dl_result:
        agreement = classical_result["predicted"] == dl_result["predicted"]

    return jsonify({
        "image_b64":   img_b64,
        "classical":   classical_result,
        "dl":          dl_result,
        "agreement":   agreement,
        "class_labels": CLASS_LABELS,
        "class_colors": CLASS_COLORS,
        "class_icons":  CLASS_ICONS,
    })


@app.route("/sample/<stage>", methods=["GET"])
def sample(stage):
    if stage not in CLASS_NAMES:
        return jsonify({"error": f"Unknown stage: {stage}"}), 400
    import glob
    candidates = glob.glob(str(BASE_DIR / "data" / "test" / stage / "*.*"))
    if not candidates:
        return jsonify({"error": f"No samples found for {stage}"}), 404
    sample_path = candidates[0]
    bgr_img = cv2.imread(sample_path)
    if bgr_img is None:
        return jsonify({"error": "Failed to read sample image"}), 500

    classical_result = predict_classical(bgr_img)
    dl_result        = predict_dl(bgr_img)
    img_b64          = image_to_base64(bgr_img)

    agreement = None
    if classical_result and dl_result:
        agreement = classical_result["predicted"] == dl_result["predicted"]

    return jsonify({
        "image_b64":    img_b64,
        "classical":    classical_result,
        "dl":           dl_result,
        "agreement":    agreement,
        "class_labels": CLASS_LABELS,
        "class_colors": CLASS_COLORS,
        "class_icons":  CLASS_ICONS,
        "sample_name":  Path(sample_path).name,
    })


@app.route("/health")
def health():
    return jsonify({
        "status":         "ok",
        "classical_model": svm_model is not None,
        "dl_model":        dl_model is not None,
    })


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  Green Pods Labs - Orange Disease Classifier")
    print("=" * 55)
    load_models()
    print("\nStarting Flask server at http://localhost:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=False)

