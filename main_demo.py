#!/usr/bin/env python3
"""
main_demo.py
============
Single-image inference demo for BOTH trained models:
  - Approach 1: Classical CV (SVM + handcrafted features)
  - Approach 2: Deep Learning (MobileNetV2 / TensorFlow-Keras)

Usage:
    python main_demo.py --image path/to/avocado.jpg

Requirements:
    - models/scaler.pkl     (saved by Notebook 2)
    - models/svm_model.pkl  (saved by Notebook 2)
    - models/mobilenetv2_best.h5 (saved by Notebook 3)
"""

import argparse
import os
import sys
import time
import numpy as np
import cv2
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).parent
MODELS_DIR   = SCRIPT_DIR / "models"
SRC_DIR      = SCRIPT_DIR / "src"

sys.path.insert(0, str(SCRIPT_DIR))

CLASS_NAMES  = ["blackspot", "canker", "fresh", "grenning"]
CLASS_COLORS = ["#1565C0", "#E65100", "#2E7D32", "#F9A825"]
IMG_SIZE     = 224


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction (mirrors src/features.py — standalone for demo)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_features(bgr_img: np.ndarray) -> np.ndarray:
    """Extract the same feature vector used in training (Approach 1)."""
    from skimage.feature import local_binary_pattern, graycomatrix, graycoprops

    # CIELAB colour statistics
    lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_feats = []
    for ch in range(3):
        c = lab[:, :, ch]
        lab_feats.extend([c.mean(), c.std()])

    # HSV colour histograms (16 bins each channel)
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv_feats = []
    for ch, (lo, hi) in enumerate([(0, 180), (0, 256), (0, 256)]):
        hist = cv2.calcHist([hsv], [ch], None, [16], [lo, hi])
        hist = hist.flatten() / (hist.sum() + 1e-7)
        hsv_feats.extend(hist.tolist())

    # LBP texture
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    lbp_feats = []
    for P, R in [(8, 1), (16, 2)]:
        n_bins = P + 2
        lbp = local_binary_pattern(gray, P, R, method="uniform")
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins,
                               range=(0, n_bins), density=True)
        lbp_feats.extend(hist.tolist())

    # GLCM texture
    gray_r = (gray // 4).astype(np.uint8)
    glcm = graycomatrix(gray_r, distances=[1], angles=[0, np.pi / 2],
                        levels=64, symmetric=True, normed=True)
    glcm_feats = []
    for prop in ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]:
        glcm_feats.extend(graycoprops(glcm, prop).flatten().tolist())

    return np.concatenate([lab_feats, hsv_feats, lbp_feats, glcm_feats])


# ─────────────────────────────────────────────────────────────────────────────
# Load models
# ─────────────────────────────────────────────────────────────────────────────

def load_classical_models():
    """Load saved scaler and SVM model. Returns (scaler, svm) or None if not found."""
    scaler_path = MODELS_DIR / "scaler.pkl"
    svm_path    = MODELS_DIR / "svm_model.pkl"
    if not scaler_path.exists() or not svm_path.exists():
        print(f"[WARNING] Classical models not found in {MODELS_DIR}. "
              "Run Notebook 2 first.")
        return None, None
    scaler = joblib.load(scaler_path)
    svm    = joblib.load(svm_path)
    print(f"[OK] Loaded SVM model from {svm_path.name}")
    return scaler, svm


def load_dl_model():
    """Load saved Keras MobileNetV2 model. Returns model or None if not found."""
    model_path = MODELS_DIR / "mobilenetv2_best.h5"
    if not model_path.exists():
        print(f"[WARNING] DL model not found at {model_path}. "
              "Run Notebook 3 first.")
        return None
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(str(model_path))
        print(f"[OK] Loaded MobileNetV2 model from {model_path.name}")
        return model
    except ImportError:
        print("[WARNING] TensorFlow not installed. DL inference skipped.")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

def predict_classical(image_path: str, scaler, svm):
    """
    Run Approach 1 inference.
    Returns (predicted_class_name, confidence_dict, inference_ms).
    """
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    bgr_resized = cv2.resize(bgr, (IMG_SIZE, IMG_SIZE),
                             interpolation=cv2.INTER_AREA)

    t0 = time.perf_counter()
    feat_vec = _extract_features(bgr_resized).reshape(1, -1)
    feat_scaled = scaler.transform(feat_vec)
    proba = svm.predict_proba(feat_scaled)[0]
    elapsed_ms = (time.perf_counter() - t0) * 1000

    pred_idx   = int(np.argmax(proba))
    pred_class = CLASS_NAMES[pred_idx]
    conf_dict  = {CLASS_NAMES[i]: float(p) for i, p in enumerate(proba)}

    return pred_class, conf_dict, elapsed_ms


def predict_dl(image_path: str, model):
    """
    Run Approach 2 inference.
    Returns (predicted_class_name, confidence_dict, inference_ms).
    """
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mv2_preprocess

    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb_resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE),
                             interpolation=cv2.INTER_AREA)
    x = mv2_preprocess(rgb_resized.astype(np.float32))
    x = np.expand_dims(x, axis=0)   # (1, 224, 224, 3)

    t0 = time.perf_counter()
    proba = model.predict(x, verbose=0)[0]
    elapsed_ms = (time.perf_counter() - t0) * 1000

    pred_idx   = int(np.argmax(proba))
    pred_class = CLASS_NAMES[pred_idx]
    conf_dict  = {CLASS_NAMES[i]: float(p) for i, p in enumerate(proba)}

    return pred_class, conf_dict, elapsed_ms


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def visualise_results(image_path: str,
                      classical_result=None,
                      dl_result=None):
    """
    Display the input image alongside prediction bars for both models.
    """
    bgr = cv2.imread(image_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb_display = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))

    n_panels = 1 + (classical_result is not None) + (dl_result is not None)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 6))
    if n_panels == 1:
        axes = [axes]

    # ── Image panel ──────────────────────────────────────────────────
    axes[0].imshow(rgb_display)
    axes[0].set_title("Input Image", fontweight="bold", fontsize=12)
    axes[0].axis("off")

    ax_idx = 1

    # ── Classical result panel ────────────────────────────────────────
    if classical_result is not None:
        pred_cls, conf, ms = classical_result
        _bar_chart(axes[ax_idx], conf, pred_cls,
                   f"Approach 1: SVM\n({ms:.1f} ms)")
        ax_idx += 1

    # ── DL result panel ───────────────────────────────────────────────
    if dl_result is not None:
        pred_cls, conf, ms = dl_result
        _bar_chart(axes[ax_idx], conf, pred_cls,
                   f"Approach 2: MobileNetV2\n({ms:.1f} ms)")

    plt.suptitle("Orange Quality & Disease Classification — Inference Demo",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.show()


def _bar_chart(ax, conf_dict: dict, predicted: str, title: str):
    """Draw horizontal confidence bars for each class."""
    classes = list(conf_dict.keys())
    values  = [conf_dict[c] for c in classes]
    colors  = [CLASS_COLORS[CLASS_NAMES.index(c)] for c in classes]
    edge    = ["black" if c == predicted else "none" for c in classes]

    bars = ax.barh(classes, values, color=colors, edgecolor=edge,
                   linewidth=2, height=0.55)
    for bar, val in zip(bars, values):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.1%}", va="center", fontsize=10)

    ax.set_xlim(0, 1.15)
    ax.set_xlabel("Confidence")
    ax.set_title(title, fontweight="bold", fontsize=11)
    ax.axvline(0.5, color="gray", linestyle="--", alpha=0.4)

    # Star marker on predicted class
    pred_idx = classes.index(predicted)
    ax.text(1.05, pred_idx, "★", va="center", ha="center",
            fontsize=14, color="gold")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Orange Disease & Quality Classification — Dual-Model Inference Demo"
    )
    parser.add_argument(
        "--image", "-i",
        required=True,
        help="Path to the orange image to classify.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Suppress the matplotlib visualisation window.",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        print(f"[ERROR] Image not found: {args.image}")
        sys.exit(1)

    # Fix Windows console encoding for UTF-8 output if possible
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    print(f"  Image: {args.image}\n")

    classical_result = None
    dl_result        = None

    # ── Approach 1: Classical ─────────────────────────────────────────
    scaler, svm = load_classical_models()
    if scaler is not None and svm is not None:
        pred_cls, conf, ms = predict_classical(args.image, scaler, svm)
        classical_result = (pred_cls, conf, ms)
        print(f"\n[Approach 1 — SVM]")
        print(f"  Prediction : {pred_cls.upper()}")
        print(f"  Confidence : {conf[pred_cls]:.1%}")
        print(f"  Time       : {ms:.1f} ms")
        print("  Per-class probabilities:")
        for cls, p in conf.items():
            bar = "#" * int(p * 30)
            print(f"    {cls:12s}: {p:6.1%}  {bar}")

    # ── Approach 2: MobileNetV2 ───────────────────────────────────────
    dl_model = load_dl_model()
    if dl_model is not None:
        pred_cls, conf, ms = predict_dl(args.image, dl_model)
        dl_result = (pred_cls, conf, ms)
        print(f"\n[Approach 2 - MobileNetV2]")
        print(f"  Prediction : {pred_cls.upper()}")
        print(f"  Confidence : {conf[pred_cls]:.1%}")
        print(f"  Time       : {ms:.1f} ms")
        print("  Per-class probabilities:")
        for cls, p in conf.items():
            bar = "#" * int(p * 30)
            print(f"    {cls:12s}: {p:6.1%}  {bar}")

    print(f"\n{'='*55}")
    if classical_result and dl_result:
        c_pred = classical_result[0]
        d_pred = dl_result[0]
        agree  = "[AGREE]" if c_pred == d_pred else "[DISAGREE]"
        print(f"  Model Agreement: {agree}")
        print(f"    SVM         -> {c_pred}")
        print(f"    MobileNetV2 -> {d_pred}")
    print(f"{'='*55}\n")

    if not args.no_plot:
        visualise_results(args.image, classical_result, dl_result)


if __name__ == "__main__":
    main()
