# ============================================================
# FLASK APP — FACIAL EXPRESSION LAB
#
# Serves the website (templates/index.html) and two JSON
# endpoints:
#
#   POST /api/predict   - accepts an uploaded photo, runs it
#                          through the trained CNN + Attention
#                          model, returns the predicted emotion
#                          and the full probability distribution.
#
#   GET  /api/results    - returns the baseline-vs-attention
#                          comparison + dataset stats produced
#                          by generate_dashboard_data.py, so the
#                          "Results" section on the site reflects
#                          your actual training run rather than
#                          hard-coded numbers.
# ============================================================

import os
import json

import numpy as np
import cv2

from flask import Flask, request, jsonify, render_template

from tensorflow.keras.models import load_model


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_PATH = r"C:\Users\91994\PycharmProjects\PythonProject7\CK+"

IMG_SIZE = 48

# Order must match the label integers used throughout the
# project (0 = Anger ... 7 = Neutral), same as predict_emotion.py
EMOTIONS = [
    "Anger",
    "Contempt",
    "Disgust",
    "Fear",
    "Happy",
    "Sadness",
    "Surprise",
    "Neutral",
]

APP_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(APP_DIR, "static", "data", "results.json")

# Shown in the Results/Dataset sections until you've run
# generate_dashboard_data.py on your own trained models, so the
# site isn't empty on a fresh checkout. Clearly marked as example
# data via "is_demo" — the front end labels it accordingly rather
# than presenting it as your real results.
DEFAULT_RESULTS = {
    "available": True,
    "is_demo": True,
    "generated_at": None,
    "classes": EMOTIONS,
    "attention": {
        "accuracy": 0.84,
        "per_class_accuracy": [0.64, 0.28, 0.71, 0.58, 0.89, 0.66, 0.84, 0.73],
    },
    "baseline": {
        "accuracy": 0.52,
        "per_class_accuracy": [0.76, 0.52, 0.80, 0.71, 0.95, 0.78, 0.90, 0.83],
    },
    "agreement_rate": 0.83,
    "class_distribution": {
        "train": [85, 25, 95, 60, 180, 90, 140, 110],
        "val": [10, 3, 12, 7, 22, 11, 17, 14],
        "test": [12, 4, 13, 8, 24, 12, 19, 15],
    },
    "test_set_size": 107,
}


# ============================================================
# 2. LOAD MODEL (best checkpoint if available, else final save)
# ============================================================

MODEL_CANDIDATES = [
    os.path.join(BASE_PATH, "cnn_attention_best.h5"),
    os.path.join(BASE_PATH, "cnn_attention.h5"),
]

MODEL_PATH = next((p for p in MODEL_CANDIDATES if os.path.exists(p)), None)

model = None
if MODEL_PATH:
    print("Loading model from:", MODEL_PATH)
    model = load_model(MODEL_PATH)
else:
    print(
        "WARNING: no trained model found. Train CNN_Attention_improved.py "
        "first — /api/predict will return an error until then."
    )


# ============================================================
# 3. FLASK APP
# ============================================================

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB upload cap


# ============================================================
# 4. IMAGE PREPROCESSING (mirrors Preprocessing.py exactly:
#    grayscale -> CLAHE -> resize -> normalize)
# ============================================================

def preprocess_uploaded_image(file_bytes):
    array = np.frombuffer(file_bytes, dtype=np.uint8)
    img = cv2.imdecode(array, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError("Could not read that file as an image.")

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)   # batch
    img = np.expand_dims(img, axis=-1)  # channel

    return img


# ============================================================
# 5. ROUTES
# ============================================================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({
            "error": "No trained model found on the server. Train the "
                     "CNN + Attention model first, then restart the app."
        }), 503

    if "image" not in request.files or request.files["image"].filename == "":
        return jsonify({"error": "No image was uploaded."}), 400

    try:
        img = preprocess_uploaded_image(request.files["image"].read())
    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    probabilities = model.predict(img, verbose=0)[0]

    ranked = sorted(
        zip(EMOTIONS, probabilities.tolist()),
        key=lambda pair: pair[1],
        reverse=True,
    )

    return jsonify({
        "top_emotion": ranked[0][0],
        "confidence": ranked[0][1],
        "probabilities": [
            {"label": label, "value": value} for label, value in ranked
        ],
    })


@app.route("/api/results")
def results():
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, "r") as f:
            return jsonify(json.load(f))

    return jsonify(DEFAULT_RESULTS)


# ============================================================
# 6. RUN
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("FACIAL EXPRESSION LAB — starting server")
    print("Open: http://127.0.0.1:5000")
    print("=" * 60 + "\n")
    app.run(debug=True, port=5000)