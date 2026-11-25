# ============================================================
# GENERATE DASHBOARD DATA
#
# Run this AFTER train_CNN.py, CNN_Attention_improved.py, and
# comparison.py have all been run at least once.
#
# It computes baseline vs attention accuracy (overall and
# per-class), the confusion matrices, and the train/val/test
# class distribution, and writes it all to:
#
#     app/static/data/results.json
#
# The website's "Results" and "Dataset" sections fetch this
# file (via GET /api/results) and render real numbers from
# your actual training run instead of placeholders.
# ============================================================

import os
import json
import datetime

import numpy as np

from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, accuracy_score


# ============================================================
# 1. PATHS
# ============================================================

BASE_PATH = r"C:\Users\91994\PycharmProjects\PythonProject7\CK+"

DATA_PATH = os.path.join(BASE_PATH, "preprocessed_data.npz")
BASELINE_MODEL_PATH = os.path.join(BASE_PATH, "baseline_cnn.h5")

ATTENTION_MODEL_CANDIDATES = [
    os.path.join(BASE_PATH, "cnn_attention_best.h5"),
    os.path.join(BASE_PATH, "cnn_attention.h5"),
]
ATTENTION_MODEL_PATH = next(
    (p for p in ATTENTION_MODEL_CANDIDATES if os.path.exists(p)), None
)

# Written relative to THIS script's location, inside the
# app/static/data folder of the website.
OUTPUT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "app",
    "static",
    "data",
    "results.json",
)
OUTPUT_PATH = os.path.normpath(OUTPUT_PATH)

EMOTIONS = [
    "Anger", "Contempt", "Disgust", "Fear",
    "Happy", "Sadness", "Surprise", "Neutral",
]


# ============================================================
# 2. VALIDATE FILES EXIST
# ============================================================

missing = [
    p for p in [DATA_PATH, BASELINE_MODEL_PATH, ATTENTION_MODEL_PATH]
    if p is None or not os.path.exists(p)
]

if missing:
    raise FileNotFoundError(
        "Missing required file(s), run the training scripts first:\n"
        + "\n".join(str(p) for p in missing)
    )


# ============================================================
# 3. LOAD DATA + MODELS
# ============================================================

print("Loading dataset and models...")

data = np.load(DATA_PATH)

X_train, y_train = data["X_train"], data["y_train"].astype(int)
X_val, y_val = data["X_val"], data["y_val"].astype(int)
X_test, y_test = data["X_test"], data["y_test"].astype(int)

baseline = load_model(BASELINE_MODEL_PATH)
attention = load_model(ATTENTION_MODEL_PATH)


# ============================================================
# 4. PREDICTIONS
# ============================================================

print("Running predictions on the test set...")

baseline_pred = np.argmax(baseline.predict(X_test, verbose=0), axis=1)
attention_pred = np.argmax(attention.predict(X_test, verbose=0), axis=1)

y_true = y_test


# ============================================================
# 5. OVERALL + PER-CLASS ACCURACY
# ============================================================

def per_class_accuracy(y_true, y_pred, num_classes):
    accuracies = []
    for class_id in range(num_classes):
        mask = y_true == class_id
        if mask.sum() == 0:
            accuracies.append(None)
        else:
            accuracies.append(float(accuracy_score(y_true[mask], y_pred[mask])))
    return accuracies


baseline_accuracy = float(accuracy_score(y_true, baseline_pred))
attention_accuracy = float(accuracy_score(y_true, attention_pred))

baseline_per_class = per_class_accuracy(y_true, baseline_pred, len(EMOTIONS))
attention_per_class = per_class_accuracy(y_true, attention_pred, len(EMOTIONS))


# ============================================================
# 6. CONFUSION MATRICES
# ============================================================

baseline_cm = confusion_matrix(
    y_true, baseline_pred, labels=list(range(len(EMOTIONS)))
).tolist()

attention_cm = confusion_matrix(
    y_true, attention_pred, labels=list(range(len(EMOTIONS)))
).tolist()


# ============================================================
# 7. CLASS DISTRIBUTION (train / val / test)
# ============================================================

def class_counts(labels, num_classes):
    counts = np.bincount(labels, minlength=num_classes)
    return counts.tolist()


class_distribution = {
    "train": class_counts(y_train, len(EMOTIONS)),
    "val": class_counts(y_val, len(EMOTIONS)),
    "test": class_counts(y_test, len(EMOTIONS)),
}


# ============================================================
# 8. AGREEMENT BETWEEN MODELS
# ============================================================

agreement = float(np.mean(baseline_pred == attention_pred))


# ============================================================
# 9. ASSEMBLE + SAVE
# ============================================================

payload = {
    "available": True,
    "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
    "classes": EMOTIONS,
    "baseline": {
        "accuracy": baseline_accuracy,
        "per_class_accuracy": baseline_per_class,
        "confusion_matrix": baseline_cm,
    },
    "attention": {
        "accuracy": attention_accuracy,
        "per_class_accuracy": attention_per_class,
        "confusion_matrix": attention_cm,
    },
    "agreement_rate": agreement,
    "class_distribution": class_distribution,
    "test_set_size": int(len(y_true)),
}

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

with open(OUTPUT_PATH, "w") as f:
    json.dump(payload, f, indent=2)

print("\nSaved dashboard data:")
print(OUTPUT_PATH)
print("\nAttention accuracy: ", round(baseline_accuracy * 100, 2), "%")
print(" Baseline accuracy:", round(attention_accuracy * 100, 2), "%")
print("\nRestart app.py (or refresh the page) to see it on the website.")
