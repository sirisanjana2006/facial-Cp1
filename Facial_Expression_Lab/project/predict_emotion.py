# ============================================================
# STEP 8 - PREDICT EMOTION USING CNN + ATTENTION
# ============================================================

import os
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


# ============================================================
# 1. BASE FOLDER
# ============================================================

BASE_PATH = r"C:\Users\91994\PycharmProjects\PythonProject7\CK+"


# ============================================================
# 2. MODEL PATH
# ============================================================

MODEL_PATH = os.path.join(
    BASE_PATH,
    "cnn_attention.h5"
)


# ============================================================
# 3. IMAGE PATH
# ============================================================

IMAGE_PATH = os.path.join(
    BASE_PATH,
    "processed_images",
    "all_images",
    "image_0000.png"
)


# ============================================================
# 4. EMOTION LABELS
# ============================================================

EMOTIONS = [
    "Anger",
    "Contempt",
    "Disgust",
    "Fear",
    "Happy",
    "Sadness",
    "Surprise",
    "Neutral"
]


# ============================================================
# 5. START
# ============================================================

print("=" * 60)
print("CNN + ATTENTION EMOTION PREDICTION")
print("=" * 60)


# ============================================================
# 6. CHECK MODEL
# ============================================================

print("\nChecking model...")

if not os.path.exists(MODEL_PATH):

    print("\nERROR: Model not found!")
    print("Expected location:")
    print(MODEL_PATH)
    exit()

print("Model found:")
print(MODEL_PATH)


# ============================================================
# 7. LOAD MODEL
# ============================================================

print("\nLoading CNN + Attention model...")

model = load_model(
    MODEL_PATH
)

print("Model loaded successfully!")


# ============================================================
# 8. CHECK IMAGE
# ============================================================

print("\nChecking image...")

if not os.path.exists(IMAGE_PATH):

    print("\nERROR: Image not found!")
    print("Expected location:")
    print(IMAGE_PATH)
    exit()

print("Image found:")
print(IMAGE_PATH)


# ============================================================
# 9. LOAD IMAGE
# ============================================================

print("\nLoading image...")

img = image.load_img(
    IMAGE_PATH,
    target_size=(48, 48),
    color_mode="grayscale"
)


# ============================================================
# 10. CONVERT IMAGE TO ARRAY
# ============================================================

img_array = image.img_to_array(
    img
)


# ============================================================
# 11. NORMALIZE
# ============================================================

img_array = img_array / 255.0


# ============================================================
# 12. ADD BATCH DIMENSION
# ============================================================

img_array = np.expand_dims(
    img_array,
    axis=0
)


print("Image shape:")
print(img_array.shape)


# ============================================================
# 13. PREDICT
# ============================================================

print("\nPredicting emotion...")

prediction = model.predict(
    img_array,
    verbose=0
)


# ============================================================
# 14. GET PREDICTED CLASS
# ============================================================

predicted_index = np.argmax(
    prediction[0]
)


predicted_emotion = EMOTIONS[
    predicted_index
]


# ============================================================
# 15. GET CONFIDENCE
# ============================================================

confidence = (
    prediction[0][predicted_index]
    * 100
)


# ============================================================
# 16. DISPLAY RESULT
# ============================================================

print("\n" + "=" * 60)
print("PREDICTION RESULT")
print("=" * 60)

print(
    "\nPredicted Emotion:",
    predicted_emotion
)

print(
    "Confidence:",
    round(confidence, 2),
    "%"
)

print("=" * 60)


# ============================================================
# 17. ALL PROBABILITIES
# ============================================================

print("\nAll Emotion Probabilities:")

for i in range(NUM_CLASSES := len(EMOTIONS)):

    print(
        EMOTIONS[i],
        ":",
        round(
            prediction[0][i] * 100,
            2
        ),
        "%"
    )


print("\nPrediction completed!")