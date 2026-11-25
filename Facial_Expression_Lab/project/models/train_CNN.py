# ============================================================
# BASELINE CNN - CK+ EXTENDED
# STEP 4 + STEP 5
# ============================================================

import os
import numpy as np
import matplotlib

# Prevent GUI/Tkinter problems
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)

from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.utils import to_categorical


# ============================================================
# 1. PATH
# ============================================================

BASE_PATH = r"C:\Users\91994\PycharmProjects\PythonProject7\CK+"

DATA_PATH = os.path.join(
    BASE_PATH,
    "preprocessed_data.npz"
)

MODEL_PATH = os.path.join(
    BASE_PATH,
    "baseline_cnn.h5"
)

PLOT_PATH = os.path.join(
    BASE_PATH,
    "preprocessing_results"
)

os.makedirs(
    PLOT_PATH,
    exist_ok=True
)


# ============================================================
# 2. CONFIGURATION
# ============================================================

IMG_SIZE = 48
NUM_CLASSES = 8
BATCH_SIZE = 32
EPOCHS = 20


# ============================================================
# 3. LOAD PREPROCESSED DATA
# ============================================================

print("=" * 70)
print("BASELINE CNN")
print("=" * 70)

print("\nLoading preprocessed dataset...")

data = np.load(DATA_PATH)

X_train = data["X_train"]
y_train = data["y_train"]

X_val = data["X_val"]
y_val = data["y_val"]

X_test = data["X_test"]
y_test = data["y_test"]


print("\nDataset loaded successfully!")

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("X_val:", X_val.shape)
print("y_val:", y_val.shape)

print("X_test:", X_test.shape)
print("y_test:", y_test.shape)


# ============================================================
# 4. CHECK DATA
# ============================================================

print("\n" + "=" * 70)
print("DATA CHECK")
print("=" * 70)

print("Pixel minimum:", X_train.min())
print("Pixel maximum:", X_train.max())

print("Image size:", X_train.shape[1:])
print("Number of classes:", NUM_CLASSES)


# ============================================================
# 5. CONVERT LABELS TO ONE-HOT ENCODING
# ============================================================

print("\nConverting labels to one-hot encoding...")

y_train = to_categorical(
    y_train,
    NUM_CLASSES
)

y_val = to_categorical(
    y_val,
    NUM_CLASSES
)

y_test = to_categorical(
    y_test,
    NUM_CLASSES
)


print("New y_train shape:", y_train.shape)
print("New y_val shape:", y_val.shape)
print("New y_test shape:", y_test.shape)


# ============================================================
# 6. CREATE BASELINE CNN
# ============================================================

print("\n" + "=" * 70)
print("CREATING BASELINE CNN")
print("=" * 70)


model = Sequential()


# ------------------------------------------------------------
# Convolution Layer 1
# ------------------------------------------------------------

model.add(
    Conv2D(
        32,
        (3, 3),
        activation="relu",
        input_shape=(IMG_SIZE, IMG_SIZE, 1)
    )
)


# ------------------------------------------------------------
# Pooling Layer 1
# ------------------------------------------------------------

model.add(
    MaxPooling2D(
        pool_size=(2, 2)
    )
)


# ------------------------------------------------------------
# Convolution Layer 2
# ------------------------------------------------------------

model.add(
    Conv2D(
        64,
        (3, 3),
        activation="relu"
    )
)


# ------------------------------------------------------------
# Pooling Layer 2
# ------------------------------------------------------------

model.add(
    MaxPooling2D(
        pool_size=(2, 2)
    )
)


# ------------------------------------------------------------
# Convolution Layer 3
# ------------------------------------------------------------

model.add(
    Conv2D(
        128,
        (3, 3),
        activation="relu"
    )
)


# ------------------------------------------------------------
# Pooling Layer 3
# ------------------------------------------------------------

model.add(
    MaxPooling2D(
        pool_size=(2, 2)
    )
)


# ------------------------------------------------------------
# Flatten
# ------------------------------------------------------------

model.add(
    Flatten()
)


# ------------------------------------------------------------
# Fully Connected Layer
# ------------------------------------------------------------

model.add(
    Dense(
        128,
        activation="relu"
    )
)


# ------------------------------------------------------------
# Dropout
# ------------------------------------------------------------

model.add(
    Dropout(0.5)
)


# ------------------------------------------------------------
# Output Layer
# ------------------------------------------------------------

model.add(
    Dense(
        NUM_CLASSES,
        activation="softmax"
    )
)


# ============================================================
# 7. COMPILE MODEL
# ============================================================

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# 8. SHOW MODEL
# ============================================================

print("\nCNN Model Summary:")

model.summary()


# ============================================================
# 9. EARLY STOPPING
# ============================================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)


# ============================================================
# 10. TRAIN CNN
# ============================================================

print("\n" + "=" * 70)
print("TRAINING BASELINE CNN")
print("=" * 70)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[
        early_stopping
    ],
    verbose=1
)


# ============================================================
# 11. EVALUATE ON TEST DATA
# ============================================================

print("\n" + "=" * 70)
print("TESTING BASELINE CNN")
print("=" * 70)

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=1
)


# ============================================================
# 12. PRINT RESULTS
# ============================================================

print("\n" + "=" * 70)
print("BASELINE CNN RESULTS")
print("=" * 70)

print(
    "Test Loss:",
    round(test_loss, 4)
)

print(
    "Test Accuracy:",
    round(test_accuracy, 4)
)

print(
    "Test Accuracy (%):",
    round(test_accuracy * 100, 2),
    "%"
)


# ============================================================
# 13. SAVE MODEL
# ============================================================

model.save(
    MODEL_PATH
)

print("\nBaseline CNN saved successfully!")

print(
    "Model location:",
    MODEL_PATH
)


# ============================================================
# 14. TRAINING ACCURACY GRAPH
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title(
    "Baseline CNN Accuracy"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Accuracy"
)

plt.legend()

plt.tight_layout()


ACCURACY_PLOT = os.path.join(
    PLOT_PATH,
    "baseline_cnn_accuracy.png"
)

plt.savefig(
    ACCURACY_PLOT,
    dpi=150
)

plt.close()


# ============================================================
# 15. TRAINING LOSS GRAPH
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title(
    "Baseline CNN Loss"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Loss"
)

plt.legend()

plt.tight_layout()


LOSS_PLOT = os.path.join(
    PLOT_PATH,
    "baseline_cnn_loss.png"
)

plt.savefig(
    LOSS_PLOT,
    dpi=150
)

plt.close()


# ============================================================
# 16. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("BASELINE CNN TRAINING COMPLETED")
print("=" * 70)

print("\nSaved model:")
print(MODEL_PATH)

print("\nSaved accuracy graph:")
print(ACCURACY_PLOT)

print("\nSaved loss graph:")
print(LOSS_PLOT)

print("\nREADY FOR CNN + ATTENTION MODEL")
print("=" * 70)