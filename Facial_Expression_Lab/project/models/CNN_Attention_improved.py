# ============================================================
# CNN + LIGHTWEIGHT ATTENTION - CK+ EXTENDED  (IMPROVED)
#
# Changes from the original version, and why:
#
#   1. Actually USES the ImageDataGenerator augmentation that
#      Preprocessing.py already builds. CK+ Extended has under
#      ~1000 images, so training on raw, unaugmented arrays
#      overfits fast. Augmenting multiplies the effective
#      training set every epoch.
#
#   2. Applies class_weight (from sklearn's "balanced" scheme)
#      during training. CK+ is very imbalanced (e.g. "contempt"
#      has far fewer samples than "happy"), so the loss was
#      previously dominated by majority classes.
#
#   3. Adds BatchNormalization after every conv layer for
#      faster, more stable convergence.
#
#   4. Adds SpatialDropout2D + L2 weight decay on conv/dense
#      layers to fight overfitting.
#
#   5. Replaces Flatten() -> Dense(128) with
#      GlobalAveragePooling2D() -> Dense(256). This cuts the
#      parameter count drastically (Flatten on a 128-channel
#      feature map produces a huge dense layer that is the
#      single biggest overfitting risk in the original model).
#
#   6. Applies channel attention after BOTH the 2nd and 3rd
#      conv blocks (not just the 3rd), so attention can
#      reweight both mid- and high-level features.
#
#   7. Uses label smoothing (0.1) in the loss - a mild but
#      effective regularizer for small, noisy-label datasets.
#
#   8. Longer training budget (up to 100 epochs) with
#      EarlyStopping (patience 15, monitors val_accuracy) and
#      ReduceLROnPlateau, instead of a short 20-epoch run with
#      patience 5.
#
#   9. Saves the BEST checkpoint (via ModelCheckpoint), not
#      just whatever early stopping happens to restore.
# ============================================================

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

from tensorflow.keras.layers import (
    Input,
    Conv2D,
    BatchNormalization,
    Activation,
    MaxPooling2D,
    SpatialDropout2D,
    GlobalAveragePooling2D,
    Dense,
    Dropout,
    Reshape,
    Multiply,
    Add,
)

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
)

from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# ============================================================
# 1. PATH
# ============================================================

BASE_PATH = r"C:\Users\91994\PycharmProjects\PythonProject7\CK+"

DATA_PATH = os.path.join(BASE_PATH, "preprocessed_data.npz")
MODEL_PATH = os.path.join(BASE_PATH, "cnn_attention.h5")
BEST_MODEL_PATH = os.path.join(BASE_PATH, "cnn_attention_best.h5")
PLOT_PATH = os.path.join(BASE_PATH, "preprocessing_results")

os.makedirs(PLOT_PATH, exist_ok=True)


# ============================================================
# 2. CONFIGURATION
# ============================================================

IMG_SIZE = 48
NUM_CLASSES = 8
BATCH_SIZE = 32
EPOCHS = 100          # early stopping will cut this short
L2_REG = 1e-4
LABEL_SMOOTHING = 0.1


# ============================================================
# 3. LOAD PREPROCESSED DATA
# ============================================================

print("=" * 70)
print("CNN + ATTENTION (IMPROVED)")
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
print("X_train:", X_train.shape, "y_train:", y_train.shape)
print("X_val:  ", X_val.shape, "y_val:  ", y_val.shape)
print("X_test: ", X_test.shape, "y_test: ", y_test.shape)


# ============================================================
# 4. LABELS: KEEP INTEGER VERSION FOR CLASS WEIGHTS,
#    THEN ONE-HOT FOR TRAINING
# ============================================================

y_train_int = y_train.astype("int32")
y_val_int = y_val.astype("int32")
y_test_int = y_test.astype("int32")

y_train_cat = to_categorical(y_train_int, NUM_CLASSES)
y_val_cat = to_categorical(y_val_int, NUM_CLASSES)
y_test_cat = to_categorical(y_test_int, NUM_CLASSES)


# ============================================================
# 5. CLASS WEIGHTS (fixes class imbalance, e.g. "contempt")
# ============================================================

print("\n" + "=" * 70)
print("CLASS WEIGHTS")
print("=" * 70)

train_classes = np.unique(y_train_int)

weights = compute_class_weight(
    class_weight="balanced",
    classes=train_classes,
    y=y_train_int,
)

class_weights = {int(c): float(w) for c, w in zip(train_classes, weights)}

for class_id, weight in class_weights.items():
    print(f"Class {class_id}: {weight:.4f}")


# ============================================================
# 6. DATA AUGMENTATION (train only)
# ============================================================

train_datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.10,
    height_shift_range=0.10,
    zoom_range=0.10,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
)

train_generator = train_datagen.flow(
    X_train, y_train_cat, batch_size=BATCH_SIZE, shuffle=True
)


# ============================================================
# 7. LIGHTWEIGHT CHANNEL ATTENTION (unchanged logic)
# ============================================================

def channel_attention(x, channels, name_prefix):
    attention = GlobalAveragePooling2D(name=f"{name_prefix}_gap")(x)
    attention = Dense(
        max(channels // 8, 4), activation="relu", name=f"{name_prefix}_fc1"
    )(attention)
    attention = Dense(
        channels, activation="sigmoid", name=f"{name_prefix}_fc2"
    )(attention)
    attention = Reshape((1, 1, channels), name=f"{name_prefix}_reshape")(attention)
    attended = Multiply(name=f"{name_prefix}_scale")([x, attention])
    output = Add(name=f"{name_prefix}_residual")([x, attended])
    return output


def conv_block(x, filters, name_prefix, dropout_rate):
    x = Conv2D(
        filters, (3, 3), padding="same",
        kernel_regularizer=l2(L2_REG),
        name=f"{name_prefix}_conv",
    )(x)
    x = BatchNormalization(name=f"{name_prefix}_bn")(x)
    x = Activation("relu", name=f"{name_prefix}_relu")(x)
    x = MaxPooling2D(pool_size=(2, 2), name=f"{name_prefix}_pool")(x)
    x = SpatialDropout2D(dropout_rate, name=f"{name_prefix}_dropout")(x)
    return x


# ============================================================
# 8. BUILD MODEL
# ============================================================

inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 1))

x = conv_block(inputs, 32, "block1", dropout_rate=0.20)
x = conv_block(x, 64, "block2", dropout_rate=0.25)
x = channel_attention(x, 64, "attn2")

x = conv_block(x, 128, "block3", dropout_rate=0.30)
x = channel_attention(x, 128, "attn3")

x = GlobalAveragePooling2D(name="global_pool")(x)

x = Dense(256, kernel_regularizer=l2(L2_REG), name="fc1")(x)
x = BatchNormalization(name="fc1_bn")(x)
x = Activation("relu", name="fc1_relu")(x)
x = Dropout(0.5, name="fc1_dropout")(x)

outputs = Dense(NUM_CLASSES, activation="softmax", name="predictions")(x)

model = Model(inputs=inputs, outputs=outputs, name="cnn_attention_improved")


# ============================================================
# 9. COMPILE
# ============================================================

model.compile(
    optimizer="adam",
    loss=CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
    metrics=["accuracy"],
)

print("\n" + "=" * 70)
print("CNN + ATTENTION (IMPROVED) MODEL SUMMARY")
print("=" * 70)
model.summary()


# ============================================================
# 10. CALLBACKS
# ============================================================

early_stopping = EarlyStopping(
    monitor="val_accuracy",
    patience=15,
    restore_best_weights=True,
    verbose=1,
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=6,
    min_lr=1e-6,
    verbose=1,
)

checkpoint = ModelCheckpoint(
    BEST_MODEL_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1,
)


# ============================================================
# 11. TRAIN (using the augmented generator + class weights)
# ============================================================

print("\n" + "=" * 70)
print("TRAINING CNN + ATTENTION (IMPROVED)")
print("=" * 70)

steps_per_epoch = max(1, len(X_train) // BATCH_SIZE)

history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    validation_data=(X_val, y_val_cat),
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=[early_stopping, reduce_lr, checkpoint],
    verbose=1,
)


# ============================================================
# 12. EVALUATE
# ============================================================

print("\n" + "=" * 70)
print("TESTING CNN + ATTENTION (IMPROVED)")
print("=" * 70)

test_loss, test_accuracy = model.evaluate(X_test, y_test_cat, verbose=1)

print("\n" + "=" * 70)
print("CNN + ATTENTION (IMPROVED) RESULTS")
print("=" * 70)
print("Test Loss:", round(test_loss, 4))
print("Test Accuracy:", round(test_accuracy, 4))
print("Test Accuracy (%):", round(test_accuracy * 100, 2), "%")


# ============================================================
# 13. SAVE FINAL MODEL (best checkpoint already saved separately)
# ============================================================

model.save(MODEL_PATH)
print("\nCNN + Attention saved successfully!")
print("Final model location:", MODEL_PATH)
print("Best checkpoint location:", BEST_MODEL_PATH)


# ============================================================
# 14. ACCURACY GRAPH
# ============================================================

plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("CNN + Attention (Improved) Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()

ACCURACY_PLOT = os.path.join(PLOT_PATH, "cnn_attention_improved_accuracy.png")
plt.savefig(ACCURACY_PLOT, dpi=150)
plt.close()


# ============================================================
# 15. LOSS GRAPH
# ============================================================

plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("CNN + Attention (Improved) Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()

LOSS_PLOT = os.path.join(PLOT_PATH, "cnn_attention_improved_loss.png")
plt.savefig(LOSS_PLOT, dpi=150)
plt.close()


# ============================================================
# 16. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("CNN + ATTENTION (IMPROVED) TRAINING COMPLETED")
print("=" * 70)
print("\nSaved model:", MODEL_PATH)
print("Saved best checkpoint:", BEST_MODEL_PATH)
print("Saved accuracy graph:", ACCURACY_PLOT)
print("Saved loss graph:", LOSS_PLOT)
print("\nREADY FOR COMPARISON")
print("=" * 70)
