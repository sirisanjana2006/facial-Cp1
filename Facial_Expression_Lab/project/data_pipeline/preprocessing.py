# ============================================================
# FACIAL EXPRESSION / DISTRESS PROJECT
# CK+ EXTENDED
#
# PIXELS -> 48x48 IMAGES -> PREPROCESSING
# -> TRAIN / VALIDATION / TEST
#
# GUI-FREE VERSION FOR PYCHARM / WINDOWS
# ============================================================


# ============================================================
# 1. IMPORTS
# ============================================================

import os

# IMPORTANT:
# Use a non-GUI matplotlib backend.
# This prevents the Tkinter error.
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.preprocessing.image import ImageDataGenerator


# ============================================================
# 2. CONFIGURATION
# ============================================================

BASE_PATH = r"C:\Users\91994\PycharmProjects\PythonProject8"

CSV_PATH = os.path.join(
    BASE_PATH,
    "ckextended.csv"
)

OUTPUT_PATH = os.path.join(
    BASE_PATH,
    "processed_images"
)

PLOT_PATH = os.path.join(
    BASE_PATH,
    "preprocessing_results"
)

IMG_SIZE = 48

BATCH_SIZE = 32

NUM_CLASSES = 8


# ============================================================
# 3. CREATE OUTPUT DIRECTORIES
# ============================================================

os.makedirs(
    OUTPUT_PATH,
    exist_ok=True
)

os.makedirs(
    PLOT_PATH,
    exist_ok=True
)


# ============================================================
# 4. START
# ============================================================

print("=" * 70)

print(
    "FACIAL EXPRESSION / DISTRESS"
)

print(
    "CK+ EXTENDED PREPROCESSING"
)

print("=" * 70)

print(
    "TensorFlow version:",
    tf.__version__
)

print(
    "CSV:",
    CSV_PATH
)


# ============================================================
# 5. CHECK CSV
# ============================================================

if not os.path.exists(CSV_PATH):

    raise FileNotFoundError(
        "\nCSV file not found:\n"
        + CSV_PATH
    )


print("\nCSV file found successfully!")


# ============================================================
# 6. LOAD CSV
# ============================================================

print("\nReading CSV...")

df = pd.read_csv(
    CSV_PATH
)


print(
    "CSV loaded successfully!"
)

print(
    "Number of rows:",
    len(df)
)

print(
    "Columns:",
    list(df.columns)
)


# ============================================================
# 7. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "emotion",
    "pixels",
    "Usage"
]


for column in required_columns:

    if column not in df.columns:

        raise ValueError(
            f"Missing required column: {column}"
        )


# ============================================================
# 8. DATASET INFORMATION
# ============================================================

print("\n" + "=" * 70)

print(
    "DATASET INFORMATION"
)

print("=" * 70)


print(
    "\nEmotion classes:"
)

print(
    sorted(
        df["emotion"].unique()
    )
)


print(
    "\nUsage:"
)

print(
    df["Usage"].value_counts()
)


# ============================================================
# 9. PIXELS -> IMAGE
# ============================================================

def pixels_to_image(pixel_string):

    # Convert pixel string to numbers

    pixels = np.array(
        str(pixel_string).split(),
        dtype=np.uint8
    )


    # Check number of pixels

    expected_pixels = (
        IMG_SIZE * IMG_SIZE
    )


    if len(pixels) != expected_pixels:

        raise ValueError(
            f"Expected {expected_pixels} pixels "
            f"but received {len(pixels)}"
        )


    # Convert:
    #
    # 2304 pixels
    #
    # into
    #
    # 48 x 48 image

    image = pixels.reshape(
        IMG_SIZE,
        IMG_SIZE
    )


    return image


# ============================================================
# 10. IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image):

    # --------------------------------------------------------
    # Ensure uint8
    # --------------------------------------------------------

    image = image.astype(
        np.uint8
    )


    # --------------------------------------------------------
    # CLAHE
    # --------------------------------------------------------

    clahe = cv2.createCLAHE(

        clipLimit=2.0,

        tileGridSize=(8, 8)
    )


    image = clahe.apply(
        image
    )


    # --------------------------------------------------------
    # Resize
    # --------------------------------------------------------

    image = cv2.resize(

        image,

        (
            IMG_SIZE,
            IMG_SIZE
        ),

        interpolation=cv2.INTER_AREA
    )


    # --------------------------------------------------------
    # Normalize
    #
    # 0 - 255
    #     ↓
    # 0 - 1
    # --------------------------------------------------------

    image = image.astype(
        np.float32
    )

    image = image / 255.0


    return image


# ============================================================
# 11. CONVERT PIXELS INTO IMAGES
# ============================================================

print("\n" + "=" * 70)

print(
    "CONVERTING PIXELS INTO IMAGES"
)

print("=" * 70)


X = []

y = []

usage_list = []


for index, row in df.iterrows():

    try:

        # Get pixel string

        pixel_string = row[
            "pixels"
        ]


        # Convert pixels to image

        image = pixels_to_image(
            pixel_string
        )


        # Preprocess

        image = preprocess_image(
            image
        )


        # Store image

        X.append(
            image
        )


        # Store emotion

        y.append(
            int(row["emotion"])
        )


        # Store Usage

        usage_list.append(
            str(row["Usage"])
        )


        # Progress

        if (index + 1) % 100 == 0:

            print(
                f"Processed "
                f"{index + 1}/"
                f"{len(df)}"
            )


    except Exception as e:

        print(
            f"Skipping row {index}: {e}"
        )


# ============================================================
# 12. NUMPY CONVERSION
# ============================================================

X = np.array(
    X,
    dtype=np.float32
)


y = np.array(
    y,
    dtype=np.int32
)


usage_list = np.array(
    usage_list
)


# ============================================================
# 13. CONVERSION RESULTS
# ============================================================

print("\n" + "=" * 70)

print(
    "CONVERSION COMPLETED"
)

print("=" * 70)


print(
    "Total images:",
    len(X)
)


print(
    "Image shape:",
    X.shape
)


print(
    "Labels shape:",
    y.shape
)


print(
    "Pixel minimum:",
    X.min()
)


print(
    "Pixel maximum:",
    X.max()
)


# ============================================================
# 14. SAVE ORIGINAL CONVERTED IMAGES
# ============================================================

RAW_IMAGES_PATH = os.path.join(
    OUTPUT_PATH,
    "all_images"
)


os.makedirs(
    RAW_IMAGES_PATH,
    exist_ok=True
)


print("\nSaving converted images...")


for i in range(
    len(X)
):

    image = (
        X[i] * 255
    ).astype(
        np.uint8
    )


    filename = os.path.join(
        RAW_IMAGES_PATH,
        f"image_{i:04d}.png"
    )


    cv2.imwrite(
        filename,
        image
    )


print(
    "Saved:",
    len(X),
    "images"
)


# ============================================================
# 15. SHOW FIRST IMAGES
# ============================================================

print(
    "\nCreating preprocessing visualization..."
)


plt.figure(
    figsize=(12, 6)
)


for i in range(
    min(10, len(X))
):

    plt.subplot(
        2,
        5,
        i + 1
    )


    plt.imshow(
        X[i],
        cmap="gray"
    )


    plt.title(
        f"Emotion {y[i]}"
    )


    plt.axis(
        "off"
    )


plt.suptitle(
    "Pixels Converted to 48x48 Images"
)


plt.tight_layout()


FIRST_IMAGES_PLOT = os.path.join(
    PLOT_PATH,
    "converted_images.png"
)


plt.savefig(
    FIRST_IMAGES_PLOT,
    dpi=150,
    bbox_inches="tight"
)


plt.close()


print(
    "Saved visualization:",
    FIRST_IMAGES_PLOT
)


# ============================================================
# 16. CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)

print(
    "CLASS DISTRIBUTION"
)

print("=" * 70)


classes, counts = np.unique(
    y,
    return_counts=True
)


for class_id, count in zip(
    classes,
    counts
):

    print(
        f"Class {class_id}: "
        f"{count} images"
    )


# ============================================================
# 17. CLASS DISTRIBUTION PLOT
# ============================================================

plt.figure(
    figsize=(10, 5)
)


plt.bar(
    classes,
    counts
)


plt.title(
    "Emotion Class Distribution"
)


plt.xlabel(
    "Emotion Class"
)


plt.ylabel(
    "Number of Images"
)


plt.xticks(
    classes
)


plt.tight_layout()


CLASS_PLOT = os.path.join(
    PLOT_PATH,
    "class_distribution.png"
)


plt.savefig(
    CLASS_PLOT,
    dpi=150,
    bbox_inches="tight"
)


plt.close()


print(
    "Saved class distribution:",
    CLASS_PLOT
)


# ============================================================
# 18. CREATE TRAIN / VALIDATION / TEST MASKS
# ============================================================

usage_lower = np.array(
    [
        str(value).strip().lower()
        for value in usage_list
    ]
)


# Training

train_mask = (
    usage_lower == "training"
)


# PublicTest = validation

val_mask = (
    usage_lower == "publictest"
)


# PrivateTest = test

test_mask = (
    usage_lower == "privatetest"
)


# ============================================================
# 19. SPLIT DATA
# ============================================================

X_train = X[
    train_mask
]

y_train = y[
    train_mask
]


X_val = X[
    val_mask
]

y_val = y[
    val_mask
]


X_test = X[
    test_mask
]

y_test = y[
    test_mask
]


# ============================================================
# 20. PRINT SPLIT
# ============================================================

print("\n" + "=" * 70)

print(
    "DATASET SPLIT"
)

print("=" * 70)


print(
    "Training:",
    X_train.shape,
    y_train.shape
)


print(
    "Validation:",
    X_val.shape,
    y_val.shape
)


print(
    "Testing:",
    X_test.shape,
    y_test.shape
)


# ============================================================
# 21. ADD CNN CHANNEL
# ============================================================

X_train = np.expand_dims(
    X_train,
    axis=-1
)


X_val = np.expand_dims(
    X_val,
    axis=-1
)


X_test = np.expand_dims(
    X_test,
    axis=-1
)


print(
    "\nCNN input shape:",
    X_train.shape
)


# ============================================================
# 22. ONE-HOT ENCODING
# ============================================================

y_train_cat = tf.keras.utils.to_categorical(

    y_train,

    NUM_CLASSES
)


y_val_cat = tf.keras.utils.to_categorical(

    y_val,

    NUM_CLASSES
)


y_test_cat = tf.keras.utils.to_categorical(

    y_test,

    NUM_CLASSES
)


# ============================================================
# 23. CLASS WEIGHTS
# ============================================================

print("\n" + "=" * 70)

print(
    "CLASS WEIGHTS"
)

print("=" * 70)


train_classes = np.unique(
    y_train
)


weights = compute_class_weight(

    class_weight="balanced",

    classes=train_classes,

    y=y_train
)


class_weights = {

    int(class_id): float(weight)

    for class_id, weight in zip(
        train_classes,
        weights
    )

}


for class_id, weight in class_weights.items():

    print(
        f"Class {class_id}: "
        f"{weight:.4f}"
    )


# ============================================================
# 24. DATA AUGMENTATION
# ============================================================

train_datagen = ImageDataGenerator(

    rotation_range=10,

    width_shift_range=0.10,

    height_shift_range=0.10,

    zoom_range=0.10,

    horizontal_flip=True,

    brightness_range=[
        0.8,
        1.2
    ]
)


val_datagen = ImageDataGenerator()

test_datagen = ImageDataGenerator()


# ============================================================
# 25. GENERATORS
# ============================================================

train_generator = train_datagen.flow(

    X_train,

    y_train_cat,

    batch_size=BATCH_SIZE,

    shuffle=True
)


val_generator = val_datagen.flow(

    X_val,

    y_val_cat,

    batch_size=BATCH_SIZE,

    shuffle=False
)


test_generator = test_datagen.flow(

    X_test,

    y_test_cat,

    batch_size=BATCH_SIZE,

    shuffle=False
)


# ============================================================
# 26. SAVE TRAIN / VAL / TEST IMAGES
# ============================================================

def save_dataset_images(
    images,
    labels,
    output_directory,
    prefix
):

    os.makedirs(
        output_directory,
        exist_ok=True
    )


    # Create class directories

    for class_id in range(
        NUM_CLASSES
    ):

        os.makedirs(
            os.path.join(
                output_directory,
                str(class_id)
            ),
            exist_ok=True
        )


    # Save images

    for i in range(
        len(images)
    ):

        image = images[i].squeeze()


        image_uint8 = (
            image * 255
        ).astype(
            np.uint8
        )


        class_directory = os.path.join(

            output_directory,

            str(
                labels[i]
            )
        )


        filename = os.path.join(

            class_directory,

            f"{prefix}_{i:05d}.png"
        )


        cv2.imwrite(

            filename,

            image_uint8
        )


    print(
        f"Saved {len(images)} images:"
    )

    print(
        output_directory
    )


# ============================================================
# TRAIN
# ============================================================

TRAIN_OUTPUT = os.path.join(
    OUTPUT_PATH,
    "train"
)


save_dataset_images(

    X_train,

    y_train,

    TRAIN_OUTPUT,

    "train"
)


# ============================================================
# VALIDATION
# ============================================================

VAL_OUTPUT = os.path.join(
    OUTPUT_PATH,
    "val"
)


save_dataset_images(

    X_val,

    y_val,

    VAL_OUTPUT,

    "val"
)


# ============================================================
# TEST
# ============================================================

TEST_OUTPUT = os.path.join(
    OUTPUT_PATH,
    "test"
)


save_dataset_images(

    X_test,

    y_test,

    TEST_OUTPUT,

    "test"
)


# ============================================================
# 27. AUGMENTATION VISUALIZATION
# ============================================================

sample_image = X_train[0:1]


augmentation_iterator = train_datagen.flow(

    sample_image,

    batch_size=1
)


plt.figure(
    figsize=(12, 6)
)


for i in range(10):

    augmented_image = next(
        augmentation_iterator
    )[0]


    plt.subplot(
        2,
        5,
        i + 1
    )


    plt.imshow(
        augmented_image.squeeze(),
        cmap="gray"
    )


    plt.axis(
        "off"
    )


plt.suptitle(
    "Training Data Augmentation"
)


plt.tight_layout()


AUGMENTATION_PLOT = os.path.join(

    PLOT_PATH,

    "augmentation.png"
)


plt.savefig(

    AUGMENTATION_PLOT,

    dpi=150,

    bbox_inches="tight"
)


plt.close()


print(
    "Saved augmentation plot:",
    AUGMENTATION_PLOT
)


# ============================================================
# 28. SAVE NUMPY DATA
# ============================================================

NPY_PATH = os.path.join(
    BASE_PATH,
    "preprocessed_data.npz"
)


np.savez_compressed(

    NPY_PATH,

    X_train=X_train,

    y_train=y_train,

    X_val=X_val,

    y_val=y_val,

    X_test=X_test,

    y_test=y_test
)


print(
    "\nSaved NumPy dataset:"
)

print(
    NPY_PATH
)


# ============================================================
# 29. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)

print(
    "PREPROCESSING COMPLETED SUCCESSFULLY"
)

print("=" * 70)


print(
    "\nTotal images:",
    len(X)
)


print(
    "Training images:",
    len(X_train)
)


print(
    "Validation images:",
    len(X_val)
)


print(
    "Test images:",
    len(X_test)
)


print(
    "\nImage size:",
    "48 x 48"
)


print(
    "Channels:",
    "1 - Grayscale"
)


print(
    "Normalization:",
    "0 - 1"
)


print(
    "CLAHE:",
    "Applied"
)


print(
    "Data augmentation:",
    "Training only"
)


print(
    "\nCNN input:",
    X_train.shape[1:]
)


print(
    "\nProcessed images:"
)

print(
    OUTPUT_PATH
)


print(
    "\nPlots:"
)

print(
    PLOT_PATH
)


print(
    "\nNumPy dataset:"
)

print(
    NPY_PATH
)


print("\n" + "=" * 70)

print(
    "READY FOR CNN + ATTENTION MODEL"
)

print("=" * 70)