import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report

# Load test data
data = np.load(
    r"C:\Users\91994\PycharmProjects\PythonProject7\CK+\preprocessed_data.npz"
)

X_test = data["X_test"]
y_test = data["y_test"]

# Load models
cnn = load_model(
    r"C:\Users\91994\PycharmProjects\PythonProject7\CK+\baseline_cnn.h5"
)

attention = load_model(
    r"C:\Users\91994\PycharmProjects\PythonProject7\CK+\cnn_attention.h5"
)

# Predictions
cnn_prob = cnn.predict(X_test, verbose=0)
attention_prob = attention.predict(X_test, verbose=0)

cnn_pred = np.argmax(cnn_prob, axis=1)
attention_pred = np.argmax(attention_prob, axis=1)

# If labels are one-hot encoded
if len(y_test.shape) > 1:
    y_true = np.argmax(y_test, axis=1)
else:
    y_true = y_test

# Accuracy
cnn_accuracy = np.mean(cnn_pred == y_true)
attention_accuracy = np.mean(attention_pred == y_true)

print("\n========== ACCURACY ==========")
print("CNN + Attention Accuracy          :", cnn_accuracy)
print("CNN       :", attention_accuracy)

# Compare predictions
different = np.sum(cnn_pred != attention_pred)

print("\n========== PREDICTION COMPARISON ==========")
print("Total test images     :", len(y_true))
print("Different predictions :", different)
print("Same predictions      :", len(y_true) - different)

# Confusion matrices
print("\n========== CNN CONFUSION MATRIX ==========")
print(confusion_matrix(y_true, cnn_pred))

print("\n========== CNN + ATTENTION CONFUSION MATRIX ==========")
print(confusion_matrix(y_true, attention_pred))

# Classification reports
print("\n========== CNN REPORT ==========")
print(classification_report(y_true, cnn_pred, zero_division=0))

print("\n========== CNN + ATTENTION REPORT ==========")
print(classification_report(y_true, attention_pred, zero_division=0))