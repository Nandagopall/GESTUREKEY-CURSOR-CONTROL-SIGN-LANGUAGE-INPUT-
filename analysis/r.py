from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from function import *  # Assuming this contains X_train, X_test, y_train, y_test, actions
import os

# Load your trained model
model = load_model("model.h5")

# label_map creation if not defined already
label_map = {label: num for num, label in enumerate(actions)}

# Loading data and labels
sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(
                os.path.join(
                    DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)
                ),
                allow_pickle=True,
            )
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

# Convert lists in sequences to numpy arrays
sequences_arrays = [np.array(seq) for seq in sequences]

# Determine the shapes of all elements in sequences
shapes = [seq.shape for seq in sequences_arrays]

# Find the most common shape
most_common_shape = max(set(shapes), key=shapes.count)

# Identify elements with shapes different from the most common shape
indices_with_different_shape = [
    i for i, shape in enumerate(shapes) if shape != most_common_shape
]

# Remove sequences with different shapes from X and their corresponding labels from y
X_clean = [
    sequences_arrays[i]
    for i in range(len(sequences_arrays))
    if i not in indices_with_different_shape
]
y_clean = [
    labels[i] for i in range(len(labels)) if i not in indices_with_different_shape
]

# Convert sequences_clean to a numpy array
X = np.array(X_clean)
y = to_categorical(y_clean).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

X_train_rf = X_train.reshape(X_train.shape[0], -1)
X_test_rf = X_test.reshape(X_test.shape[0], -1)

# Load your trained model
model = load_model("model.h5")

# Evaluate model on test data
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# Generate predictions for Keras model
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

# Classification Report for Keras model
print("LSTM Classification Report:")
print(classification_report(y_test_classes, y_pred_classes, target_names=actions))

# Confusion Matrix for Keras model
conf_matrix = confusion_matrix(y_test_classes, y_pred_classes)
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, cmap=plt.cm.Blues)
plt.colorbar()
plt.title("LSTM Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.xticks(range(len(actions)), actions, rotation=45)
plt.yticks(range(len(actions)), actions)
plt.show()

# Additional Metrics for Keras model
accuracy = accuracy_score(y_test_classes, y_pred_classes)
print("LSTM Accuracy:", accuracy)

precision = precision_score(y_test_classes, y_pred_classes, average="weighted")
print("LSTM Model Precision:", precision)

recall = recall_score(y_test_classes, y_pred_classes, average="weighted")
print("LSTM Model Recall:", recall)

f1 = f1_score(y_test_classes, y_pred_classes, average="weighted")
print("LSTM Model F1 Score:", f1)

# Visualize some predictions for Keras model
for i in range(10):
    print(
        f"Sample {i+1}: Predicted={actions[y_pred_classes[i]]}, Actual={actions[y_test_classes[i]]}"
    )

# Reshape the input data for Random Forest
X_train_rf = X_train.reshape(X_train.shape[0], -1)
X_test_rf = X_test.reshape(X_test.shape[0], -1)

# Create and train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train_rf, y_train)

# Generate predictions for Random Forest
y_pred_rf = rf_model.predict(X_test_rf)
y_test_categorical = np.argmax(y_test, axis=1)
y_pred_categorical = np.argmax(y_pred_rf, axis=1)
# Classification Report for Random Forest
print("Random Forest Classification Report:")
print(
    classification_report(y_test_categorical, y_pred_categorical, target_names=actions)
)

# Confusion Matrix for Random Forest
conf_matrix_rf = confusion_matrix(y_test_categorical, y_pred_categorical)
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix_rf, cmap=plt.cm.Blues)
plt.colorbar()
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.xticks(range(len(actions)), actions, rotation=45)
plt.yticks(range(len(actions)), actions)
plt.show()

# Additional Metrics for Random Forest
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("Random Forest Accuracy:", accuracy_rf)

precision_rf = precision_score(y_test, y_pred_rf, average="weighted")
print("Random Forest Precision:", precision_rf)

recall_rf = recall_score(y_test, y_pred_rf, average="weighted")
print("Random Forest Recall:", recall_rf)

f1_rf = f1_score(y_test, y_pred_rf, average="weighted")
print("Random Forest F1 Score:", f1_rf)

# Reshape the input data for SVM
X_train_svm = X_train.reshape(X_train.shape[0], -1)
X_test_svm = X_test.reshape(X_test.shape[0], -1)
y_train_categorical = np.argmax(y_train, axis=1)
# Create and train the SVM model
svm_model = SVC(kernel="linear")
svm_model.fit(X_train_svm, y_train_categorical)

# Generate predictions for SVM
y_pred_svm = svm_model.predict(X_test_svm)

# Classification Report for SVM
print("SVM Classification Report:")
print(classification_report(y_test_categorical, y_pred_svm, target_names=actions))

# Confusion Matrix for SVM
conf_matrix_svm = confusion_matrix(y_test_categorical, y_pred_svm)
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix_svm, cmap=plt.cm.Blues)
plt.colorbar()
plt.title("SVM Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.xticks(range(len(actions)), actions, rotation=45)
plt.yticks(range(len(actions)), actions)
plt.show()

# Additional Metrics for SVM
accuracy_svm = accuracy_score(y_test_categorical, y_pred_svm)
print("SVM Accuracy:", accuracy_svm)

precision_svm = precision_score(y_test_categorical, y_pred_svm, average="weighted")
print("SVM Precision:", precision_svm)

recall_svm = recall_score(y_test_categorical, y_pred_svm, average="weighted")
print("SVM Recall:", recall_svm)

f1_svm = f1_score(y_test_categorical, y_pred_svm, average="weighted")
print("SVM F1 Score:", f1_svm)

models = ["Random Forest", "SVM", "LSTM Model"]
metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]

# Metrics data for each model
model_metrics = {
    "Random Forest": [accuracy_rf, precision_rf, recall_rf, f1_rf],
    "SVM": [accuracy_svm, precision_svm, recall_svm, f1_svm],
    "LSTM Model": [accuracy, precision, recall, f1],
}

plt.figure(figsize=(10, 6))

# Plot lines for each metric
for i, metric in enumerate(metrics):
    plt.plot(
        models, [model_metrics[model][i] for model in models], marker="o", label=metric
    )

plt.xlabel("Models")
plt.ylabel("Scores")
plt.title("Performance Metrics Comparison")
plt.legend()
plt.grid(True)
plt.show()
