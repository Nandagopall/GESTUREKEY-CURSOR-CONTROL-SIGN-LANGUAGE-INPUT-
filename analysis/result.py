from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score,
    auc,
)
from sklearn.preprocessing import label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from function import *
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
y = np.array(y_clean)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
X_train_rf = X_train.reshape(X_train.shape[0], -1)
X_test_rf = X_test.reshape(X_test.shape[0], -1)

print("yes")


# Generate predictions for Keras model
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=0)

# Classification Report for Keras model
print("LSTM Model Classification Report:")
print(classification_report(y_test, y_pred_classes, target_names=actions))

# Confusion Matrix for Keras model
conf_matrix = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, cmap=plt.cm.Blues)
plt.colorbar()
plt.title("LSTM Model Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.xticks(range(len(actions)), actions, rotation=45)
plt.yticks(range(len(actions)), actions)
plt.show()

# Additional Metrics for Keras model
accuracy = accuracy_score(y_test, y_pred_classes)
print("LSTM Model Accuracy:", accuracy)

precision = precision_score(y_test, y_pred_classes, average="weighted")
print("LSTM Model Precision:", precision)

recall = recall_score(y_test, y_pred_classes, average="weighted")
print("LSTM Model Recall:", recall)

f1 = f1_score(y_test, y_pred_classes, average="weighted")
print("LSTM Model F1 Score:", f1)

# Create and train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train_rf, y_train)

# Generate predictions for Random Forest
y_pred_rf = rf_model.predict(X_test_rf)
y_test_categorical = np.argmax(y_test, axis=0)
y_pred_categorical = np.argmax(y_pred_rf, axis=1)

# Classification Report for Random Forest
print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred_categorical, target_names=actions))

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

# Create and train the SVM model
svm_model = SVC(kernel="linear")
svm_model.fit(X_train_svm, y_train)

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
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print("SVM Accuracy:", accuracy_svm)

precision_svm = precision_score(y_test, y_pred_svm, average="weighted")
print("SVM Precision:", precision_svm)

recall_svm = recall_score(y_test, y_pred_svm, average="weighted")
print("SVM Recall:", recall_svm)

f1_svm = f1_score(y_test, y_pred_svm, average="weighted")
print("SVM F1 Score:", f1_svm)

# Create lists for metrics and models
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

from sklearn.preprocessing import label_binarize

# Convert labels to binary format
y_test_bin = label_binarize(y_test, classes=np.unique(y_test))

# Compute probabilities for each class prediction using softmax
y_pred_prob_lstm = model.predict_proba(X_test)
y_pred_prob_lstm_softmax = np.exp(y_pred_prob_lstm) / np.sum(
    np.exp(y_pred_prob_lstm), axis=1, keepdims=True
)

# Compute ROC curve and ROC area for each class
fpr_lstm = dict()
tpr_lstm = dict()
roc_auc_lstm = dict()
for i in range(len(actions)):
    fpr_lstm[i], tpr_lstm[i], _ = roc_curve(
        y_test_bin[:, i], y_pred_prob_lstm_softmax[:, i]
    )
    roc_auc_lstm[i] = auc(fpr_lstm[i], tpr_lstm[i])

# Compute macro-average ROC curve and ROC area
fpr_lstm["macro"], tpr_lstm["macro"], _ = roc_curve(
    y_test_bin.ravel(), y_pred_prob_lstm_softmax.ravel()
)
roc_auc_lstm["macro"] = auc(fpr_lstm["macro"], tpr_lstm["macro"])

# Plot ROC curve for each class and macro-average ROC curve
plt.figure(figsize=(8, 6))
plt.plot(
    fpr_lstm["macro"],
    tpr_lstm["macro"],
    color="navy",
    linestyle="-",
    label="Macro-average ROC curve (AUC = %0.2f)" % roc_auc_lstm["macro"],
)

for i in range(len(actions)):
    plt.plot(
        fpr_lstm[i],
        tpr_lstm[i],
        linestyle="-",
        label=f"ROC curve - {actions[i]} (AUC = %0.2f)" % roc_auc_lstm[i],
    )

plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - LSTM Model")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()
# Plot ROC curves for each model
plt.figure(figsize=(10, 6))

# Plot ROC curves for Random Forest and SVM
for model_name, model_obj in [("Random Forest", rf_model), ("SVM", svm_model)]:
    y_pred_prob = model_obj.predict_proba(X_test_rf)
    roc_auc = roc_auc_score(y_test, y_pred_prob, average="macro", multi_class="ovr")
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob, pos_label=label_map["action"])
    plt.plot(
        fpr,
        tpr,
        linestyle="-",
        label=f"{model_name} (AUC = {roc_auc:.2f})",
    )

# Plot ROC curve for Keras LSTM model
y_pred_prob_lstm = model.predict_proba(X_test)
y_pred_prob_lstm_softmax = np.exp(y_pred_prob_lstm) / np.sum(
    np.exp(y_pred_prob_lstm), axis=1, keepdims=True
)
roc_auc_lstm = roc_auc_score(
    y_test, y_pred_prob_lstm_softmax, average="macro", multi_class="ovr"
)
fpr_lstm, tpr_lstm, _ = roc_curve(
    y_test, y_pred_prob_lstm_softmax, pos_label=label_map["action"]
)
plt.plot(
    fpr_lstm,
    tpr_lstm,
    linestyle="-",
    label=f"LSTM Model (AUC = {roc_auc_lstm:.2f})",
)

plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curves")
plt.legend()
plt.grid(True)
plt.show()
