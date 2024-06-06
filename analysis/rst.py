from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from function import *
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

label_map = {label: num for num, label in enumerate(actions)}
# print(label_map)
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
X_train_rf = X_train.reshape(X_train.shape[0], -1)
X_test_rf = X_test.reshape(X_test.shape[0], -1)

# Create and train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train_rf, y_train)

# Generate predictions
y_pred_rf = rf_model.predict(X_test_rf)

# Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred_rf, target_names=actions))

# Confusion Matrix
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix_rf, cmap=plt.cm.Blues)
plt.colorbar()
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.xticks(range(len(actions)), actions, rotation=45)
plt.yticks(range(len(actions)), actions)
plt.show()

# Additional Metrics
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("Random Forest Accuracy:", accuracy_rf)

precision_rf = precision_score(y_test, y_pred_rf, average="weighted")
print("Random Forest Precision:", precision_rf)

recall_rf = recall_score(y_test, y_pred_rf, average="weighted")
print("Random Forest Recall:", recall_rf)

f1_rf = f1_score(y_test, y_pred_rf, average="weighted")
print("Random Forest F1 Score:", f1_rf)
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# Convert labels to one-hot encoded format
y_test_one_hot = label_binarize(y_test, classes=np.unique(y_test))
n_classes = y_test_one_hot.shape[1]

# Compute predicted probabilities
y_pred_prob_rf = rf_model.predict_proba(X_test_rf)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_one_hot[:, i], y_pred_prob_rf[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(
    y_test_one_hot.ravel(), y_pred_prob_rf.ravel()
)
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Plot ROC curve for each class and micro-average ROC curve
plt.figure(figsize=(8, 6))
plt.plot(
    fpr["micro"],
    tpr["micro"],
    label=f'micro-average ROC curve (AUC = {roc_auc["micro"]:.2f})',
    color="deeppink",
    linestyle=":",
    linewidth=4,
)

for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label=f"ROC curve - Class {i} (AUC = {roc_auc[i]:.2f})")

plt.plot([0, 1], [0, 1], "k--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Random Forest Classifier")
plt.legend(loc="lower right")
plt.show()
