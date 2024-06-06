from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from trainmodel import *  # Assuming this contains X_train, X_test, y_train, y_test, and actions

# Reshape the input data for SVM
X_train_svm = X_train.reshape(X_train.shape[0], -1)
X_test_svm = X_test.reshape(X_test.shape[0], -1)

# Create and train the SVM model
svm_model = SVC(
    kernel="linear"
)  # Linear kernel is used here, you can change it as needed
svm_model.fit(X_train_svm, y_train)

# Generate predictions
y_pred_svm = svm_model.predict(X_test_svm)

# Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred_svm, target_names=actions))

# Confusion Matrix
conf_matrix_svm = confusion_matrix(y_test, y_pred_svm)
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix_svm, cmap=plt.cm.Blues)
plt.colorbar()
plt.title("SVM Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.xticks(range(len(actions)), actions, rotation=45)
plt.yticks(range(len(actions)), actions)
plt.show()

# Additional Metrics
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print("SVM Accuracy:", accuracy_svm)

precision_svm = precision_score(y_test, y_pred_svm, average="weighted")
print("SVM Precision:", precision_svm)

recall_svm = recall_score(y_test, y_pred_svm, average="weighted")
print("SVM Recall:", recall_svm)

f1_svm = f1_score(y_test, y_pred_svm, average="weighted")
print("SVM F1 Score:", f1_svm)
