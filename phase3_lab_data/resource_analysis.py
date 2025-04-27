import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

# import for: Logistic Regression, SVM, Random Forest, XGBoost
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve, auc

def plot_roc_curve(y_test, y_score, model_name):
    # Binarize the output
    y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
    n_classes = y_test_bin.shape[1]

    # Compute ROC curve and AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute macro-average AUC
    roc_auc["macro"] = roc_auc_score(y_test_bin, y_score, average="macro")

    # Plot all ROC curves
    plt.figure()
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f"Class {i} (AUC = {roc_auc[i]:.2f})")

    plt.plot([0, 1], [0, 1], "k--", label="Chance")
    plt.title(f"ROC Curve for {model_name} (Macro AUC = {roc_auc['macro']:.2f})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.savefig(f"phase3_lab_data/figures/roc_curve_{model_name}.png")
    plt.close()

    return roc_auc["macro"]

# Define a function to train and evaluate a model
def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, evaluate=True):
    # get model name
    model_name = model.__class__.__name__
    print("==========================")
    print(f"⚙️ Training {model_name}...")
    print("==========================")

    # adjustment for xgboost
    if model_name == "XGBClassifier":
        y_test = y_test - 1
        y_train = y_train - 1

    # Train the model
    model.fit(X_train, y_train)

    if not evaluate:
        return model
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Print classification report
    print(classification_report(y_test, y_pred))
    
    # Print confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix for {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(f"phase3_lab_data/figures/confusion_matrix_{model_name}.png")
    plt.clf()

    # AUC-ROC (macro)
    y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
    
    # Use predict_proba or decision_function
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test)
    else:
        raise ValueError(f"{model_name} does not support probability predictions for AUC.")

    # Plot and get macro AUC
    macro_auc = plot_roc_curve(y_test, y_score, model_name)
    print(f"🟢 Macro-average ROC AUC for {model_name}: {macro_auc:.4f}")

    return accuracy, macro_auc


# Load dataset
# Load dataset
data = pd.read_csv("phase3_lab_data/dataset/S1_WS128_SS64.csv")
y = data["Task"]
X = data.drop(columns=["Task"])

# Encode the labels
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
# remap from -1, 0, 1 to 0, 1, 2
y = y + 1

# Print the encoding mapping
print("📚 Encoding mapping:")
for i, label in enumerate(le.classes_):
    print(f"{label} -> {i}")

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)



# Initialize models
logistic_model = LogisticRegression(max_iter=5000, random_state=42)
svm_model = SVC(random_state=42)
rf_model = RandomForestClassifier(random_state=42)
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

# Train only
logistic_model = train_and_evaluate_model(logistic_model, X_train, y_train, X_test, y_test, evaluate=False)
svm_model = train_and_evaluate_model(svm_model, X_train, y_train, X_test, y_test, evaluate=False)
rf_model = train_and_evaluate_model(rf_model, X_train, y_train, X_test, y_test, evaluate=False)
xgb_model = train_and_evaluate_model(xgb_model, X_train, y_train, X_test, y_test, evaluate=False)

# Imports for saving models
import joblib
import os

def save_and_get_model_size(model, model_name, path="phase3_lab_data/models"):
    os.makedirs(path, exist_ok=True)
    filepath = os.path.join(path, f"{model_name}.pkl")
    
    # Save the model
    joblib.dump(model, filepath)
    
    # Get file size in MB
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"💾 {model_name} size: {size_mb:.2f} MB")
    return size_mb

logistic_size = save_and_get_model_size(logistic_model, "LogisticRegression")
svm_size = save_and_get_model_size(svm_model, "SVM")
rf_size = save_and_get_model_size(rf_model, "RandomForest")
xgb_size = save_and_get_model_size(xgb_model, "XGBoost")

# print model sizes
print(f"\n📦 Model Sizes:")
print(f"Logistic Regression → Size: {logistic_size:.2f} MB")
print(f"SVM                 → Size: {svm_size:.2f} MB")
print(f"Random Forest       → Size: {rf_size:.2f} MB")
print(f"XGBoost             → Size: {xgb_size:.2f} MB")

# plot it as bar chart
plt.figure(figsize=(10, 6))
plt.bar(["Logistic Regression", "SVM", "Random Forest", "XGBoost"], 
        [logistic_size, svm_size, rf_size, xgb_size], color=['blue', 'orange', 'green', 'red'])
plt.title("Model Sizes")
plt.ylabel("Size (MB)")
plt.xlabel("Models")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("phase3_lab_data/figures/model_sizes.png")
plt.clf()