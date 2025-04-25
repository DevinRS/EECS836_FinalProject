import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# import for: Logistic Regression, SVM, Random Forest, XGBoost
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve, auc

# Dataset minimal loading function with subset of features
# 1 tBodyAcc-mean()-X
# 2 tBodyAcc-mean()-Y
# 3 tBodyAcc-mean()-Z
# 4 tBodyAcc-std()-X
# 5 tBodyAcc-std()-Y
# 6 tBodyAcc-std()-Z
# 121 tBodyGyro-mean()-X
# 122 tBodyGyro-mean()-Y
# 123 tBodyGyro-mean()-Z
# 124 tBodyGyro-std()-X
# 125 tBodyGyro-std()-Y
# 126 tBodyGyro-std()-Z
# 201 tBodyAccMag-mean()
# 240 tBodyGyroMag-mean()
def load_minimal_dataset(base_path="UCI HAR Dataset"):
    # Load training data
    X_train = np.loadtxt(f"{base_path}/train/X_train.txt")
    y_train = np.loadtxt(f"{base_path}/train/y_train.txt").astype(int)
    subject_train = np.loadtxt(f"{base_path}/train/subject_train.txt").astype(int)

    # Load test data
    X_test = np.loadtxt(f"{base_path}/test/X_test.txt")
    y_test = np.loadtxt(f"{base_path}/test/y_test.txt").astype(int)
    subject_test = np.loadtxt(f"{base_path}/test/subject_test.txt").astype(int)

    # Select only the minimal features
    selected_features = [0, 1, 2, 3, 4, 5, 120, 121, 122, 123, 124, 125, 200, 239]
    X_train = X_train[:, selected_features]
    X_test = X_test[:, selected_features]

    return {
        "X_train": X_train, "y_train": y_train, "subject_train": subject_train,
        "X_test": X_test, "y_test": y_test, "subject_test": subject_test,
    }

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
    plt.savefig(f"phase2_uci_minimal/figures/roc_curve_{model_name}.png")
    plt.close()

    return roc_auc["macro"]

# Define a function to train and evaluate a model
def train_and_evaluate_model(model, X_train, y_train, X_test, y_test):
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
    plt.savefig(f"phase2_uci_minimal/figures/confusion_matrix_{model_name}.png")
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
data = load_minimal_dataset()
X_train, y_train, subject_train = data["X_train"], data["y_train"], data["subject_train"]
X_test, y_test, subject_test = data["X_test"], data["y_test"], data["subject_test"]

# Initialize models
logistic_model = LogisticRegression(max_iter=1000, random_state=42)
svm_model = SVC(random_state=42)
rf_model = RandomForestClassifier(random_state=42)
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

logistic_accuracy, logistic_auc = train_and_evaluate_model(logistic_model, X_train, y_train, X_test, y_test)
svm_accuracy, svm_auc = train_and_evaluate_model(svm_model, X_train, y_train, X_test, y_test)
rf_accuracy, rf_auc = train_and_evaluate_model(rf_model, X_train, y_train, X_test, y_test)
xgb_accuracy, xgb_auc = train_and_evaluate_model(xgb_model, X_train, y_train, X_test, y_test)

print(f"\n📊 Final Results:")
print(f"Logistic Regression → Accuracy: {logistic_accuracy:.2f}, Macro AUC: {logistic_auc:.4f}")
print(f"SVM                 → Accuracy: {svm_accuracy:.2f}, Macro AUC: {svm_auc:.4f}")
print(f"Random Forest       → Accuracy: {rf_accuracy:.2f}, Macro AUC: {rf_auc:.4f}")
print(f"XGBoost             → Accuracy: {xgb_accuracy:.2f}, Macro AUC: {xgb_auc:.4f}")

# Create two bar graph side to side one for accuracy and one for macro AUC
def plot_results(accuracies, aucs):
    models = list(accuracies.keys())
    accuracy_values = list(accuracies.values())
    auc_values = list(aucs.values())

    x = np.arange(len(models))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, accuracy_values, width, label='Accuracy')
    rects2 = ax.bar(x + width/2, auc_values, width, label='Macro AUC')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()

    plt.savefig("phase2_uci_minimal/figures/model_performance_comparison.png")
# Store accuracies and AUCs in dictionaries
accuracies = {
    "Logistic Regression": logistic_accuracy,
    "SVM": svm_accuracy,
    "Random Forest": rf_accuracy,
    "XGBoost": xgb_accuracy
}
aucs = {
    "Logistic Regression": logistic_auc,
    "SVM": svm_auc,
    "Random Forest": rf_auc,
    "XGBoost": xgb_auc
}
# Plot results
plot_results(accuracies, aucs)
