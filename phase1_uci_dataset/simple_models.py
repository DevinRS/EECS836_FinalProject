import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# import for: Logistic Regression, SVM, Random Forest, XGBoost
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from utils.loader import load_dataset_UCI
from utils.trainer import train_and_evaluate_model


# Load dataset
data = load_dataset_UCI(base_path="UCI HAR Dataset", minimal=False)
X_train, y_train, subject_train = data["X_train"], data["y_train"], data["subject_train"]
X_test, y_test, subject_test = data["X_test"], data["y_test"], data["subject_test"]

# Initialize models
logistic_model = LogisticRegression(max_iter=1000, random_state=42)
svm_model = SVC(random_state=42)
rf_model = RandomForestClassifier(random_state=42)
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

logistic_accuracy = train_and_evaluate_model(logistic_model, X_train, y_train, X_test, y_test, evaluate=True, save_path="phase1_uci_dataset/figures")
svm_accuracy = train_and_evaluate_model(svm_model, X_train, y_train, X_test, y_test, evaluate=True, save_path="phase1_uci_dataset/figures")
rf_accuracy = train_and_evaluate_model(rf_model, X_train, y_train, X_test, y_test, evaluate=True, save_path="phase1_uci_dataset/figures")
xgb_accuracy = train_and_evaluate_model(xgb_model, X_train, y_train, X_test, y_test, evaluate=True, save_path="phase1_uci_dataset/figures")

print(f"\n📊 Final Results:")
print(f"Logistic Regression → Accuracy: {logistic_accuracy:.2f}")
print(f"SVM                 → Accuracy: {svm_accuracy:.2f}")
print(f"Random Forest       → Accuracy: {rf_accuracy:.2f}")
print(f"XGBoost             → Accuracy: {xgb_accuracy:.2f}")


# Plot accuracies
accuracies = [logistic_accuracy, svm_accuracy, rf_accuracy, xgb_accuracy]
models = ["Logistic Regression", "SVM", "Random Forest", "XGBoost"]
plt.figure(figsize=(10, 6))
sns.barplot(x=models, y=accuracies)
# add the accuracy value on top of the bar
for i, acc in enumerate(accuracies):
    plt.text(i, acc + 0.01, f"{acc:.2f}", ha='center', va='bottom')
plt.title("Model Accuracies")
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("phase1_uci_dataset/figures/model_accuracies.png")
print("==========================")
print(f"Visualization saved as 'phase1_uci_dataset/figures/model_accuracies.png'")
print("==========================\n\n")
