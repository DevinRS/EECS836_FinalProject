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
from utils.loader import preprocess_lab_data
from utils.trainer import train_and_evaluate_model

# Load dataset
# df = preprocess_lab_data(base_path="Labeled", window_size=128, step_size=64, save_path=None)
df = pd.read_csv("phase3_lab_data/dataset/lab_W128_S64.csv")
Y = df['Task']
X = df.drop(columns=['Task'])
# Encode the labels
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
Y = le.fit_transform(Y)
# print the label mapping
print("📚 Label mapping:")
for i, label in enumerate(le.classes_):
    print(f"{i}: {label}")
# Split the data into training and testing sets making sure to keep the same ratio of classes in both sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)
# convert to numpy arrays
X_train = X_train.to_numpy()
X_test = X_test.to_numpy()

# Initialize models
logistic_model = LogisticRegression(max_iter=5000, random_state=42)
svm_model = SVC(random_state=42)
rf_model = RandomForestClassifier(random_state=42)
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

# Train only
logistic_model = train_and_evaluate_model(logistic_model, X_train, y_train, X_test, y_test, evaluate=False, save_path=None, remap=False)
svm_model = train_and_evaluate_model(svm_model, X_train, y_train, X_test, y_test, evaluate=False, save_path=None, remap=False)
rf_model = train_and_evaluate_model(rf_model, X_train, y_train, X_test, y_test, evaluate=False, save_path=None, remap=False)
xgb_model = train_and_evaluate_model(xgb_model, X_train, y_train, X_test, y_test, evaluate=False, save_path=None, remap=False)

# Imports for saving models
import joblib
import os

def save_and_get_model_size(model, model_name, path="phase3_lab_data/models"):
    os.makedirs(path, exist_ok=True)
    filepath = os.path.join(path, f"{model_name}.pkl")
    
    # Save the model
    joblib.dump(model, filepath)
    
    # Get file size in MB
    size_kb = os.path.getsize(filepath) / 1024
    print(f"💾 {model_name} size: {size_kb:.2f} KB")
    return size_kb

logistic_size = save_and_get_model_size(logistic_model, "LogisticRegression")
svm_size = save_and_get_model_size(svm_model, "SVM")
rf_size = save_and_get_model_size(rf_model, "RandomForest")
xgb_size = save_and_get_model_size(xgb_model, "XGBoost")

# print model sizes
print(f"\n📦 Model Sizes:")
print(f"Logistic Regression → Size: {logistic_size:.2f} KB")
print(f"SVM                 → Size: {svm_size:.2f} KB")
print(f"Random Forest       → Size: {rf_size:.2f} KB")
print(f"XGBoost             → Size: {xgb_size:.2f} KB")

# plot it as bar chart
plt.figure(figsize=(10, 6))
plt.bar(["Logistic Regression", "SVM", "Random Forest", "XGBoost"], 
        [logistic_size, svm_size, rf_size, xgb_size], color=['blue', 'orange', 'green', 'red'])
plt.title("Model Sizes")
plt.ylabel("Size (KB)")
plt.xlabel("Models")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("phase3_lab_data/figures/model_sizes.png")
plt.clf()