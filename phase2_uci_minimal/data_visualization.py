import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Dataset loading function
def load_dataset(base_path="UCI HAR Dataset"):
    # Load training data
    X_train = np.loadtxt(f"{base_path}/train/X_train.txt")
    y_train = np.loadtxt(f"{base_path}/train/y_train.txt").astype(int)
    subject_train = np.loadtxt(f"{base_path}/train/subject_train.txt").astype(int)

    # Load test data
    X_test = np.loadtxt(f"{base_path}/test/X_test.txt")
    y_test = np.loadtxt(f"{base_path}/test/y_test.txt").astype(int)
    subject_test = np.loadtxt(f"{base_path}/test/subject_test.txt").astype(int)

    return {
        "X_train": X_train, "y_train": y_train, "subject_train": subject_train,
        "X_test": X_test, "y_test": y_test, "subject_test": subject_test,
    }

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

# Pretty print the dataset information: number of samples, number of features, number of subjects, number of classes
def pretty_print_dataset_info(data):
    print("==========================")
    print("📝 Dataset Information:")
    print("==========================")
    print(f"Number of training samples: {data['X_train'].shape[0]}")
    print(f"Number of test samples: {data['X_test'].shape[0]}")
    print(f"Number of features: {data['X_train'].shape[1]}")
    print(f"Number of subjects: {len(np.unique(data['subject_train']))}")
    print(f"Number of classes: {len(np.unique(data['y_train']))}")
    print("\n\n")
    
# Example usage
print("Normal Dataset: ")
data = load_dataset()
pretty_print_dataset_info(data)

print("Minimal Dataset: ")
data_minimal = load_minimal_dataset()
pretty_print_dataset_info(data_minimal)