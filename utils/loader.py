import numpy as np
import pandas as pd
import os

# Function: load_dataset_UCI
# This function loads the UCI HAR dataset from the specified base path.
# It reads the training and test data, including features, labels, and subject IDs.
def load_dataset_UCI(base_path="UCI HAR Dataset", minimal=False):
    # Load training data
    X_train = np.loadtxt(f"{base_path}/train/X_train.txt")
    y_train = np.loadtxt(f"{base_path}/train/y_train.txt").astype(int)
    subject_train = np.loadtxt(f"{base_path}/train/subject_train.txt").astype(int)

    # Load test data
    X_test = np.loadtxt(f"{base_path}/test/X_test.txt")
    y_test = np.loadtxt(f"{base_path}/test/y_test.txt").astype(int)
    subject_test = np.loadtxt(f"{base_path}/test/subject_test.txt").astype(int)

    # If minimal dataset is requested, select only a subset of features
    if minimal:
        # Select only the minimal features
        selected_features = [0, 1, 2, 3, 4, 5, 120, 121, 122, 123, 124, 125]
        X_train = X_train[:, selected_features]
        X_test = X_test[:, selected_features]

    return {
        "X_train": X_train, "y_train": y_train, "subject_train": subject_train,
        "X_test": X_test, "y_test": y_test, "subject_test": subject_test,
    }