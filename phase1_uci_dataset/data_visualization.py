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
    

# Visualize the distribution of classes in the training set and test set
def visualize_class_distribution(y_train, y_test):
    y_label_train = pd.Series(y_train).map(label_map)
    y_label_test = pd.Series(y_test).map(label_map)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.countplot(x=y_train, palette="viridis", hue=y_label_train)
    plt.title("Class Distribution in Training Set")
    plt.xlabel("Class")
    plt.ylabel("Count")

    plt.subplot(1, 2, 2)
    sns.countplot(x=y_test, palette="viridis", hue=y_label_test)
    plt.title("Class Distribution in Test Set")
    plt.xlabel("Class")
    plt.ylabel("Count")

    plt.tight_layout()
    plt.savefig("phase1_uci_dataset/figures/class_distribution.png")
    print("==========================")
    print("Visualization saved as 'phase1_uci_dataset/figures/class_distribution.png'")
    print("==========================\n\n")

# Pretty print the dataset information from 'UCI HAR Dataset/features_info.txt'
def pretty_print_features_info():
    print("==========================")
    print("📊 Features Information:")
    print("==========================")
    with open("UCI HAR Dataset/features_info.txt", "r") as f:
        features_info = f.read()
    print(features_info)
    print("\n\n")
    
# Visualize the dataset using PCA and t-SNE
def visualize_dataset(X_train, y_train):
    y_labels = pd.Series(y_train).map(label_map)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_train)

    plt.figure(figsize=(12, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y_labels, palette="viridis")
    plt.title("PCA of UCI HAR Dataset")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.savefig("phase1_uci_dataset/figures/pca.png")

    tsne = TSNE(n_components=2, perplexity=30)
    X_tsne = tsne.fit_transform(X_train)

    plt.figure(figsize=(12, 6))
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y_labels, palette="viridis")
    plt.title("t-SNE of UCI HAR Dataset")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.savefig("phase1_uci_dataset/figures/tsne.png")
    print("==========================")
    print("Visualization saved as 'phase1_uci_dataset/figures/pca.png' and 'phase1_uci_dataset/figures/tsne.png'")
    print("==========================\n\n")
    

label_map = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING"
}
data = load_dataset()
pretty_print_dataset_info(data)
visualize_class_distribution(data['y_train'], data['y_test'])
pretty_print_features_info()
visualize_dataset(data['X_train'], data['y_train'])