import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Function: visualize_class_distribution
# This function visualizes the distribution of classes in the training and test sets.
def visualize_class_distribution(y_train, y_test, label_map, save_path=None):
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
    if save_path:
        plt.savefig(f"{save_path}/class_distribution.png")
        print("==========================")
        print(f"Visualization saved as '{save_path}/class_distribution.png'")
        print("==========================\n\n")

# Function: visualize_dataset_2D
# This function visualizes the dataset in 2D using PCA and t-SNE.
# It saves the visualizations as PNG files.
def visualize_dataset_2D(X_train, y_train, label_map, save_path=None):
    y_labels = pd.Series(y_train).map(label_map)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_train)

    plt.figure(figsize=(12, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y_labels, palette="viridis")
    plt.title("PCA of UCI HAR Dataset")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    if save_path:
        plt.savefig(f"{save_path}/pca.png")
        print("==========================")
        print(f"Visualization saved as '{save_path}/pca.png'")
        print("==========================\n\n")
    plt.clf()
    
    tsne = TSNE(n_components=2, perplexity=30)
    X_tsne = tsne.fit_transform(X_train)

    plt.figure(figsize=(12, 6))
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y_labels, palette="viridis")
    plt.title("t-SNE of UCI HAR Dataset")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    if save_path:
        plt.savefig(f"{save_path}/tsne.png")
        print("==========================")
        print(f"Visualization saved as '{save_path}/tsne.png'")
        print("==========================\n\n")
    plt.clf()