import numpy as np
from utils.loader import load_dataset_UCI
from utils.plotting import visualize_class_distribution, visualize_dataset_2D

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

# # Pretty print the dataset information from 'UCI HAR Dataset/features_info.txt'
# def pretty_print_features_info():
#     print("==========================")
#     print("📊 Features Information:")
#     print("==========================")
#     with open("UCI HAR Dataset/features_info.txt", "r") as f:
#         features_info = f.read()
#     print(features_info)
#     print("\n\n")

label_map = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING"
}
data = load_dataset_UCI(base_path="UCI HAR Dataset", minimal=True)
pretty_print_dataset_info(data)
visualize_class_distribution(data['y_train'], data['y_test'], label_map, save_path="phase2_uci_minimal/figures")
visualize_dataset_2D(data['X_train'], data['y_train'], label_map, save_path="phase2_uci_minimal/figures")
# pretty_print_features_info()