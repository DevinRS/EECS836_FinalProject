import numpy as np
from utils.loader import preprocess_lab_data
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

df = preprocess_lab_data(base_path="Labeled", window_size=128, step_size=64, save_path=None)
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

data = {
    "X_train": X_train.to_numpy(),
    "y_train": y_train,
    "subject_train": None,
    "X_test": X_test.to_numpy(),
    "y_test": y_test,
    "subject_test": None,
}

label_map = {i: label for i, label in enumerate(le.classes_)}

pretty_print_dataset_info(data)
visualize_class_distribution(data['y_train'], data['y_test'], label_map, save_path="phase3_lab_data/figures")
visualize_dataset_2D(data['X_train'], data['y_train'], label_map, save_path="phase3_lab_data/figures")