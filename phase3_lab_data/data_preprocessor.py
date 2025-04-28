from utils.loader import preprocess_lab_data

# Load the dataset
df = preprocess_lab_data(base_path="Labeled", window_size=128, step_size=16, save_path="phase3_lab_data/dataset")
# y = df['Task']
# X = df.drop(columns=['Task'])
# # Encode the labels
# le = LabelEncoder()
# y = le.fit_transform(y)
# # Split the data into training and testing sets making sure to keep the same ratio of classes in both sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
# # Create a logistic regression model
# model = LogisticRegression(max_iter=5000)
# # Fit the model to the training data
# model.fit(X_train, y_train)
# # Make predictions on the test data
# y_pred = model.predict(X_test)
# # Print the classification report
# print(classification_report(y_test, y_pred, target_names=le.classes_))