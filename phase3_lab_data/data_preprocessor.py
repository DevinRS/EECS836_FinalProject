import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
import os

# Define a sliding window function
def sliding_window(data, window_size=128, step_size=64):
    # get the column names
    column_names = data.columns.tolist()
    column_names.remove('Task')
    # create an empty pd.DataFrame to store the results with featuure names [column_names+mean and std]
    result = pd.DataFrame(columns=[f"{col}_mean" for col in column_names] + [f"{col}_std" for col in column_names])
    # add the task column as empty
    result['Task'] = ''

    # take the first 128 rows of the data
    for i in range(0, len(data) - window_size + 1, step_size):
        # get the current window
        window = data.iloc[i:i + window_size]
        # get the value for task by taking the last value of the task column
        task = window['Task'].values[-1]
        window = window.drop(columns=['Task'])
        # calculate the mean and std for each column
        mean_values = window.mean()
        std_values = window.std()
        # make a new row
        new_row = pd.Series({'Var2_mean': mean_values['Var2'], 'Var2_std': std_values['Var2'],
                             'Var3_mean': mean_values['Var3'], 'Var3_std': std_values['Var3'],
                             'Var4_mean': mean_values['Var4'], 'Var4_std': std_values['Var4'],
                             'Var5_mean': mean_values['Var5'], 'Var5_std': std_values['Var5'],
                             'Var6_mean': mean_values['Var6'], 'Var6_std': std_values['Var6'],
                             'Var7_mean': mean_values['Var7'], 'Var7_std': std_values['Var7'],
                             'Task': task})
        # append the new row to the result
        result.loc[len(result)] = new_row

    return result

def load_dataset(base_path="Labeled", window_size=128, step_size=64, save_path=None):
    # Load all files ending with .csv in the base_path directory
    dataframes = []
    files = os.listdir(base_path)
    # sort ascending
    files.sort()
    for filename in files:
        if filename.endswith(".csv"):
            print(f"Loading {filename}...")
            file_path = os.path.join(base_path, filename)
            df = pd.read_csv(file_path)
            dataframes.append(df)

    # [THIS IS SPECIFIC TO THIS EXAMPLE - SUBJECT 1]
    original_last_index = dataframes[1].shape[0]
    dataframes[1] = dataframes[1].drop(index=range(102))
    dataframes[1] = dataframes[1].drop(index=range(original_last_index-9, original_last_index))

    # Drop the 'Var1' and 'time' columns from all dataframes
    for df in dataframes:
        if 'Var1' in df.columns:
            df.drop(columns=['Var1'], inplace=True)
        if 'time' in df.columns:
            df.drop(columns=['time'], inplace=True)
    # Replace missing values in 'Task' with 'Unknown'
    for df in dataframes:
        if 'Task' in df.columns:
            df['Task'] = df['Task'].fillna('Unknown')

    # Do sliding window on all dataframes
    for i, df in enumerate(dataframes):
        print(f"Applying sliding window to DataFrame {i}...")
        df = sliding_window(df, window_size=window_size, step_size=step_size)

        # add suffix
        if i == 0:
            suffix = '_lower'
        else:
            suffix = '_upper'

        # rename columns
        new_columns = {col: f"{col}{suffix}" for col in df.columns if col != 'Task'}
        df.rename(columns=new_columns, inplace=True)

        dataframes[i] = df

        print(f"DataFrame {i} shape after sliding window: {dataframes[i].shape}")
        print(f"DataFrame {i} head after sliding window: {dataframes[i].head()}")

    # Do inner join on all dataframes based on index columns
    print("Merging dataframes...")
    merged_df = pd.concat(dataframes, axis=1, join='inner')
    # Drop the last columns (duplicate 'Task' columns)
    merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]
    print(f"Merged DataFrame shape: {merged_df.shape}")
    print(f"Merged DataFrame head: {merged_df.head()}")

    # Save the merged dataframe to a csv file
    if save_path:
        merged_df.to_csv(save_path, index=False)
        print(f"Merged DataFrame saved to {save_path}")

    return merged_df

# Load the dataset
df = load_dataset(window_size=128, step_size=64, save_path='phase3_lab_data/dataset/S1_WS128_SS64.csv')
y = df['Task']
X = df.drop(columns=['Task'])
# Encode the labels
le = LabelEncoder()
y = le.fit_transform(y)
# Split the data into training and testing sets making sure to keep the same ratio of classes in both sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
# Create a logistic regression model
model = LogisticRegression(max_iter=5000)
# Fit the model to the training data
model.fit(X_train, y_train)
# Make predictions on the test data
y_pred = model.predict(X_test)
# Print the classification report
print(classification_report(y_test, y_pred, target_names=le.classes_))