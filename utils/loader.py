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

# Function: sliding_window
# This function applies a sliding window approach to the given data.
# It calculates the mean and standard deviation for each window of specified size and step size.
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

# Function: preprocess_lab_data
# This function preprocesses the lab data by loading CSV files, applying a sliding window,
# and merging the dataframes. It also handles missing values and renames columns.
# It saves the final merged dataframe to a CSV file if a save path is provided.
def preprocess_lab_data(base_path="Labeled", window_size=128, step_size=64, save_path=None):
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
        if save_path:
            print(f"Applying sliding window to DataFrame {i}...")
        df = sliding_window(df, window_size=window_size, step_size=step_size)

        # add suffix
        if i == 0 or i == 2:
            suffix = '_lower'
        else:
            suffix = '_upper'

        # rename columns
        new_columns = {col: f"{col}{suffix}" for col in df.columns if col != 'Task'}
        df.rename(columns=new_columns, inplace=True)

        dataframes[i] = df

        # Print the shape and head of the dataframe after sliding window
        if save_path:
            print(f"DataFrame {i} shape after sliding window: {dataframes[i].shape}")
            print(f"DataFrame {i} head after sliding window: {dataframes[i].head()}")

    # # Do inner join on all dataframes based on index columns
    # if save_path:
    #     print("Merging dataframes...")
    # merged_df = pd.concat(dataframes, axis=1, join='inner')
    # # Drop the last columns (duplicate 'Task' columns)
    # merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]
    # if save_path:
    #     print(f"Merged DataFrame shape: {merged_df.shape}")
    #     print(f"Merged DataFrame head: {merged_df.head()}")


    # Now merge first two dataframes (index 0 and 1)
    if save_path:
        print("Merging first two dataframes...")
    merged1 = pd.concat([dataframes[0], dataframes[1]], axis=1, join='inner')
    merged1 = merged1.loc[:, ~merged1.columns.duplicated()]

    # Merge second two dataframes (index 2 and 3)
    if save_path:
        print("Merging second two dataframes...")
    merged2 = pd.concat([dataframes[2], dataframes[3]], axis=1, join='inner')
    merged2 = merged2.loc[:, ~merged2.columns.duplicated()]

    # Finally, concatenate merged1 and merged2 row-wise
    if save_path:
        print("Concatenating merged dataframes...")
    merged_df = pd.concat([merged1, merged2], axis=0).reset_index(drop=True)

    if save_path:
        print(f"Final merged DataFrame shape: {merged_df.shape}")
        print(f"Final merged DataFrame head: {merged_df.head()}")

    # Save the merged dataframe to a csv file
    if save_path:
        merged_df.to_csv(f"{save_path}/lab_W{window_size}_S{step_size}.csv", index=False)
        print(f"Merged DataFrame saved to '{save_path}/lab_W{window_size}_S{step_size}.csv'")

    return merged_df
