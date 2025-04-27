import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import the data from 'Labeled/01_train_lower_labeled.csv' and 'Labeled/01_train_upper_labeled.csv'
lower_labeled = pd.read_csv('Labeled/01_train_lower_labeled.csv')
upper_labeled = pd.read_csv('Labeled/01_train_upper_labeled.csv')

# print the number of rows and columns in the data
print(f"Lower Labeled Data: {lower_labeled.shape[0]} rows, {lower_labeled.shape[1]} columns")
print(f"Upper Labeled Data: {upper_labeled.shape[0]} rows, {upper_labeled.shape[1]} columns")

# drop Var1
lower_labeled = lower_labeled.drop(columns=['Var1'])
upper_labeled = upper_labeled.drop(columns=['Var1'])
# print the first few rows of the data
print(lower_labeled.head())
print(upper_labeled.head())
print(lower_labeled.tail())
print(upper_labeled.tail())

# Replace the missing values in 'Task' with 'Unknown'
lower_labeled['Task'] = lower_labeled['Task'].fillna('Unknown')
upper_labeled['Task'] = upper_labeled['Task'].fillna('Unknown')

# Plot the distribution of the 'Task' column in both datasets on the same figure side by side
def plot_task_distribution(lower_labeled, upper_labeled):
    plt.figure(figsize=(12, 6))

    # Plot for lower_labeled
    plt.subplot(1, 2, 1)
    sns.countplot(data=lower_labeled, x='Task', order=lower_labeled['Task'].value_counts().index)
    plt.title('Lower Labeled Task Distribution')
    plt.xticks(rotation=45)

    # Plot for upper_labeled
    plt.subplot(1, 2, 2)
    sns.countplot(data=upper_labeled, x='Task', order=upper_labeled['Task'].value_counts().index)
    plt.title('Upper Labeled Task Distribution')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig('phase3_lab_data/figures/task_distribution.png')

plot_task_distribution(lower_labeled, upper_labeled)



