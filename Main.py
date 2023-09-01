from numpy import cov
from scipy.stats import pearsonr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Daumenregel: Je höher, desto schlechter
# Zustand: :) = 0, :/ = 1, :( = 2, skull = 3
# Essen: ++ = 0, 0 = 2, -- = 4, skull = 6
# Sex/Journal/Sport: no = 0, yes = 1, twice = 2 etc.

data = pd.read_csv(open("Data_Raw.csv"), index_col=0)
data = data.drop("Datum", axis=1)

def add_shifted_columns(df, column_name):
    # Create new column names for shifted values
    shifted_columns = [f'{column_name}_shifted_{i}' for i in range(1, 4)]

    # Add shifted columns to the DataFrame
    for i, col in enumerate(shifted_columns):
        df[col] = df[column_name].shift(i+1)

    # Add a column for the sum of the last three days
    df[f'{column_name}_shifted_1+2'] = df[f'{column_name}_shifted_1'] + df[f'{column_name}_shifted_2']
    df[f'{column_name}_shifted_2+3'] = df[f'{column_name}_shifted_2'] + df[f'{column_name}_shifted_3']
    df[f'{column_name}_shifted_1+2+3'] = df[f'{column_name}_shifted_1'] + df[f'{column_name}_shifted_2'] + df[f'{column_name}_shifted_3']

    return df

def shift_all_columns(df, except_column, shift_values=(1, 2, 3)):
    shifted_df = df.copy()

    for col in df.columns:
        if not col in except_column:
            for i in range(1, 4):
                shifted_col = f"{col}_shifted_{i}"
                shifted_df[shifted_col] = df[col].shift(i)

            shifted_df[f"{col}_shifted_1+2"] = shifted_df[f"{col}_shifted_1"] + shifted_df[f"{col}_shifted_2"]
            shifted_df[f"{col}_shifted_2+3"] = shifted_df[f"{col}_shifted_2"] + shifted_df[f"{col}_shifted_3"]
            shifted_df[f"{col}_shifted_1+2+3"] = shifted_df[f"{col}_shifted_1"] + shifted_df[f"{col}_shifted_2"] + shifted_df[f"{col}_shifted_3"]

    return shifted_df
    
def get_correlated_pairs(df, threshold):
    correlations = df.corr()  # Calculate pairwise correlations
    pairs = []

    # Iterate over the upper triangular portion of the correlation matrix
    for i in range(len(correlations.columns)):
        for j in range(i + 1, len(correlations.columns)):
            col1 = correlations.columns[i]
            col2 = correlations.columns[j]
            correlation = correlations.iloc[i, j]

            if not("shifted" in col1 and "shifted" in col2) and not(col1.startswith(col2) or col2.startswith(col1)) and abs(correlation) >= threshold:
                pairs.append((col1, col2, correlation))

    # Sort the pairs in descending order based on correlation value
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)

    return pairs

def print_correlated_pairs(correlated_pairs):
    # Print the correlated pairs
    for pair in correlated_pairs:
        col1, col2, correlation = pair
        print(f"Columns: {col1}, {col2}, Correlation: {correlation}")

#from https://medium.com/@sebastiannorena/finding-correlation-between-many-variables-multidimensional-dataset-with-python-5deb3f39ffb3
def visualize(data):
    corr = data.corr()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)
    fig.colorbar(cax)
    ticks = np.arange(0,len(data.columns),1)
    ax.set_xticks(ticks)
    plt.xticks(rotation=90)
    ax.set_yticks(ticks)
    ax.set_xticklabels(data.columns)
    ax.set_yticklabels(data.columns)
    plt.show()

modified_data = add_shifted_columns(data, "Schritte")

# zoom in on one output, shifting it in comparison to all other values
#output_columns = "Schubflanke, Schmerz, Zustand, Mieps, Verstopftheit, Energie"
#modified_data = shift_all_columns(data, output_columns)

# Save the modified dataset to a new CSV file
modified_data.to_csv('debug.csv', index=False)

correlated_pairs = get_correlated_pairs(modified_data, 0.3)

print_correlated_pairs(correlated_pairs)

# output graph
visualize(modified_data)