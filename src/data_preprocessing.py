import os
import pandas as pd
import numpy as np

# Read the training and testing datasets
train_data = pd.read_csv("./data/raw/train.csv")
test_data = pd.read_csv("./data/raw/test.csv")


def fill_missing_with_median(df):
    df = df.copy()
    for column in df.columns:
        if df[column].isnull().any():
            median_value = df[column].median()
            df[column] = df[column].fillna(median_value)
    return df


# Fill missing values
train_processed_data = fill_missing_with_median(train_data)
test_processed_data = fill_missing_with_median(test_data)

# Create the processed data directory
data_path = os.path.join("data", "processed")
os.makedirs(data_path, exist_ok=True)

# Save the processed datasets
train_processed_data.to_csv(
    os.path.join(data_path, "train_processed.csv"),
    index=False
)

test_processed_data.to_csv(
    os.path.join(data_path, "test_processed.csv"),
    index=False
)

print("Data preprocessing completed successfully!")