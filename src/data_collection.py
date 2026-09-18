import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# Locate the dataset in the repository root
candidate_paths = [
    Path("water_potability.csv"),
    Path("water_potability (1).csv")
]

dataset_path = next((path for path in candidate_paths if path.exists()), None)
if dataset_path is None:
    raise FileNotFoundError("No water_potability dataset file was found in the repository root.")

# Read the dataset
data = pd.read_csv(dataset_path)

# Split the dataset into training and testing sets
train_data, test_data = train_test_split(
    data,
    test_size=0.20,
    random_state=42
)

# Create the output directory
data_path = Path("data") / "raw"
data_path.mkdir(parents=True, exist_ok=True)

# Save the train and test datasets
train_data.to_csv(data_path / "train.csv", index=False)
test_data.to_csv(data_path / "test.csv", index=False)

print("Train and Test datasets saved successfully!")