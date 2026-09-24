"""
data_preprocessing.py — ChurnOps

Builds a reproducible sklearn preprocessing pipeline:
  - fills nulls in Offer / Internet Type
  - binary-encodes Yes/No and Male/Female columns
  - ordinal-encodes Contract (Month-to-Month < One Year < Two Year)
  - one-hot-encodes Offer, Internet Type, Payment Method
  - StandardScaler on all numeric features

The pipeline is fitted on train data only (no leakage into test).
Both transformed CSVs and the fitted preprocessor.pkl are written to disk.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder, OneHotEncoder, StandardScaler

ROOT_DIR = Path(__file__).resolve().parent.parent

# --------------------------------------------------------------------------- #
# Column groups                                                                #
# --------------------------------------------------------------------------- #
TARGET = "Churn Value"

# Filled before the pipeline runs so encoders see consistent categories
FILL_NULLS = {
    "Offer":         "No Offer",
    "Internet Type": "None",
}

# Yes/No binary columns (encoded as 1/0)
YES_NO_COLS = [
    "Referred a Friend", "Phone Service", "Multiple Lines",
    "Internet Service", "Online Security", "Online Backup",
    "Device Protection Plan", "Premium Tech Support",
    "Streaming TV", "Streaming Movies", "Streaming Music",
    "Unlimited Data", "Paperless Billing", "Married",
]

# Gender: Male -> 1, Female -> 0
GENDER_COLS = ["Gender"]

# Ordinal: month-to-month is shortest commitment, two-year is longest
CONTRACT_CATS = [["Month-to-Month", "One Year", "Two Year"]]

# One-hot encoded (drop first to avoid dummy trap)
OHE_COLS = ["Offer", "Internet Type", "Payment Method"]

# All numeric columns (determined at preprocessing time from what's left)
# We define them explicitly so the API gets an identical list
NUMERIC_COLS = [
    "Age", "Number of Dependents", "Number of Referrals",
    "Tenure in Months", "Avg Monthly Long Distance Charges",
    "Avg Monthly GB Download", "Monthly Charge", "Total Charges",
    "Total Refunds", "Total Extra Data Charges",
    "Total Long Distance Charges", "Total Revenue",
    "Satisfaction Score",
]


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def _fill_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """Fill known nullable columns before transformation."""
    df = df.copy()
    for col, fill_val in FILL_NULLS.items():
        if col in df.columns:
            df[col] = df[col].fillna(fill_val)
    return df


def _binary_encode(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Map Yes->1 / No->0 for listed columns."""
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = df[col].map({"Yes": 1, "No": 0}).astype(float)
    return df


def _gender_encode(df: pd.DataFrame) -> pd.DataFrame:
    """Map Male->1 / Female->0."""
    df = df.copy()
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0}).astype(float)
    return df


def build_preprocessor() -> ColumnTransformer:
    """
    Build a ColumnTransformer for the churn feature set.
    All fitted state lives inside this object — no external state.
    """
    ordinal = OrdinalEncoder(categories=CONTRACT_CATS, handle_unknown="use_encoded_value", unknown_value=-1)
    ohe     = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
    scaler  = StandardScaler()

    return ColumnTransformer(
        transformers=[
            ("num",      scaler,  NUMERIC_COLS),
            ("contract", ordinal, ["Contract"]),
            ("ohe",      ohe,     OHE_COLS),
        ],
        remainder="passthrough",   # binary-encoded columns pass through unchanged
    )


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply null-fills and manual encodings that happen before the sklearn pipeline."""
    df = _fill_nulls(df)
    df = _binary_encode(df, YES_NO_COLS)
    df = _gender_encode(df)
    return df


def main() -> None:
    train_path = ROOT_DIR / "data" / "raw" / "train.csv"
    test_path  = ROOT_DIR / "data" / "raw" / "test.csv"

    train = pd.read_csv(train_path)
    test  = pd.read_csv(test_path)

    # Separate target
    y_train = train[TARGET]
    y_test  = test[TARGET]
    X_train = train.drop(columns=[TARGET])
    X_test  = test.drop(columns=[TARGET])

    # Pre-encoding steps (before the sklearn pipeline)
    X_train = prepare_features(X_train)
    X_test  = prepare_features(X_test)

    # Build and fit the preprocessor on training data ONLY
    preprocessor = build_preprocessor()
    X_train_enc = preprocessor.fit_transform(X_train)
    X_test_enc  = preprocessor.transform(X_test)

    # Recover column names for the transformed output
    num_names      = NUMERIC_COLS
    contract_names = ["Contract_encoded"]
    ohe_names      = list(preprocessor.named_transformers_["ohe"].get_feature_names_out(OHE_COLS))
    passthrough_cols = [
        c for c in X_train.columns
        if c not in NUMERIC_COLS + ["Contract"] + OHE_COLS
    ]
    all_cols = num_names + contract_names + ohe_names + passthrough_cols

    train_processed = pd.DataFrame(X_train_enc, columns=all_cols)
    test_processed  = pd.DataFrame(X_test_enc,  columns=all_cols)

    # Re-attach target
    train_processed[TARGET] = y_train.values
    test_processed[TARGET]  = y_test.values

    out_dir = ROOT_DIR / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_processed.to_csv(out_dir / "train_processed.csv", index=False)
    test_processed.to_csv(out_dir  / "test_processed.csv",  index=False)
    print(f"Processed shapes  train={train_processed.shape}  test={test_processed.shape}")

    # Save the fitted preprocessor so the API can apply it at inference time
    preprocessor_path = ROOT_DIR / "preprocessor.pkl"
    with preprocessor_path.open("wb") as f:
        pickle.dump(preprocessor, f)
    print(f"Preprocessor saved to {preprocessor_path}")

    print("Data preprocessing completed successfully.")


if __name__ == "__main__":
    main()