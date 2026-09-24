"""
data_collection.py — ChurnOps

Reads the five Churn_Data Excel files, merges them on Customer ID,
drops identifiers / leakage columns / geographic columns, clips
any negative Number of Dependents values to 0, then writes an
80/20 stratified train/test split to data/raw/.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "Churn_Data"


# --------------------------------------------------------------------------- #
# Columns to drop before any modelling                                        #
# --------------------------------------------------------------------------- #
DROP_COLS = [
    # Identifiers / surrogate keys
    "Customer ID", "Count", "Location ID", "Service ID", "Status ID",
    # Direct leakage — derived from / perfectly correlated with churn outcome
    "Churn Label",   # string duplicate of target
    "Churn Score",   # propensity score computed post-hoc from churn status
    "CLTV",          # Customer Lifetime Value computed using known churn dates
    # Geographic — high cardinality or low individual-level predictive signal
    "City", "Zip Code", "Latitude", "Longitude", "Population",
    # Redundant binary flags (covered by Age / Number of Dependents)
    "Under 30", "Senior Citizen", "Dependents",
    # Temporal constant — single value in this export
    "Quarter",
]


def load_and_merge() -> pd.DataFrame:
    """Load all five Excel source files and merge into one DataFrame."""
    demo = pd.read_excel(DATA_DIR / "Demographics.xlsx")
    loc  = pd.read_excel(DATA_DIR / "Location.xlsx")
    pop  = pd.read_excel(DATA_DIR / "Population.xlsx")
    svc  = pd.read_excel(DATA_DIR / "Services.xlsx")
    stat = pd.read_excel(DATA_DIR / "Status.xlsx")

    # Keep only the columns we need from each file to avoid duplicate Count cols
    loc_cols  = ["Customer ID", "City", "Zip Code", "Latitude", "Longitude"]
    pop_cols  = ["Zip Code", "Population"]
    stat_cols = ["Customer ID", "Satisfaction Score",
                 "Churn Label", "Churn Value", "Churn Score", "CLTV"]

    df = (
        demo
        .merge(loc[loc_cols],  on="Customer ID", how="left")
        .merge(
            pop[pop_cols],
            on="Zip Code",
            how="left",
        )
        .merge(svc.drop(columns=["Count", "Service ID"], errors="ignore"),
               on="Customer ID", how="left")
        .merge(stat[stat_cols], on="Customer ID", how="left")
    )
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop leakage / administrative columns and fix data-quality issues."""
    # Drop the columns we identified in the migration plan
    existing_drops = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=existing_drops)

    # Clip negative Number of Dependents to 0 (data entry error in source)
    if "Number of Dependents" in df.columns:
        df["Number of Dependents"] = df["Number of Dependents"].clip(lower=0)

    return df


def main() -> None:
    print("Loading and merging Churn_Data Excel files...")
    df = load_and_merge()
    print(f"  Merged shape: {df.shape}")

    df = clean(df)
    print(f"  After dropping leakage/ID/geo columns: {df.shape}")
    print(f"  Target distribution:\n{df['Churn Value'].value_counts().to_string()}")

    # Stratified split so both classes appear proportionally in each split
    train_df, test_df = train_test_split(
        df,
        test_size=0.20,
        random_state=42,
        stratify=df["Churn Value"],
    )

    out_dir = ROOT_DIR / "data" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(out_dir / "train.csv", index=False)
    test_df.to_csv(out_dir / "test.csv",  index=False)

    print(f"  Train: {len(train_df)} rows  |  Test: {len(test_df)} rows")
    print("Data collection completed successfully.")


if __name__ == "__main__":
    main()