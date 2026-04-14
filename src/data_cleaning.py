from pathlib import Path
import pandas as pd
from src.utils import make_dirs, load_data, save_data

RAW_FILENAME = "Credit_Card_Churn.csv"


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Standardize column names
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]

    # Common target mapping for this dataset
    if "Attrition_Flag" in df.columns:
        df["churned"] = df["Attrition_Flag"].map(
            {"Attrited Customer": 1, "Existing Customer": 0}
        )
        df = df.drop(columns=["Attrition_Flag"])

    # Drop ID-like column if present
    if "CLIENTNUM" in df.columns:
        df = df.drop(columns=["CLIENTNUM"])

    # Remove duplicates
    df = df.drop_duplicates()

    # Replace common placeholders with NaN
    df = df.replace(["Unknown", "NA", "N/A", ""], pd.NA)

    # Basic type cleanup
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype("string")

    return df


def main():
    make_dirs()
    df = load_data(RAW_FILENAME)
    cleaned = clean_data(df)
    save_data(cleaned, "credit_card_churn_cleaned.csv")
    print("Saved cleaned data to data/processed/credit_card_churn_cleaned.csv")
    print(cleaned.head())


if __name__ == "__main__":
    main()