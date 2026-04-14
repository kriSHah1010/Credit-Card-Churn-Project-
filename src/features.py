import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Age groups
    if "Customer_Age" in df.columns:
        df["age_group"] = pd.cut(
            df["Customer_Age"],
            bins=[0, 30, 45, 60, 100],
            labels=["18-30", "31-45", "46-60", "60+"],
            include_lowest=True
        )

    # Tenure groups
    if "Months_on_book" in df.columns:
        df["tenure_group"] = pd.cut(
            df["Months_on_book"],
            bins=[0, 12, 24, 36, 60],
            labels=["0-12", "13-24", "25-36", "36+"],
            include_lowest=True
        )

    # Utilization groups
    if "Avg_Utilization_Ratio" in df.columns:
        df["utilization_group"] = pd.cut(
            df["Avg_Utilization_Ratio"],
            bins=[-0.01, 0.25, 0.5, 0.75, 1.0],
            labels=["Low", "Medium", "High", "Very High"]
        )

    # Activity proxy
    if {"Total_Trans_Amt", "Total_Trans_Ct"}.issubset(df.columns):
        df["avg_transaction_value"] = df["Total_Trans_Amt"] / df["Total_Trans_Ct"].replace(0, pd.NA)

    return df