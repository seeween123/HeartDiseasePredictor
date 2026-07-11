import pandas as pd


def preprocess_data(
    df: pd.DataFrame,
    target_col: str = "target",
) -> pd.DataFrame:
    """
    Data cleaning for the UCI Heart Disease dataset.
    """

    df = df.copy()

    # --------------------------------------------------
    # Clean column names
    # --------------------------------------------------

    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    # --------------------------------------------------
    # Remove duplicates
    # --------------------------------------------------

    df = df.drop_duplicates()

    # --------------------------------------------------
    # Missing values
    # --------------------------------------------------

    numeric_cols = df.select_dtypes(include="number").columns

    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    # --------------------------------------------------
    # Convert integer categorical columns
    # --------------------------------------------------

    categorical_cols = [
        "sex",
        "cp",
        "fbs",
        "restecg",
        "exang",
        "slope",
        "ca",
        "thal",
    ]

    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")

    return df