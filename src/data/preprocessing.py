import pandas as pd

EXPECTED_COLUMNS = [
    "age",
    "sex",
    "trestbps",
    "chol",
    "fbs",
    "thalach",
    "exang",
    "oldpeak",
    "ca_1.0",
    "ca_2.0",
    "ca_3.0",
    "slope_2",
    "slope_3",
    "thal_6.0",
    "thal_7.0",
    "restecg_1",
    "restecg_2",
    "cp_2",
    "cp_3",
    "cp_4",
]

def preprocess_data(
    df: pd.DataFrame,
    target_col: str = "target",
) -> pd.DataFrame:
    """
    Data cleaning for the UCI Heart Disease dataset.
    """

    df = df.copy()
    df["sex"] = df["sex"].astype(int)
    df["fbs"] = df["fbs"].astype(int)
    df["exang"] = df["exang"].astype(int)

    # Categorical columns with > 2 unique values
    multi_cat_cols = [
        'ca', 'slope', 'thal', 'restecg', 'cp'
    ]

    # One-hot encode
    df = pd.get_dummies(df, columns=multi_cat_cols, drop_first=True)

    df = df.reindex(columns=EXPECTED_COLUMNS, fill_value=0)

    return df
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