import pandas as pd


def _map_binary_series(s: pd.Series) -> pd.Series:
    """
    Apply deterministic binary encoding to 2-category features.

    This function implements the core binary encoding logic that converts
    categorical features with exactly 2 values into 0/1 integers. The mappings
    are deterministic and must be consistent between training and serving.

    """
    # Get unique values and remove NaN
    vals = list(pd.Series(s.dropna().unique()).astype(str))

    # === GENERIC BINARY MAPPING ===
    # For any other 2-category feature, use stable alphabetical ordering
    if len(vals) == 2:
        # Sort values to ensure consistent mapping across runs
        sorted_vals = sorted(vals)
        mapping = {sorted_vals[0]: 0, sorted_vals[1]: 1}
        return s.astype(str).map(mapping).astype("Int64")

    # === NON-BINARY FEATURES ===
    # Return unchanged - will be handled by one-hot encoding
    return s


def build_features(
    df: pd.DataFrame,
    target_col: str = "target",
) -> pd.DataFrame:

    if target_col not in df:
        raise ValueError(...)

    df = df.copy()

    # categorical = [
    #     "cp",
    #     "restecg",
    #     "slope",
    #     "thal",
    #     "ca",
    # ]

    categorical = [
        col
        for col in df.select_dtypes(include="category").columns
        if df[col].nunique() > 2
    ]

    # One hot encoding for categorical features with > 2 unique values
    df = pd.get_dummies(df, columns=categorical, drop_first=True)

    return df
