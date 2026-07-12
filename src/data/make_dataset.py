from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo


def load_data(output_path: str = "data/raw/heart_disease.csv") -> pd.DataFrame:
    """
    Download the UCI Heart Disease dataset and save it locally.

    Returns
    -------
    pd.DataFrame
    """

    DATASET_ID = 45
    heart = fetch_ucirepo(id=DATASET_ID)

    X = heart.data.features.copy()
    y = heart.data.targets.copy()

    df = pd.concat([X, y], axis=1)

    # Convert multi-class target to binary classification
    if "num" in df.columns:
        df["target"] = (df["num"] > 0).astype(int)
        df.drop(columns=["num"], inplace=True)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)

    print(f"Dataset saved to {output_path}")
    print(df.head())

    return df


if __name__ == "__main__":
    load_data()
