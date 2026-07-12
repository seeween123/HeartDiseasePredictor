from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


def predict_model(model, df, target_col):
    """
    Evaluates an XGBoost model on test data.

    Args:
        model: Trained model.
        df: Test dataset.
        target_col: Name of the target column.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    preds = model.predict(X_test)
    print("Classification Report:\n", classification_report(y_test, preds))
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))

    return preds
