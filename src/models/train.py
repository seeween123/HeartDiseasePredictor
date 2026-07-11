import os
from pathlib import Path

import mlflow
import pandas as pd
import mlflow.xgboost
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        classification_report,
        )
from joblib import dump

DEFAULT_OUT = Path("serving/model/Candidate_Model.json")

THRESHOLD = 0.3

def train_model(
    df,
    target_col,
    best_params,
    model_output: Path | str = DEFAULT_OUT,
):
    """
    Trains an XGBoost model and logs with MLflow.

    Args:
        df (pd.DataFrame): Feature dataset.
                target_col (str): Name of the target column.
        best_params (dict): Best hyperparameters for the model.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y,  random_state=42
    )

    best_model = XGBClassifier(
        **best_params,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss",
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum()
    )

    best_model.fit(X_train, y_train)

    # ---------------------------------------------------------------
    # Final Test Evaluation
    # ---------------------------------------------------------------
    proba = best_model.predict_proba(X_test)[:, 1]
    y_pred = (proba >= THRESHOLD).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, proba)

    print("\nClassification Report")
    print(classification_report(y_test, y_pred))

    # ---------------------------------------------------------------
    # Log Best Model
    # ---------------------------------------------------------------
    # Log final best model to MLflow
    with mlflow.start_run(run_name="XGBoost Candidate Model"):
        mlflow.log_params(best_params)

        mlflow.log_metric("Accuracy", accuracy)
        mlflow.log_metric("Precision", precision)
        mlflow.log_metric("Recall", recall)
        mlflow.log_metric("F1", f1)
        mlflow.log_metric("ROC_AUC", roc_auc)

        mlflow.xgboost.log_model(best_model, "candidate_model")
        mlflow.log_text(classification_report(y_test, y_pred), "classification_report.txt")

        # Optional: save the Optuna study as an artifact
        with open("best_params.txt", "w") as f:
            f.write(str(best_params))

        mlflow.log_artifact("best_params.txt")

        # Delete the local file
        os.remove("best_params.txt")

        # Save to models/
        out = Path(model_output)
        out.parent.mkdir(parents=True, exist_ok=True)
        dump(best_model, out)
        print(f"✅ Best model saved to {out}")

        # Optional raw model
        best_model.save_model("HeartDiseasePredictor.json")
        mlflow.log_artifact(
            "HeartDiseasePredictor.json",
            artifact_path="candidate_model"
        )
        os.remove("HeartDiseasePredictor.json")

        print(f"Model trained. Accuracy: {accuracy:.4f}, Recall: {recall:.4f}")

    return best_model
