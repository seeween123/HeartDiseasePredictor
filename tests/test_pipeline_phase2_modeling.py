import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.xgboost
from xgboost import XGBClassifier
import optuna
import os

from sklearn.metrics import (
    recall_score,
    precision_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    classification_report,
)

print("=== Phase 2: Modeling with XGBoost ===")

df = pd.read_csv("data/processed/heart_disease_processed.csv")

# target must be numeric 0/1
if df["target"].dtype == "object":
    df["target"] = df["target"].str.strip().map({"No": 0, "Yes": 1})

assert df["target"].isna().sum() == 0, "target has NaNs"
assert set(df["target"].unique()) <= {0, 1}, "target not 0/1"

X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

THRESHOLD = 0.3


def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 300, 800),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 10),
        "random_state": 42,
        "n_jobs": -1,
        "eval_metric": "logloss",
        "scale_pos_weight": (y_train == 0).sum() / (y_train == 1).sum(),
    }

    # Nested MLflow run for every Optuna trial
    with mlflow.start_run(run_name=f"Trial_{trial.number}", nested=True):
        mlflow.log_params(params)

        model = XGBClassifier(**params)

        model.fit(X_train, y_train)

        proba = model.predict_proba(X_test)[:, 1]
        y_pred = (proba >= THRESHOLD).astype(int)

        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, proba)

        mlflow.log_metric("Recall", recall)
        mlflow.log_metric("Precision", precision)
        mlflow.log_metric("F1", f1)
        mlflow.log_metric("Accuracy", accuracy)
        mlflow.log_metric("ROC_AUC", auc)

        return recall


# -------------------------------------------------------------------
# Parent MLflow Run
# -------------------------------------------------------------------
with mlflow.start_run(run_name="Optuna_XGBoost"):
    study = optuna.create_study(direction="maximize")

    study.optimize(objective, n_trials=30)

    print("\nBest Parameters")
    print(study.best_params)

    print("\nBest Recall")
    print(study.best_value)

    # ---------------------------------------------------------------
    # Train Best Model on Entire Training Data
    # ---------------------------------------------------------------
    best_params = study.best_params

    best_model = XGBClassifier(
        **best_params,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss",
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
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
    with mlflow.start_run(run_name="Candidate Model"):
        mlflow.log_params(best_params)

        mlflow.log_metric("Accuracy", accuracy)
        mlflow.log_metric("Precision", precision)
        mlflow.log_metric("Recall", recall)
        mlflow.log_metric("F1", f1)
        mlflow.log_metric("ROC_AUC", roc_auc)

        mlflow.xgboost.log_model(best_model, "candidate_model")

        # Optional: save the Optuna study as an artifact
        with open("best_params.txt", "w") as f:
            f.write(str(study.best_params))

        mlflow.log_artifact("best_params.txt")

        # Delete the local file
        os.remove("best_params.txt")

        # # No need to Save to models since it is just testing
        # out = Path(model_output)
        # out.parent.mkdir(parents=True, exist_ok=True)
        # dump(best_model, out)
        # print(f"✅ Best model saved to {out}")

print("\nOptimization Complete!")
