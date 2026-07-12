import mlflow
import optuna
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    recall_score,
    precision_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
)


def tune_model(df, target_col):
    """
    Tunes an XGBoost model using Optuna.

    Args:
        df (pd.DataFrame): Feature dataset.
        target_col (str): Name of the target column.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

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
    with mlflow.start_run(run_name="XGBoost_Tuning"):
        study = optuna.create_study(direction="maximize")

        study.optimize(objective, n_trials=30)

        print("\nBest Parameters")
        print(study.best_params)

        print("\nBest Recall")
        print(study.best_value)

        return study.best_params
