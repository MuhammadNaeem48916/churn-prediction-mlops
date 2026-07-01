"""
Customer Churn Prediction — Training Script
Full MLOps stack with CI/CD-ready metric output.
"""
import os
import sys
import json
import pickle
import subprocess
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, classification_report,
    ConfusionMatrixDisplay
)
import mlflow
import mlflow.sklearn


def get_git_info():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode().strip()
    except Exception:
        commit, branch = "unknown", "unknown"
    return commit, branch


def load_data(path):
    df = pd.read_csv(path)
    X = df.drop("churn", axis=1)
    y = df["churn"]
    return X, y


def get_model(model_type, params):
    if model_type == "logistic":
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(
            C=params.get("C", 1.0),
            class_weight="balanced",   # critical for imbalanced churn data
            max_iter=params.get("max_iter", 1000),
            random_state=42
        )
    elif model_type == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", 6),
            class_weight="balanced",
            random_state=42
        )
    elif model_type == "gradient_boosting":
        from sklearn.ensemble import GradientBoostingClassifier
        clf = GradientBoostingClassifier(
            n_estimators=params.get("n_estimators", 100),
            learning_rate=params.get("learning_rate", 0.1),
            max_depth=params.get("max_depth", 3),
            random_state=42
        )
    else:
        raise ValueError(f"Unknown model: {model_type}")

    return Pipeline([("scaler", StandardScaler()), ("classifier", clf)])


def save_confusion_matrix(model, X_test, y_test, path):
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test,
        display_labels=["Stayed", "Churned"],
        colorbar=False, ax=ax
    )
    ax.set_title("Confusion Matrix — Churn Prediction")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def train(model_type="logistic", params=None):
    if params is None:
        params = {}

    DATA_PATH = "data/raw/churn.csv"
    X, y = load_data(DATA_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    git_commit, git_branch = get_git_info()
    mlflow.set_experiment("churn-prediction")

    with mlflow.start_run(run_name=f"{model_type}_run"):
        mlflow.set_tag("developer", "naeem")
        mlflow.set_tag("model_type", model_type)
        mlflow.set_tag("git_commit", git_commit)
        mlflow.set_tag("git_branch", git_branch)
        mlflow.set_tag("dataset", "telecom_churn_v1")

        all_params = {"model_type": model_type, **params}
        mlflow.log_params(all_params)

        pipeline = get_model(model_type, params)
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        preds_prob = pipeline.predict_proba(X_test)[:, 1]

        accuracy  = accuracy_score(y_test, preds)
        f1        = f1_score(y_test, preds)
        precision = precision_score(y_test, preds)
        recall    = recall_score(y_test, preds)
        roc_auc   = roc_auc_score(y_test, preds_prob)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc")

        mlflow.log_metrics({
            "accuracy":  round(accuracy, 4),
            "f1_score":  round(f1, 4),
            "precision": round(precision, 4),
            "recall":    round(recall, 4),
            "roc_auc":   round(roc_auc, 4),
            "cv_roc_auc_mean": round(cv_scores.mean(), 4),
        })

        print("=" * 50)
        print(f"  Model      : {model_type}")
        print(f"  Accuracy   : {accuracy:.4f}")
        print(f"  Recall     : {recall:.4f}  ← catch churners")
        print(f"  Precision  : {precision:.4f}")
        print(f"  F1 Score   : {f1:.4f}")
        print(f"  ROC-AUC    : {roc_auc:.4f}")
        print(f"  CV ROC-AUC : {cv_scores.mean():.4f}")
        print("=" * 50)
        print(classification_report(y_test, preds, target_names=["Stayed", "Churned"]))

        os.makedirs("outputs", exist_ok=True)
        cm_path = f"outputs/confusion_matrix_{model_type}.png"
        save_confusion_matrix(pipeline, X_test, y_test, cm_path)
        mlflow.log_artifact(cm_path)

        metrics_data = {
            "model_type": model_type,
            "accuracy":   round(accuracy, 4),
            "f1_score":   round(f1, 4),
            "precision":  round(precision, 4),
            "recall":     round(recall, 4),
            "roc_auc":    round(roc_auc, 4),
            "cv_roc_auc_mean": round(cv_scores.mean(), 4)
        }
        os.makedirs("metrics", exist_ok=True)
        with open("metrics/scores.json", "w") as f:
            json.dump(metrics_data, f, indent=2)
        mlflow.log_artifact("metrics/scores.json")

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            registered_model_name=f"churn-{model_type}"
        )

        os.makedirs("models", exist_ok=True)
        with open("models/model.pkl", "wb") as f:
            pickle.dump(pipeline, f)

        print(f"\nMLflow run logged ✓  Git: {git_commit[:8]} on {git_branch}")

    return metrics_data


if __name__ == "__main__":
    model_type = sys.argv[1] if len(sys.argv) > 1 else "logistic"
    params = {}
    for arg in sys.argv[2:]:
        if "=" in arg:
            k, v = arg.split("=", 1)
            try: v = int(v)
            except ValueError:
                try: v = float(v)
                except ValueError: pass
            params[k] = v
    train(model_type, params)