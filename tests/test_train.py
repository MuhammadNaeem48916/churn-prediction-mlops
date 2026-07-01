"""
Unit tests for the churn prediction pipeline.
These run automatically in CI on every push.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
import pytest
from train import load_data, get_model


def test_data_file_exists():
    """Dataset must exist before any training can happen."""
    assert os.path.exists("data/raw/churn.csv"), \
        "churn.csv not found — run create_dataset.py first"


def test_data_has_required_columns():
    """Dataset must have the columns the model expects."""
    df = pd.read_csv("data/raw/churn.csv")
    required = [
        "tenure_months", "monthly_charges", "total_charges",
        "support_calls", "contract_month_to_month",
        "has_tech_support", "senior_citizen",
        "paperless_billing", "churn"
    ]
    for col in required:
        assert col in df.columns, f"Missing required column: {col}"


def test_no_missing_values():
    """Data quality check — no nulls allowed."""
    df = pd.read_csv("data/raw/churn.csv")
    assert df.isnull().sum().sum() == 0, "Dataset contains missing values"


def test_churn_is_binary():
    """Target column must only contain 0 and 1."""
    df = pd.read_csv("data/raw/churn.csv")
    assert set(df["churn"].unique()) <= {0, 1}, \
        "churn column must be binary (0 or 1)"


def test_load_data_returns_correct_shapes():
    """load_data() must split features and target correctly."""
    X, y = load_data("data/raw/churn.csv")
    assert len(X) == len(y)
    assert "churn" not in X.columns
    assert X.shape[1] == 8  # 8 feature columns expected


def test_get_model_logistic():
    """Pipeline must build correctly for logistic regression."""
    model = get_model("logistic", {})
    assert model is not None
    assert "scaler" in model.named_steps
    assert "classifier" in model.named_steps


def test_get_model_random_forest():
    """Pipeline must build correctly for random forest."""
    model = get_model("random_forest", {"n_estimators": 50})
    assert model is not None


def test_get_model_invalid_type_raises_error():
    """Unknown model type must raise a clear error, not fail silently."""
    with pytest.raises(ValueError):
        get_model("not_a_real_model", {})


def test_model_can_fit_and_predict():
    """End-to-end smoke test — model must train and predict without error."""
    X, y = load_data("data/raw/churn.csv")
    model = get_model("logistic", {})
    model.fit(X, y)
    preds = model.predict(X)
    assert len(preds) == len(y)
    assert set(preds) <= {0, 1}