"""Training and evaluation utilities for ML Assignment 2.

The module keeps one deterministic train/test split so the Streamlit app and
README report the same results.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
TEST_SIZE = 0.20
TARGET_COLUMN = "target"


def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Return the UCI Wisconsin Diagnostic Breast Cancer dataset."""
    bunch = load_breast_cancer(as_frame=True)
    features = bunch.data.copy()
    target = bunch.target.astype(int).copy()
    target.name = TARGET_COLUMN
    return features, target


def make_split() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    features, target = load_dataset()
    return train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )


def build_model_catalog() -> OrderedDict[str, object]:
    """Create the five named models plus SVM to satisfy the stated count of six."""
    return OrderedDict(
        {
            "Logistic Regression": Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=2_000,
                            random_state=RANDOM_STATE,
                            solver="liblinear",
                        ),
                    ),
                ]
            ),
            "Decision Tree": DecisionTreeClassifier(
                max_depth=5,
                min_samples_leaf=3,
                random_state=RANDOM_STATE,
            ),
            "kNN": Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("model", KNeighborsClassifier(n_neighbors=7)),
                ]
            ),
            "Naive Bayes": Pipeline(
                [("scale", StandardScaler()), ("model", GaussianNB())]
            ),
            "Random Forest (Ensemble)": RandomForestClassifier(
                n_estimators=400,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
            "Support Vector Machine (Additional)": Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "model",
                        CalibratedClassifierCV(
                            estimator=SVC(
                                C=2.0,
                                kernel="rbf",
                                class_weight="balanced",
                                random_state=RANDOM_STATE,
                            ),
                            method="sigmoid",
                            cv=5,
                        ),
                    ),
                ]
            ),
        }
    )


def calculate_metrics(
    estimator: object, features: pd.DataFrame, target: pd.Series
) -> dict[str, float]:
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        predictions = estimator.predict(features)
        probabilities = estimator.predict_proba(features)[:, 1]
    try:
        auc = roc_auc_score(target, probabilities)
    except ValueError:
        auc = np.nan
    return {
        "Accuracy": accuracy_score(target, predictions),
        "AUC": auc,
        "Precision": precision_score(target, predictions, zero_division=0),
        "Recall": recall_score(target, predictions, zero_division=0),
        "F1": f1_score(target, predictions, zero_division=0),
        "MCC": matthews_corrcoef(target, predictions),
    }


def train_all_models() -> tuple[
    OrderedDict[str, object], pd.DataFrame, pd.DataFrame, pd.Series, pd.Series
]:
    x_train, x_test, y_train, y_test = make_split()
    models = build_model_catalog()
    for estimator in models.values():
        estimator.fit(x_train, y_train)
    return models, x_train, x_test, y_train, y_test


def evaluate_all_models(
    models: OrderedDict[str, object], features: pd.DataFrame, target: pd.Series
) -> pd.DataFrame:
    rows = {
        name: calculate_metrics(estimator, features, target)
        for name, estimator in models.items()
    }
    result = pd.DataFrame.from_dict(rows, orient="index")
    result.index.name = "ML Model Name"
    return result


if __name__ == "__main__":
    trained_models, _, test_features, _, test_target = train_all_models()
    final_metrics = evaluate_all_models(
        trained_models, test_features, test_target
    )
    print(final_metrics.round(4).to_string())
