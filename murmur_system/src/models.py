from __future__ import annotations

from typing import Tuple

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import OUTPUT_DIR, TRAINING_CONFIG


def build_model() -> Pipeline:
    if TRAINING_CONFIG.classifier == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=200,
            random_state=TRAINING_CONFIG.random_seed,
            class_weight="balanced",
        )
    else:
        clf = LogisticRegression(
            max_iter=2000,
            solver="lbfgs",
            random_state=TRAINING_CONFIG.random_seed,
        )
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def train_model(x_train: np.ndarray, y_train: np.ndarray) -> Pipeline:
    model = build_model()
    model.fit(x_train, y_train)
    return model


def save_model(model: Pipeline, feature_names: list[str]) -> None:
    output = {
        "model": model,
        "feature_names": feature_names,
    }
    joblib.dump(output, OUTPUT_DIR / "models" / "baseline_model.joblib")


def load_model() -> Tuple[Pipeline, list[str]]:
    output = joblib.load(OUTPUT_DIR / "models" / "baseline_model.joblib")
    return output["model"], output["feature_names"]
