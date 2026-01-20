from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from .config import OUTPUT_DIR, TRAINING_CONFIG
from .models import build_model


def bootstrap_confidence(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(TRAINING_CONFIG.random_seed)
    probs = []
    for _ in range(TRAINING_CONFIG.bootstrap_samples):
        indices = rng.integers(0, len(x_train), len(x_train))
        model = build_model()
        model.fit(x_train[indices], y_train[indices])
        probs.append(model.predict_proba(x_eval)[:, 1])
    prob_array = np.vstack(probs)
    mean_prob = prob_array.mean(axis=0)
    std_prob = prob_array.std(axis=0)
    return mean_prob, std_prob


def explain_with_permutation(
    model,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    feature_names: List[str],
) -> pd.DataFrame:
    result = permutation_importance(
        model,
        x_eval,
        y_eval,
        n_repeats=20,
        random_state=TRAINING_CONFIG.random_seed,
    )
    importances = pd.DataFrame(
        {
            "feature": feature_names,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    return importances


def save_feature_importance(importances: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    top = importances.head(10)
    plt.figure(figsize=(8, 6))
    plt.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"], color="steelblue")
    plt.gca().invert_yaxis()
    plt.title("Top Feature Importance (Permutation)")
    plt.xlabel("Importance")
    plt.tight_layout()
    OUTPUT_DIR.joinpath("figures", "feature_importance.png").parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_DIR / "figures" / "feature_importance.png", dpi=150)
    plt.close()


def generate_explanations(
    importances: pd.DataFrame,
    sample_ids: List[str],
    mean_prob: np.ndarray,
    std_prob: np.ndarray,
) -> pd.DataFrame:
    top_features = ", ".join(importances.head(5)["feature"].tolist())
    explanations = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "prediction_confidence": mean_prob,
            "uncertainty": std_prob,
            "top_features": top_features,
        }
    )
    return explanations


def gap1_confidence_and_explain(
    model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    feature_names: List[str],
    sample_ids: List[str],
) -> Dict[str, pd.DataFrame]:
    mean_prob, std_prob = bootstrap_confidence(x_train, y_train, x_eval)
    importances = explain_with_permutation(model, x_eval, y_eval, feature_names)
    save_feature_importance(importances)
    explanations = generate_explanations(importances, sample_ids, mean_prob, std_prob)
    explanations.to_csv(OUTPUT_DIR / "tables" / "explanations.csv", index=False)
    return {
        "mean_prob": pd.Series(mean_prob),
        "std_prob": pd.Series(std_prob),
        "importances": importances,
        "explanations": explanations,
    }
