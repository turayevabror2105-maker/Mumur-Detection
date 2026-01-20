from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

from .config import OUTPUT_DIR, TRAINING_CONFIG
from .eval_protocol import expected_calibration_error
from .models import build_model


def calibrate_model(x_train: np.ndarray, y_train: np.ndarray) -> CalibratedClassifierCV:
    base_model = build_model()
    calibrator = CalibratedClassifierCV(
        base_estimator=base_model,
        method="isotonic",
        cv=3,
    )
    calibrator.fit(x_train, y_train)
    return calibrator


def reliability_diagram(y_true: np.ndarray, y_prob: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
    plt.figure(figsize=(6, 5))
    plt.plot(prob_pred, prob_true, marker="o", label="Model")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.title("Reliability Diagram")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figures" / "reliability_diagram.png", dpi=150)
    plt.close()


def triage_decision(prob: np.ndarray, quality_label: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    confidence = np.maximum(prob, 1 - prob)
    auto_accept = confidence >= TRAINING_CONFIG.high_confidence_threshold
    low_reliability = quality_label == "bad"
    auto_accept = np.where(low_reliability, False, auto_accept)
    return auto_accept, confidence


def triage_report(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    quality_label: np.ndarray,
) -> pd.DataFrame:
    auto_accept, confidence = triage_decision(y_prob, quality_label)
    accepted = auto_accept
    rejected = ~auto_accept
    accepted_accuracy = float(np.mean((y_true[accepted] == (y_prob[accepted] >= 0.5)).astype(float))) if np.any(accepted) else 0.0
    referral_rate = float(np.mean(rejected))
    false_negatives_before = int(np.sum((y_true == 1) & (y_prob < 0.5)))
    false_negatives_after = int(np.sum((y_true == 1) & (y_prob < 0.5) & accepted))
    report = pd.DataFrame(
        {
            "accepted_accuracy": [accepted_accuracy],
            "referral_rate": [referral_rate],
            "false_negatives_before": [false_negatives_before],
            "false_negatives_after": [false_negatives_after],
        }
    )
    report.to_csv(OUTPUT_DIR / "tables" / "triage_results.csv", index=False)
    return report


def gap3_calibration_and_triage(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    quality_label: np.ndarray,
) -> Dict[str, object]:
    calibrator = calibrate_model(x_train, y_train)
    y_prob = calibrator.predict_proba(x_eval)[:, 1]
    ece = expected_calibration_error(y_eval, y_prob)
    reliability_diagram(y_eval, y_prob)
    report = triage_report(y_eval, y_prob, quality_label)
    return {
        "calibrator": calibrator,
        "ece": ece,
        "triage_report": report,
        "calibrated_prob": y_prob,
    }
