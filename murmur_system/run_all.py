from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import OUTPUT_DIR, TRAINING_CONFIG, ensure_output_dirs
from src.data_io import LABEL_COL, PATH_COL, load_metadata, split_metadata
from src.dsp_features import extract_features, load_audio
from src.eval_protocol import compute_metrics
from src.gap1_conf_explain import gap1_confidence_and_explain
from src.gap2_quality_gate import gap2_quality_gate
from src.gap3_calibration_triage import gap3_calibration_and_triage
from src.gap4_risk_urgency import gap4_risk_urgency
from src.models import save_model, train_model
from src.reporting import write_summary


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def build_feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str], list[np.ndarray]]:
    features = []
    audio_list = []
    feature_names: list[str] = []
    for _, row in df.iterrows():
        audio, sr = load_audio(row[PATH_COL])
        vector, names = extract_features(audio, sr)
        if not feature_names:
            feature_names = names
        features.append(vector)
        audio_list.append(audio)
    return np.vstack(features), feature_names, audio_list


def main() -> None:
    ensure_output_dirs()

    try:
        metadata = load_metadata()
    except FileNotFoundError as exc:
        logging.error(str(exc))
        return

    if LABEL_COL not in metadata.columns:
        logging.error("labels.csv must include a 'label' column.")
        return

    train_df, test_df = split_metadata(metadata)

    logging.info("Extracting features...")
    x_train, feature_names, train_audio = build_feature_matrix(train_df)
    x_test, _, test_audio = build_feature_matrix(test_df)

    y_train = train_df[LABEL_COL].to_numpy()
    y_test = test_df[LABEL_COL].to_numpy()

    logging.info("Training baseline model...")
    model = train_model(x_train, y_train)
    y_prob = model.predict_proba(x_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = compute_metrics(y_test, y_pred, y_prob)
    logging.info("Baseline metrics: %s", metrics)

    save_model(model, feature_names)

    logging.info("Running Gap 1: confidence + explanations...")
    gap1 = gap1_confidence_and_explain(
        model,
        x_train,
        y_train,
        x_test,
        y_test,
        feature_names,
        sample_ids=test_df[PATH_COL].tolist(),
    )

    logging.info("Running Gap 2: quality gate...")
    quality_report = gap2_quality_gate(test_audio, test_df[PATH_COL].tolist(), y_test, y_pred)

    logging.info("Running Gap 3: calibration + triage...")
    gap3 = gap3_calibration_and_triage(
        x_train,
        y_train,
        x_test,
        y_test,
        quality_report["quality_label"].to_numpy(),
    )

    logging.info("Running Gap 4: risk / urgency...")
    risk_results = gap4_risk_urgency(test_audio, test_df)

    summary_sections = {
        "Baseline performance": (
            f"Accuracy: {metrics.get('accuracy', 0):.3f}\n"
            f"F1: {metrics.get('f1', 0):.3f}\n"
            f"ROC-AUC: {metrics.get('roc_auc', 0):.3f}"
        ),
        "Gap 1 findings": (
            "Bootstrap confidence and permutation importance were computed.\n"
            "See outputs/figures/feature_importance.png and outputs/tables/explanations.csv."
        ),
        "Gap 2 findings": (
            "Audio quality metrics were computed with SNR, clipping, silence ratio, and duration checks.\n"
            "See outputs/tables/quality_report.csv and outputs/figures/quality_vs_error.png."
        ),
        "Gap 3 findings": (
            f"ECE after calibration: {gap3['ece']:.3f}.\n"
            "A triage policy was applied using the confidence threshold.\n"
            "See outputs/figures/reliability_diagram.png and outputs/tables/triage_results.csv."
        ),
        "Gap 4 findings": (
            "Risk levels were generated.\n"
            "See outputs/tables/risk_results.csv and outputs/figures/risk_distribution.png."
        ),
        "Limitations and ethical notes": (
            "This system is a baseline and should not be used as a clinical diagnostic.\n"
            "Predictions require clinician oversight and should be validated on local data."
        ),
    }

    write_summary(OUTPUT_DIR / "logs" / "summary.txt", summary_sections)

    logging.info("Run complete. Outputs saved to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
