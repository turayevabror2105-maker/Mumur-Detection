from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from .config import AUDIO_CONFIG, OUTPUT_DIR
from .data_io import get_severity_column


def risk_proxy(audio: np.ndarray, sr: int) -> Dict[str, float | str]:
    spectrum = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(audio.size, 1 / sr)
    total_energy = np.sum(spectrum**2) + 1e-8
    band_mask = (freqs >= 200) & (freqs <= 600)
    band_energy = np.sum((spectrum[band_mask]) ** 2)
    murmur_energy_ratio = float(band_energy / total_energy)

    low_band = (freqs >= 20) & (freqs < 200)
    high_band = (freqs >= 600) & (freqs <= 1000)
    dominance = float((np.sum(spectrum[high_band] ** 2) + 1e-8) / (np.sum(spectrum[low_band] ** 2) + 1e-8))

    frame_len = int(0.5 * sr)
    hop = int(0.25 * sr)
    rms_vals = []
    for start in range(0, len(audio) - frame_len, hop):
        frame = audio[start : start + frame_len]
        rms_vals.append(np.sqrt(np.mean(frame**2)))
    rms_vals = np.array(rms_vals) if rms_vals else np.array([0.0])
    consistency = float(1.0 - np.clip(np.std(rms_vals), 0, 1))

    risk_score = 0.5 * murmur_energy_ratio + 0.3 * dominance + 0.2 * (1 - consistency)
    if risk_score < 0.2:
        level = "low"
    elif risk_score < 0.5:
        level = "medium"
    else:
        level = "high"

    return {
        "risk_proxy_score": float(risk_score),
        "risk_level": level,
        "murmur_energy_ratio": murmur_energy_ratio,
        "frequency_dominance": dominance,
        "consistency": consistency,
    }


def risk_from_labels(df: pd.DataFrame, severity_col: str) -> pd.DataFrame:
    risk_levels = df[severity_col].astype(str).str.lower()
    mapping = {"low": "low", "mild": "low", "1": "low", "medium": "medium", "moderate": "medium", "2": "medium", "high": "high", "severe": "high", "3": "high"}
    mapped = risk_levels.map(lambda x: mapping.get(x, "medium"))
    return pd.DataFrame({"risk_level": mapped})


def save_risk_distribution(risk_df: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    counts = risk_df["risk_level"].value_counts().reindex(["low", "medium", "high"], fill_value=0)
    plt.figure(figsize=(5, 4))
    plt.bar(counts.index, counts.values, color="teal")
    plt.title("Risk Distribution")
    plt.xlabel("Risk level")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figures" / "risk_distribution.png", dpi=150)
    plt.close()


def write_limitations(message: str) -> None:
    (OUTPUT_DIR / "logs" / "limitations.txt").write_text(message)


def gap4_risk_urgency(audio_list: List[np.ndarray], metadata: pd.DataFrame) -> pd.DataFrame:
    severity_col = get_severity_column(metadata)
    if severity_col:
        risk_df = risk_from_labels(metadata, severity_col)
        message = (
            "Risk levels were inferred from provided severity labels. "
            "These labels may be subjective and require clinical confirmation."
        )
    else:
        proxy_records = [risk_proxy(audio, AUDIO_CONFIG.sample_rate) for audio in audio_list]
        risk_df = pd.DataFrame(proxy_records)
        message = (
            "Risk levels are a proxy derived from audio energy patterns only. "
            "This does NOT represent clinical severity and should be used solely for triage guidance."
        )

    risk_df.to_csv(OUTPUT_DIR / "tables" / "risk_results.csv", index=False)
    save_risk_distribution(risk_df)
    write_limitations(message)
    return risk_df
