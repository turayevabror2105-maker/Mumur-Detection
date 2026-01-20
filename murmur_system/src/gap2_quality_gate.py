from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import AUDIO_CONFIG, OUTPUT_DIR, QUALITY_CONFIG


def estimate_snr(audio: np.ndarray) -> float:
    signal_power = np.mean(audio**2)
    noise_floor = np.mean((audio - np.mean(audio)) ** 2)
    if noise_floor == 0:
        return 100.0
    return 10 * np.log10(signal_power / noise_floor)


def clipping_ratio(audio: np.ndarray) -> float:
    return float(np.mean(np.abs(audio) >= 0.99))


def silence_ratio(audio: np.ndarray) -> float:
    return float(np.mean(np.abs(audio) < 0.02))


def duration_valid(audio: np.ndarray) -> bool:
    expected_len = int(AUDIO_CONFIG.sample_rate * AUDIO_CONFIG.duration_sec)
    return audio.shape[0] == expected_len


def quality_label(score: float) -> str:
    if score >= 0.7:
        return "good"
    if score >= 0.4:
        return "questionable"
    return "bad"


def quality_score(snr: float, clip: float, silence: float, valid: bool) -> float:
    snr_score = np.clip((snr - QUALITY_CONFIG.snr_bad) / (QUALITY_CONFIG.snr_good - QUALITY_CONFIG.snr_bad), 0, 1)
    clip_score = 1.0 - np.clip(clip / QUALITY_CONFIG.clipping_ratio_bad, 0, 1)
    silence_score = 1.0 - np.clip(silence / QUALITY_CONFIG.silence_ratio_bad, 0, 1)
    valid_score = 1.0 if valid else 0.0
    return float(np.mean([snr_score, clip_score, silence_score, valid_score]))


def evaluate_quality(audio: np.ndarray, sample_id: str) -> Dict[str, object]:
    snr = estimate_snr(audio)
    clip = clipping_ratio(audio)
    silence = silence_ratio(audio)
    valid = duration_valid(audio)
    score = quality_score(snr, clip, silence, valid)
    label = quality_label(score)
    reasons: List[str] = []
    if snr < QUALITY_CONFIG.snr_bad:
        reasons.append("low_snr")
    if clip > QUALITY_CONFIG.clipping_ratio_bad:
        reasons.append("clipping")
    if silence > QUALITY_CONFIG.silence_ratio_bad:
        reasons.append("too_much_silence")
    if not valid:
        reasons.append("invalid_duration")

    return {
        "sample_id": sample_id,
        "snr": snr,
        "clipping_ratio": clip,
        "silence_ratio": silence,
        "duration_valid": valid,
        "quality_score": score,
        "quality_label": label,
        "fail_reasons": ",".join(reasons) if reasons else "none",
    }


def save_quality_report(records: List[Dict[str, object]]) -> pd.DataFrame:
    report = pd.DataFrame(records)
    report.to_csv(OUTPUT_DIR / "tables" / "quality_report.csv", index=False)
    return report


def save_quality_vs_error(quality_df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    errors = (y_true != y_pred).astype(int)
    plt.figure(figsize=(6, 4))
    plt.scatter(quality_df["quality_score"], errors, alpha=0.7)
    plt.xlabel("Quality score")
    plt.ylabel("Error (1=wrong)")
    plt.title("Quality vs Error")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figures" / "quality_vs_error.png", dpi=150)
    plt.close()


def gap2_quality_gate(
    audio_list: List[np.ndarray], sample_ids: List[str], y_true: np.ndarray, y_pred: np.ndarray
) -> pd.DataFrame:
    records = [evaluate_quality(audio, sample_id) for audio, sample_id in zip(audio_list, sample_ids)]
    report = save_quality_report(records)
    save_quality_vs_error(report, y_true, y_pred)
    return report
