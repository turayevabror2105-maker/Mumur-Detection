from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
AUDIO_DIR = DATA_DIR / "audio"
OUTPUT_DIR = ROOT_DIR / "outputs"


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 4000
    duration_sec: float = 10.0
    n_mels: int = 40
    n_mfcc: int = 13
    hop_length: int = 256
    n_fft: int = 1024


@dataclass(frozen=True)
class TrainingConfig:
    random_seed: int = 42
    classifier: str = "logistic"  # or "random_forest"
    test_size: float = 0.2
    bootstrap_samples: int = 20
    high_confidence_threshold: float = 0.8


@dataclass(frozen=True)
class QualityConfig:
    snr_good: float = 20.0
    snr_bad: float = 5.0
    clipping_ratio_bad: float = 0.01
    silence_ratio_bad: float = 0.4


AUDIO_CONFIG = AudioConfig()
TRAINING_CONFIG = TrainingConfig()
QUALITY_CONFIG = QualityConfig()


def ensure_output_dirs() -> None:
    for subdir in [
        OUTPUT_DIR / "tables",
        OUTPUT_DIR / "figures",
        OUTPUT_DIR / "models",
        OUTPUT_DIR / "logs",
    ]:
        subdir.mkdir(parents=True, exist_ok=True)
