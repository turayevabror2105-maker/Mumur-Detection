from __future__ import annotations

from typing import Dict, List, Tuple

import librosa
import numpy as np

from .config import AUDIO_CONFIG


def load_audio(path: str) -> Tuple[np.ndarray, int]:
    audio, sr = librosa.load(path, sr=AUDIO_CONFIG.sample_rate, mono=True)
    if audio.size == 0:
        raise ValueError(f"Empty audio file: {path}")
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val
    target_len = int(AUDIO_CONFIG.sample_rate * AUDIO_CONFIG.duration_sec)
    if audio.shape[0] < target_len:
        pad_width = target_len - audio.shape[0]
        audio = np.pad(audio, (0, pad_width), mode="constant")
    else:
        audio = audio[:target_len]
    return audio, AUDIO_CONFIG.sample_rate


def _stats(values: np.ndarray, prefix: str) -> Dict[str, float]:
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
    }


def extract_features(audio: np.ndarray, sr: int) -> Tuple[np.ndarray, List[str]]:
    features: Dict[str, float] = {}

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=AUDIO_CONFIG.n_fft,
        hop_length=AUDIO_CONFIG.hop_length,
        n_mels=AUDIO_CONFIG.n_mels,
    )
    log_mel = librosa.power_to_db(mel + 1e-10)
    features.update(_stats(log_mel, "logmel"))

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=AUDIO_CONFIG.n_mfcc,
        n_fft=AUDIO_CONFIG.n_fft,
        hop_length=AUDIO_CONFIG.hop_length,
    )
    for i in range(mfcc.shape[0]):
        stats = _stats(mfcc[i], f"mfcc_{i+1}")
        features.update(stats)

    centroid = librosa.feature.spectral_centroid(
        y=audio, sr=sr, n_fft=AUDIO_CONFIG.n_fft, hop_length=AUDIO_CONFIG.hop_length
    )
    features.update(_stats(centroid, "centroid"))

    bandwidth = librosa.feature.spectral_bandwidth(
        y=audio, sr=sr, n_fft=AUDIO_CONFIG.n_fft, hop_length=AUDIO_CONFIG.hop_length
    )
    features.update(_stats(bandwidth, "bandwidth"))

    rolloff = librosa.feature.spectral_rolloff(
        y=audio, sr=sr, n_fft=AUDIO_CONFIG.n_fft, hop_length=AUDIO_CONFIG.hop_length
    )
    features.update(_stats(rolloff, "rolloff"))

    rms = librosa.feature.rms(y=audio, frame_length=AUDIO_CONFIG.n_fft, hop_length=AUDIO_CONFIG.hop_length)
    features.update(_stats(rms, "rms"))

    zcr = librosa.feature.zero_crossing_rate(y=audio, frame_length=AUDIO_CONFIG.n_fft, hop_length=AUDIO_CONFIG.hop_length)
    features.update(_stats(zcr, "zcr"))

    names = list(features.keys())
    values = np.array([features[name] for name in names], dtype=float)
    return values, names
