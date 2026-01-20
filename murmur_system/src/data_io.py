from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from .config import AUDIO_DIR, DATA_DIR, TRAINING_CONFIG


LABEL_COL = "label"
PATH_COL = "filepath"
PATIENT_COL = "patient_id"
SEVERITY_COL = "severity"
DIAGNOSIS_COL = "diagnosis"
GRADE_COL = "grade"


def _resolve_audio_path(path_str: str) -> str:
    path = Path(path_str)
    if path.is_absolute():
        return str(path)
    candidate = AUDIO_DIR / path
    if candidate.exists():
        return str(candidate)
    return str((DATA_DIR / path).resolve())


def load_metadata() -> pd.DataFrame:
    labels_csv = DATA_DIR / "labels.csv"
    train_csv = DATA_DIR / "train.csv"
    test_csv = DATA_DIR / "test.csv"

    if labels_csv.exists():
        df = pd.read_csv(labels_csv)
        df[PATH_COL] = df[PATH_COL].apply(_resolve_audio_path)
        df["split"] = "auto"
        return df

    if train_csv.exists() and test_csv.exists():
        train_df = pd.read_csv(train_csv)
        test_df = pd.read_csv(test_csv)
        train_df["split"] = "train"
        test_df["split"] = "test"
        df = pd.concat([train_df, test_df], ignore_index=True)
        df[PATH_COL] = df[PATH_COL].apply(_resolve_audio_path)
        return df

    raise FileNotFoundError(
        "Expected data/labels.csv or data/train.csv + data/test.csv."
    )


def split_metadata(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if "split" in df.columns and set(df["split"].unique()) != {"auto"}:
        train_df = df[df["split"] == "train"].copy()
        test_df = df[df["split"] == "test"].copy()
        return train_df, test_df

    if PATIENT_COL in df.columns:
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=TRAINING_CONFIG.test_size, random_state=TRAINING_CONFIG.random_seed
        )
        train_idx, test_idx = next(splitter.split(df, groups=df[PATIENT_COL]))
        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()
        return train_df, test_df

    train_df, test_df = train_test_split(
        df,
        test_size=TRAINING_CONFIG.test_size,
        random_state=TRAINING_CONFIG.random_seed,
        stratify=df[LABEL_COL] if LABEL_COL in df.columns else None,
    )
    return train_df.copy(), test_df.copy()


def get_severity_column(df: pd.DataFrame) -> str | None:
    for col in [SEVERITY_COL, GRADE_COL, DIAGNOSIS_COL]:
        if col in df.columns:
            return col
    return None
