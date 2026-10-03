"""Locate the competition files and load train/test metadata.

The dataset can live in different places depending on where the code runs
(local kagglehub cache, a Kaggle notebook's /kaggle/input, or a manual copy),
so we search a list of candidate roots for ``train.csv``.

Important: train/ and test/ reuse the same filenames for *different* recordings
(e.g. both contain an ``audio_0.wav``). Audio is therefore resolved inside each
split's own folder, and every file is identified by ``key = "<split>/<filename>"``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import COMPETITION, PROJECT_ROOT


def _candidate_roots() -> list[Path]:
    roots = []
    if env := os.environ.get("GRAMMAR_DATA_DIR"):
        roots.append(Path(env))
    roots += [
        PROJECT_ROOT / "data",
        Path("/kaggle/input"),
        Path.home() / ".cache/kagglehub/competitions" / COMPETITION,
    ]
    return [r for r in roots if r.exists()]


def find_data_root(download: bool = True) -> Path:
    """Return the directory containing train.csv, downloading via kagglehub if needed."""
    for root in _candidate_roots():
        hits = sorted(root.rglob("train.csv"))
        if hits:
            return hits[0].parent
    if download:
        import kagglehub

        path = Path(kagglehub.competition_download(COMPETITION))
        return find_data_root(download=False) if not (path / "train.csv").exists() else path
    raise FileNotFoundError(
        "train.csv not found. Set GRAMMAR_DATA_DIR or configure Kaggle credentials."
    )


@dataclass
class Dataset:
    root: Path
    train: pd.DataFrame  # columns: filename, label, key, path
    test: pd.DataFrame  # columns: filename, key, path
    sample_submission: pd.DataFrame | None


def load_dataset(root: Path | None = None) -> Dataset:
    root = Path(root) if root else find_data_root()
    search_root = root if any(root.rglob("*.wav")) else root.parent

    def attach_paths(df: pd.DataFrame, split: str) -> pd.DataFrame:
        split_dirs = [d for d in search_root.rglob(split) if d.is_dir()]
        if len(split_dirs) != 1:
            raise FileNotFoundError(f"expected exactly one '{split}/' folder under {search_root}, found {split_dirs}")
        wav_index = {p.name: p for p in split_dirs[0].rglob("*.wav")}
        df = df.copy()
        df["filename"] = df["filename"].astype(str)
        df["filename"] = df["filename"].where(df["filename"].str.endswith(".wav"), df["filename"] + ".wav")
        df["key"] = f"{split}/" + df["filename"]
        df["path"] = df["filename"].map(wav_index)
        missing = df["path"].isna().sum()
        if missing:
            raise FileNotFoundError(f"{missing} {split} audio files listed in CSV were not found in {split_dirs[0]}")
        return df

    train = attach_paths(pd.read_csv(root / "train.csv"), "train")
    test = attach_paths(pd.read_csv(root / "test.csv")[["filename"]], "test")  # test labels are placeholders
    sub_path = root / "sample_submission.csv"
    sample = pd.read_csv(sub_path) if sub_path.exists() else None
    return Dataset(root=root, train=train, test=test, sample_submission=sample)
