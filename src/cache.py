"""Resumable per-file feature cache.

Feature extraction (Whisper, WavLM, ...) is the slow part of the pipeline. Each
extractor writes one row per audio file to ``cache/<name>.parquet`` and saves
every ``save_every`` files, so an interrupted run resumes where it stopped and
re-running the notebook costs nothing.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from .config import CACHE_DIR


def cached_features(
    name: str,
    items: pd.DataFrame,
    compute: Callable[[pd.Series], dict],
    save_every: int = 25,
) -> pd.DataFrame:
    """Return a DataFrame indexed by ``key`` (``"<split>/<filename>"``) with one row of features per item.

    ``items`` must have ``key`` and ``path`` columns. ``compute`` receives one
    row and returns a dict of features (scalars, strings or lists/arrays).
    """
    path: Path = CACHE_DIR / f"{name}.parquet"
    done = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    todo = items[~items["key"].isin(done.index)]

    rows: list[dict] = []

    def flush() -> None:
        nonlocal done, rows
        if rows:
            done = pd.concat([done, pd.DataFrame(rows).set_index("key")])
            done.to_parquet(path)
            rows = []

    for i, (_, item) in enumerate(tqdm(todo.iterrows(), total=len(todo), desc=name, disable=todo.empty)):
        rows.append({"key": item["key"], **compute(item)})
        if (i + 1) % save_every == 0:
            flush()
    flush()
    return done.loc[items["key"]]


def chunked(seq: list, size: int) -> Iterable[list]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]
