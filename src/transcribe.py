"""Speech-to-text with Whisper.

Whisper is a strong ASR model, but it tends to "clean up" speech: it drops some
fillers and occasionally repairs grammar. We decode greedily with no language
model prompt so the transcript stays as close to verbatim as possible, and we
complement text features with acoustic features that Whisper cannot smooth over.
"""
from __future__ import annotations

import librosa
import pandas as pd
import torch
from transformers import pipeline

from .cache import cached_features
from .config import SAMPLE_RATE, WHISPER_MODEL, get_device


def load_audio(path) -> "np.ndarray":  # noqa: F821
    """Load audio as mono float32 at 16 kHz (the rate Whisper and WavLM expect)."""
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    return audio


def build_asr(model_name: str = WHISPER_MODEL):
    device = get_device()
    dtype = torch.float16 if device == "cuda" else torch.float32
    return pipeline(
        "automatic-speech-recognition",
        model=model_name,
        dtype=dtype,
        device=device,
        chunk_length_s=30,  # recordings are 45-60 s; Whisper's window is 30 s
        batch_size=4,
    )


def transcribe(items: pd.DataFrame, model_name: str = WHISPER_MODEL) -> pd.DataFrame:
    """Return a DataFrame (index=key) with a ``transcript`` column."""
    asr = None

    def compute(item: pd.Series) -> dict:
        nonlocal asr
        asr = asr or build_asr(model_name)  # lazy: no model load when everything is cached
        out = asr(
            load_audio(item["path"]),
            generate_kwargs={"language": "en", "task": "transcribe", "num_beams": 1},
        )
        return {"transcript": out["text"].strip()}

    return cached_features("transcripts", items, compute, save_every=10)
