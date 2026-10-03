"""Central configuration: paths, model names, and runtime device selection."""
from __future__ import annotations

import os
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = Path(os.environ.get("GRAMMAR_CACHE_DIR", PROJECT_ROOT / "cache"))
SUBMISSION_DIR = PROJECT_ROOT / "submissions"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSION_DIR.mkdir(exist_ok=True)

COMPETITION = "shl-hiring-assessment-2026"

# Pretrained models (all external models are permitted by the competition rules).
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "openai/whisper-large-v3-turbo")
GEC_MODEL = os.environ.get("GEC_MODEL", "vennify/t5-base-grammar-correction")
TEXT_EMBED_MODEL = os.environ.get("TEXT_EMBED_MODEL", "sentence-transformers/all-mpnet-base-v2")
AUDIO_EMBED_MODEL = os.environ.get("AUDIO_EMBED_MODEL", "microsoft/wavlm-base-plus")
AUDIO_EMBED_MODEL_LARGE = os.environ.get("AUDIO_EMBED_MODEL_LARGE", "microsoft/wavlm-large")

SAMPLE_RATE = 16_000
SEED = 42


def get_device() -> str:
    """Pick the fastest available accelerator: CUDA (Kaggle GPU) > Apple MPS > CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
