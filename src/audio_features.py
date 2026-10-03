"""Acoustic features: fluency/pause statistics and self-supervised WavLM embeddings.

Grammar is mostly a property of the words, but grammatical control shows up in
the audio too: speakers who struggle with structure pause mid-clause, restart
and speak more slowly. These cues survive even when Whisper tidies up the text.
"""
from __future__ import annotations

import librosa
import numpy as np
import pandas as pd
import torch
from transformers import AutoFeatureExtractor, WavLMModel

from .cache import cached_features
from .config import AUDIO_EMBED_MODEL, SAMPLE_RATE, get_device
from .transcribe import load_audio

PAUSE_MIN_S = 0.3  # silences shorter than this are normal articulation gaps


def prosody_features(audio: np.ndarray, sr: int = SAMPLE_RATE) -> dict:
    duration = len(audio) / sr
    # Energy-based voice activity detection: intervals more than 30 dB below the peak count as silence.
    voiced = librosa.effects.split(audio, top_db=30, frame_length=1024, hop_length=256)
    voiced_s = float(sum(e - s for s, e in voiced)) / sr
    gaps = np.array([(voiced[i + 1][0] - voiced[i][1]) / sr for i in range(len(voiced) - 1)])
    pauses = gaps[gaps >= PAUSE_MIN_S] if gaps.size else np.array([])

    f0, voiced_flag, _ = librosa.pyin(audio, fmin=65, fmax=400, sr=sr, frame_length=1024, hop_length=320)
    f0 = f0[voiced_flag] if voiced_flag is not None and voiced_flag.any() else np.array([np.nan])
    rms = librosa.feature.rms(y=audio, frame_length=1024, hop_length=256)[0]
    # Spectral flatness ~0 for tonal speech, ~1 for white noise: flags recordings swamped by noise.
    flatness = librosa.feature.spectral_flatness(y=audio)[0]

    return {
        "duration_s": duration,
        "voiced_ratio": voiced_s / max(duration, 1e-6),
        "pauses_per_min": len(pauses) / max(duration / 60, 1e-6),
        "pause_mean_s": float(pauses.mean()) if pauses.size else 0.0,
        "pause_max_s": float(pauses.max()) if pauses.size else 0.0,
        "pause_total_ratio": float(pauses.sum()) / max(duration, 1e-6),
        "speech_run_mean_s": voiced_s / max(len(voiced), 1),
        "f0_mean": float(np.nanmean(f0)),
        "f0_std": float(np.nanstd(f0)),
        "rms_cv": float(rms.std() / (rms.mean() + 1e-8)),
        "spectral_flatness": float(np.median(flatness)),
    }


def prosody_table(items: pd.DataFrame) -> pd.DataFrame:
    return cached_features("prosody", items, lambda it: prosody_features(load_audio(it["path"])), save_every=50)


class WavLMEmbedder:
    """Pools every hidden layer of WavLM over time -> array (n_layers, hidden) or (n_layers, 2*hidden) with std."""

    def __init__(self, model_name: str = AUDIO_EMBED_MODEL, with_std: bool = False):
        self.with_std = with_std
        self.device = get_device()
        self.fe = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = WavLMModel.from_pretrained(model_name).to(self.device).eval()

    @torch.inference_mode()
    def __call__(self, audio: np.ndarray) -> np.ndarray:
        inputs = self.fe(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt").to(self.device)
        hidden = self.model(**inputs, output_hidden_states=True).hidden_states
        # Mean = what is said on average; std = how much it varies over time (hesitations, rhythm changes).
        pooled = [torch.cat([h.mean(dim=1), h.std(dim=1)], dim=-1) if self.with_std else h.mean(dim=1) for h in hidden]
        return torch.stack([p.squeeze(0) for p in pooled]).float().cpu().numpy()


def wavlm_embeddings(items: pd.DataFrame, model_name: str = AUDIO_EMBED_MODEL, name: str = "wavlm",
                     with_std: bool = False) -> np.ndarray:
    """Per-layer pooled WavLM embeddings, shape (n, n_layers, hidden) - or 2*hidden with mean+std pooling."""
    embedder = None

    def compute(item: pd.Series) -> dict:
        nonlocal embedder
        embedder = embedder or WavLMEmbedder(model_name, with_std)
        layers = embedder(load_audio(item["path"]))
        return {"n_layers": layers.shape[0], "emb": layers.ravel().astype(np.float32).tolist()}

    table = cached_features(name, items, compute, save_every=50)
    n_layers = int(table["n_layers"].iloc[0])
    return np.vstack(table["emb"].to_numpy()).reshape(len(table), n_layers, -1)
