"""Compute and cache every feature block for all train + test files: `python -m src.extract_all`."""
import time

import pandas as pd

from . import audio_features as af, llm_grader, text_features as tf, transcribe
from .config import AUDIO_EMBED_MODEL_LARGE
from .data import load_dataset


def timed(name, fn) -> None:
    t = time.time()
    fn()
    print(f"[done] {name} in {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    ds = load_dataset()
    items = pd.concat([ds.train, ds.test], ignore_index=True)
    timed("prosody", lambda: af.prosody_table(items))
    tr = transcribe.transcribe(items)["transcript"]
    print("[done] transcripts", flush=True)
    timed("wavlm", lambda: af.wavlm_embeddings(items))
    timed("wavlm_large", lambda: af.wavlm_embeddings(items, AUDIO_EMBED_MODEL_LARGE, "wavlm_large", with_std=True))
    timed("linguistic", lambda: tf.linguistic_table(items, tr))
    timed("text_emb", lambda: tf.text_embeddings(items, tr))
    timed("gec", lambda: tf.gec_features(items, tr))
    try:  # optional: needs GEMINI_API_KEY (env or .env); the notebook runs without LLM features otherwise
        timed("llm_grades", lambda: llm_grader.llm_grades(items, tr))
    except RuntimeError as err:
        print(f"[skip] llm_grades: {err}", flush=True)
