"""LLM rubric grader: Gemini scores each transcript against the competition rubric.

The LLM reads the transcript the way a human rater would and returns structured
sub-scores (overall grammar, accuracy, complexity, error count) plus a one-line
rationale, so every score can be traced back to specific errors. Transcripts are
sent in small batches with retry/backoff, and results are cached per file.

The API key is read from the GEMINI_API_KEY environment variable or a local
``.env`` file (never stored in code or the notebook).
"""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm.auto import tqdm

from .cache import chunked
from .config import CACHE_DIR, PROJECT_ROOT

LLM_MODEL = os.environ.get("LLM_MODEL", "gemini-3.5-flash-lite")
BATCH_SIZE = 25  # few, larger requests: friendly to free-tier quotas
WORKERS = 1  # sequential requests avoid bursting the rate limit

RUBRIC = """\
1 - Struggles with proper sentence structure and syntax; relies on memorised patterns.
2 - Limited understanding of sentence structure; uses simple structures with consistent basic mistakes; sentences may be left incomplete.
3 - Decent grasp of sentence structure but errors in grammatical structure, OR decent grammar but errors in sentence structure/syntax.
4 - Strong understanding of sentence structure and grammar; only minor errors, mostly self-corrected.
5 - High grammatical accuracy and control of complex structures; consistent, with occasional self-correction."""

PROMPT = """You are an expert examiner of SPOKEN English grammar. Each item below is an automatic speech-recognition
transcript of a 45-60 second spoken answer. Score ONLY grammar and sentence structure using this rubric:

{rubric}

Guidance:
- Ignore punctuation, capitalisation and spelling: they come from the ASR system, not the speaker.
- Fillers (um, uh), false starts and self-corrections are normal in speech; repaired errors count in the speaker's favour.
- Judge tense, agreement, articles, prepositions, word order, sentence completeness and the range of structures used.
- Very short or garbled transcripts usually mean the speaker could not produce structured English.

For EACH item return:
- grammar_score: overall rubric score, a number from 1.0 to 5.0 (decimals allowed)
- accuracy: grammatical accuracy, 1.0 to 5.0
- complexity: range/complexity of sentence structures, 1.0 to 5.0
- error_count: number of distinct grammatical errors in the transcript
- rationale: one short sentence naming the main errors or strengths

Items:
{items}"""

SCHEMA = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "id": {"type": "STRING"},
            "grammar_score": {"type": "NUMBER"},
            "accuracy": {"type": "NUMBER"},
            "complexity": {"type": "NUMBER"},
            "error_count": {"type": "INTEGER"},
            "rationale": {"type": "STRING"},
        },
        "required": ["id", "grammar_score", "accuracy", "complexity", "error_count", "rationale"],
    },
}


def _load_api_key() -> str:
    if key := os.environ.get("GEMINI_API_KEY"):
        return key
    env = PROJECT_ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("GEMINI_API_KEY="):
                return line.split("=", 1)[1].strip()
    raise RuntimeError("GEMINI_API_KEY not set (environment variable or .env file)")


def _grade_batch(client, batch: list[tuple[str, str]], retries: int = 4) -> list[dict]:
    from google.genai import types

    # Short local ids keep the prompt compact and avoid leaking filenames into the judgement.
    items = "\n\n".join(f"[id={i}]\n{text.strip() or '(empty transcript)'}" for i, (_, text) in enumerate(batch))
    config = types.GenerateContentConfig(
        temperature=0.0, response_mime_type="application/json", response_schema=SCHEMA,
    )
    for attempt in range(retries):
        try:
            resp = client.models.generate_content(
                model=LLM_MODEL, contents=PROMPT.format(rubric=RUBRIC, items=items), config=config
            )
            rows = {str(r["id"]): r for r in json.loads(resp.text)}
            out = []
            for i, (key, _) in enumerate(batch):
                r = rows[str(i)]  # KeyError -> retry: the model must return every item
                out.append({"key": key, **{k: r[k] for k in SCHEMA["items"]["required"] if k != "id"}})
            return out
        except Exception as err:  # rate limits (429), transient 5xx, malformed JSON
            if attempt == retries - 1:
                raise
            time.sleep(65 if "429" in str(err) else 5 * (attempt + 1))  # 429: wait out the per-minute quota
    return []


def llm_grades(items: pd.DataFrame, transcripts: pd.Series) -> pd.DataFrame:
    """Return a DataFrame indexed by key with llm_* columns (cached in cache/llm_grades.parquet)."""
    path = CACHE_DIR / "llm_grades.parquet"
    done = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    todo = [(k, transcripts[k]) for k in items["key"] if k not in done.index]
    if todo:
        from google import genai

        client = genai.Client(api_key=_load_api_key())
        batches = list(chunked(todo, BATCH_SIZE))
        with ThreadPoolExecutor(WORKERS) as pool:
            futures = [pool.submit(_grade_batch, client, b) for b in batches]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="llm_grades"):
                new = pd.DataFrame(fut.result()).set_index("key")
                done = pd.concat([done, new])
                done.to_parquet(path)  # save after every batch: safe to interrupt
    table = done.loc[items["key"]]
    return table.rename(columns={c: f"llm_{c}" for c in table.columns})
