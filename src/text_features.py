"""Grammar-oriented features computed from the Whisper transcript.

Three groups, from most to least interpretable:

1. Linguistic statistics (spaCy): sentence length, syntactic depth, subordinate
   clause use, fragments, agreement errors, disfluencies, lexical diversity.
   These map directly onto the rubric ("simple structures", "incomplete
   sentences", "complex structures").
2. Grammatical-error-correction (GEC) edit rate: a seq2seq model rewrites every
   sentence into correct English; the fraction of words it has to change is a
   direct, explainable measure of grammatical error density.
3. Sentence-embedding vector of the full transcript: a dense, less
   interpretable representation that captures overall language quality.
"""
from __future__ import annotations

import difflib
import re

import numpy as np
import pandas as pd
import spacy
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .cache import cached_features, chunked
from .config import GEC_MODEL, TEXT_EMBED_MODEL, get_device

FILLERS = {"um", "uh", "er", "erm", "ah", "hmm", "mm", "uhm", "eh"}
SUBORDINATE_DEPS = {"mark", "advcl", "ccomp", "xcomp", "relcl", "acl", "csubj"}

_nlp = None


def nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def _words(text: str) -> list[str]:
    """Lower-cased word tokens without punctuation (so case/punctuation edits are ignored)."""
    return re.findall(r"[a-z0-9']+", text.lower())


def _depth(token) -> int:
    depth = 0
    while token.head.i != token.i:  # spaCy roots are their own head (compare indices, not objects)
        token = token.head
        depth += 1
    return depth


def _agreement_errors(sent) -> int:
    """Count simple subject-verb number mismatches, e.g. 'he go', 'they goes'."""
    errors = 0
    for tok in sent:
        if tok.dep_ != "nsubj" or tok.head.pos_ not in {"VERB", "AUX"}:
            continue
        # The finite (present-tense) form carries agreement: an auxiliary ('he don't like') or the verb itself.
        finite = [t for t in [*tok.head.children, tok.head] if t.dep_ in {"aux", tok.head.dep_} and t.tag_ in {"VBZ", "VBP"}]
        if not finite:
            continue
        verb_tag = finite[0].tag_
        number = tok.morph.get("Number")
        person = tok.morph.get("Person")
        singular_3rd = number == ["Sing"] and (person in ([], ["3"])) and tok.lower_ not in {"i", "you"}
        if singular_3rd and verb_tag == "VBP":
            errors += 1
        elif (number == ["Plur"] or tok.lower_ in {"i", "you"}) and verb_tag == "VBZ":
            errors += 1
    return errors


def linguistic_features(text: str) -> dict:
    doc = nlp()(text)
    sents = [s for s in doc.sents if any(t.is_alpha for t in s)]
    words = _words(text)
    n_words = max(len(words), 1)
    n_sents = max(len(sents), 1)
    sent_lens = np.array([sum(t.is_alpha for t in s) for s in sents] or [0])
    alpha = [t for t in doc if t.is_alpha]
    n_alpha = max(len(alpha), 1)

    # Moving-average type-token ratio is robust to transcript length (unlike plain TTR).
    window = 50
    if len(words) > window:
        mattr = np.mean([len(set(words[i : i + window])) / window for i in range(len(words) - window + 1)])
    else:
        mattr = len(set(words)) / n_words

    fragments = sum(
        1 for s in sents if not any(t.dep_ in {"nsubj", "nsubjpass", "expl"} for t in s)
        or not any(t.pos_ in {"VERB", "AUX"} for t in s)
    )
    return {
        "n_words": len(words),
        "n_sentences": len(sents),
        "sent_len_mean": float(sent_lens.mean()),
        "sent_len_std": float(sent_lens.std()),
        "sent_len_max": float(sent_lens.max()),
        "mattr": float(mattr),
        "avg_word_len": float(np.mean([len(w) for w in words])) if words else 0.0,
        "filler_rate": sum(w in FILLERS for w in words) / n_words,
        "repetition_rate": sum(a == b for a, b in zip(words, words[1:])) / n_words,
        "dep_depth_mean": float(np.mean([_depth(t) for t in alpha])) if alpha else 0.0,
        "dep_depth_max": max((_depth(t) for t in alpha), default=0),
        "subordinate_per_sent": sum(t.dep_ in SUBORDINATE_DEPS for t in doc) / n_sents,
        "conj_per_sent": sum(t.dep_ == "cc" for t in doc) / n_sents,
        "fragment_rate": fragments / n_sents,
        "agreement_err_per_sent": sum(_agreement_errors(s) for s in sents) / n_sents,
        "pron_ratio": sum(t.pos_ == "PRON" for t in alpha) / n_alpha,
        "verb_ratio": sum(t.pos_ == "VERB" for t in alpha) / n_alpha,
        "adj_adv_ratio": sum(t.pos_ in {"ADJ", "ADV"} for t in alpha) / n_alpha,
        "past_tense_ratio": sum(t.tag_ in {"VBD", "VBN"} for t in alpha) / max(sum(t.pos_ in {"VERB", "AUX"} for t in alpha), 1),
    }


class GrammarCorrector:
    """Wraps a T5 grammatical-error-correction model (prefix 'grammar: ')."""

    def __init__(self, model_name: str = GEC_MODEL):
        self.device = get_device()
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device).eval()

    @torch.inference_mode()
    def correct(self, sentences: list[str], batch_size: int = 64) -> list[str]:
        out: list[str] = []
        for batch in chunked(sentences, batch_size):
            enc = self.tok([f"grammar: {s}" for s in batch], return_tensors="pt", padding=True,
                           truncation=True, max_length=128).to(self.device)
            gen = self.model.generate(**enc, max_new_tokens=128, num_beams=1)  # greedy: ~4x faster, near-identical edits
            out += self.tok.batch_decode(gen, skip_special_tokens=True)
        return out


def word_edits(src: str, tgt: str) -> int:
    """Number of word-level insertions/deletions/substitutions between two strings."""
    a, b = _words(src), _words(tgt)
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    return sum(max(i2 - i1, j2 - j1) for op, i1, i2, j1, j2 in sm.get_opcodes() if op != "equal")


def gec_features(items: pd.DataFrame, transcripts: pd.Series) -> pd.DataFrame:
    corrector = None

    def compute(item: pd.Series) -> dict:
        nonlocal corrector
        corrector = corrector or GrammarCorrector()
        sents = [s.text.strip() for s in nlp()(transcripts[item["key"]]).sents if any(t.is_alpha for t in s)]
        if not sents:
            return {"gec_edit_rate": 0.0, "gec_changed_sent_frac": 0.0, "corrected": ""}
        fixed = corrector.correct(sents)
        edits = [word_edits(s, f) for s, f in zip(sents, fixed)]
        n_words = max(sum(len(_words(s)) for s in sents), 1)
        return {
            "gec_edit_rate": sum(edits) / n_words,
            "gec_changed_sent_frac": float(np.mean([e > 0 for e in edits])),
            "corrected": " ".join(fixed),
        }

    return cached_features("gec", items, compute, save_every=25)


def linguistic_table(items: pd.DataFrame, transcripts: pd.Series) -> pd.DataFrame:
    return cached_features("linguistic", items, lambda it: linguistic_features(transcripts[it["key"]]), save_every=200)


def text_embeddings(items: pd.DataFrame, transcripts: pd.Series) -> np.ndarray:
    """Sentence-transformer embedding of each full transcript, shape (n, d)."""
    model = None

    def compute(item: pd.Series) -> dict:
        nonlocal model
        if model is None:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(TEXT_EMBED_MODEL, device=get_device())
        vec = model.encode(transcripts[item["key"]], normalize_embeddings=True)
        return {"emb": vec.astype(np.float32).tolist()}

    table = cached_features("text_emb", items, compute, save_every=100)
    return np.vstack(table["emb"].to_numpy())
