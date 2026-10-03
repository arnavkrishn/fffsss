# Grammar scoring engine for spoken English

SHL hiring assessment 2026 (Kaggle). The engine predicts a 0–5 MOS grammar score from a 45–60 s `.wav` recording.

## Results (5-fold × 3-repeat stratified CV, 732 genuine training recordings)

| Version | CV RMSE | CV Pearson | Training RMSE (all 769 files) |
|---|---|---|---|
| v1: linguistic, GEC, prosody, text and WavLM-base embeddings | 0.542 | 0.845 | 0.388 |
| v2: + Gemini rubric grader | 0.530 | 0.853 | 0.332 |
| **v3: + WavLM-large, all-layer mean+std pooling (submitted)** | **0.472** | **0.885** | **0.161** |

## Approach
1. **Data checks first.** The 37 training files labelled 0 are noise-corrupted audio. Spectral flatness separates
   them perfectly, so a fixed gate predicts 0 for them and the grammar models never train on them. train/ and test/
   share 212 filenames for *different* recordings, so every file is keyed as `split/filename`.
2. **Transcribe** with Whisper large-v3-turbo. Whisper keeps learner errors such as "he don't".
3. **Interpretable features.**
   * spaCy syntax statistics: depth, subordinate clauses, fragments, agreement errors
   * the T5 grammar-correction edit rate
   * pause and pitch statistics
   * Gemini grades against the rubric, each with a written rationale
4. **Learned representations.** mpnet transcript embeddings, plus WavLM-base and WavLM-large hidden states with
   every layer concatenated.
5. **Models.** Ridge, SVR and LightGBM base models with out-of-fold CV, blended by a non-negative linear stacker
   that is evaluated with nested CV. SHAP values and LLM rationales explain individual scores.

Full report, plots and evaluation: [`notebooks/grammar_scoring.ipynb`](notebooks/grammar_scoring.ipynb).
Predictions: [`submissions/arnav_kushwaha.csv`](submissions/arnav_kushwaha.csv). The competition audio is not
included; the competition rules forbid redistributing it.

## Run
```bash
uv venv -p 3.12 .venv && source .venv/bin/activate
uv pip install -r requirements.txt && python -m spacy download en_core_web_sm
# Kaggle credentials in ~/.kaggle/kaggle.json (the data downloads automatically), or set GRAMMAR_DATA_DIR
# Optional LLM rubric grader: put GEMINI_API_KEY=... in .env (git-ignored). Without it the step is skipped.
jupyter lab notebooks/grammar_scoring.ipynb
```
The first run extracts features (about 1 h on an Apple M4 Pro, faster on a CUDA GPU) and caches them in `cache/`.
Later runs take seconds. The submission is written to `submissions/submission.csv`.

## Layout
| Path | Purpose |
|---|---|
| `notebooks/grammar_scoring.ipynb` | report, EDA, training, evaluation, plots, submission |
| `src/data.py` | locate and load the competition files |
| `src/transcribe.py` | Whisper speech-to-text |
| `src/text_features.py` | spaCy linguistic stats, GEC edit rate, sentence embeddings |
| `src/llm_grader.py` | Gemini rubric grader (scores + rationale, 25 transcripts per request) |
| `src/audio_features.py` | pause/pitch/energy stats, WavLM embeddings |
| `src/modeling.py` | repeated stratified CV, base models, non-negative stacker |
| `src/cache.py` | resumable per-file feature cache |
| `src/plots.py` | figure styling |
