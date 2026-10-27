# SentimentSense

SentimentSense is a customer-feedback intelligence application built with Streamlit. It separates
what a customer says (sentiment) from how they say it (sarcasm), then turns individual predictions
and batch feedback into trends and recommended actions.

## Features

- Single-text inference with sentiment + sarcasm classification
- Batch CSV processing with downloadable results
- Built-in customer-feedback demo dataset for repeatable presentations
- Live feed simulator for streaming-style text analysis
- Interactive charts (distribution + sentiment trend)
- Chat with Analytics (optional Gemini-powered conversational insights)
- Explainable response layer with graceful local fallback when Gemini is unavailable
- Hybrid inference engine:
  - Transformer models (when available)
  - Heuristic fallback (offline-safe mode)

## Tech Stack

- Python
- Streamlit
- Hugging Face Transformers
- Plotly
- Pandas
- Google Gemini API (`google-generativeai`)

## Quick Start

1. Create and activate a Python virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. (Optional, recommended) Configure Gemini API key for conversational analytics:

```bash
cp .env.example .env
```

On Windows PowerShell, use:

```powershell
Copy-Item .env.example .env
```

Set in `.env`:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

4. Run the app:

```bash
streamlit run app.py
```

5. Open the local URL shown in terminal (usually `http://localhost:8501`).

### Offline demo

Gemini credentials and model downloads are not required for the basic demo. The supplied
`.env.example` disables hosted Transformer downloads and the sarcasm Transformer, so the app uses
the deterministic heuristic fallback. If you do not create `.env`, the same built-in defaults apply.
Gemini-powered chat is optional; without `GEMINI_API_KEY`, Chat with Analytics remains available
through the local insight fallback.

The bundled demo CSV is `data/big_sample_data.csv`. In the app, select **Batch CSV** and leave
**Use the built-in demo dataset** selected to load it automatically.

## Evaluation

Run the evaluation command against a labeled CSV. The bundled dataset is synthetic and is useful
for an offline smoke test, not for estimating real-world generalization:

```bash
python -m scripts.evaluate_models --data data/big_sample_data.csv --dataset-type synthetic --output evaluation/report.json
```

The report includes separate accuracy, precision, recall, F1, confusion matrices, backend
information, evaluation method, dataset provenance, limitations, and measured median/p95 latency for
sentiment and sarcasm. Results on `data/big_sample_data.csv` do not establish performance on
real-world customer feedback because the data is generated from project templates that overlap
with the offline heuristic rules. Historical accuracy claims from the original project should not
be treated as benchmarks.

Evaluation is task-specific:

```bash
python -m scripts.evaluate_models --data data/big_sample_data.csv --task sentiment
python -m scripts.evaluate_models --data data/big_sample_data.csv --task sarcasm
python -m scripts.evaluate_models --data data/big_sample_data.csv --task joint
```

Sentiment-only inputs require `text` and `sentiment_label`; sarcasm-only inputs require `text` and
`sarcasm_label`; joint inputs require all three columns. Use `--text-col`, `--sentiment-col`, and
`--sarcasm-col` for different source column names. Source labels must already use the project's
labels, or be explicitly mapped with JSON such as
`--sentiment-label-map '{"pos":"positive","neg":"negative"}'`. Unknown or incomplete mappings are
rejected. The default sentiment scope is three-class (`positive`, `negative`, `neutral`). Because
the default transformer sentiment model is binary, `--sentiment-scope binary` is an explicit
positive/negative-only evaluation that excludes neutral rows and reports the exclusion; it is not
three-class performance. External datasets must be independently labeled and their licenses,
commercial-use terms, and platform restrictions checked before download or evaluation.

The default configuration is offline-safe. It uses local fine-tuned Transformer models when
available. If a local model is not present and hosted loading is enabled, the configured hosted
model may be downloaded; unavailable model backends use the deterministic heuristic fallback.
Set `SENTSENSE_ENABLE_HOSTED_MODELS=1` only when model downloads are intended.

## Local Model Training

You can fine-tune local models for sentiment and sarcasm, then have the app use them automatically.

### Dataset format

Training CSV must include:

- `text`
- `sentiment_label` (e.g., `positive`, `negative`, `neutral`)
- `sarcasm_label` (e.g., `sarcastic`, `not_sarcastic`)

A validated starter file is provided at `data/train_sample.csv`. Text values containing commas
are quoted according to CSV rules. This six-row file is a validation fixture, not a viable
training dataset: it is too small to produce stratified train/evaluation splits for all three
sentiment classes. The training command rejects it before creating model output.

For larger study datasets, generate synthetic data with:

```bash
python -m scripts.generate_big_sample_data --rows 8000 --output data/big_sample_data.csv
```

You can change `--rows` for any dataset size.

### Train both models

Use a sufficiently large, independently labeled CSV for training. It must contain enough
examples in every sentiment and sarcasm class for each class to appear in both stratified
training and evaluation splits. Install the dependencies first if needed:

```bash
python -m pip install -r requirements.txt
python -m scripts.train_models --data path/to/train.csv --output-dir models --epochs 2 --batch-size 8
```

The same commands in Windows PowerShell are:

```powershell
python -m pip install -r requirements.txt
python -m scripts.train_models --data .\path\to\train.csv --output-dir .\models --epochs 2 --batch-size 8
```

After training, local models are saved in:

- `models/sentiment`
- `models/sarcasm`

### Use local models in app

By default, the app checks `models/sentiment` and `models/sarcasm` first. The local sentiment
model is used when available; the local sarcasm/irony model also requires
`SENTSENSE_ENABLE_SARCASM_MODEL=1`.
You can override with environment variables:

```env
SENTSENSE_SENTIMENT_MODEL_DIR=models/sentiment
SENTSENSE_SARCASM_MODEL_DIR=models/sarcasm
```

If a model backend is unavailable, SentimentSense uses the rule-based heuristic fallback for that
task. Hosted Transformer loading is attempted only when explicitly enabled and no local model
directory is available for that task.

### Speed Tuning (Important)

If analysis feels slow, use these environment variables in `.env`:

```env
# Disable hosted model downloads for fastest startup (default behavior)
SENTSENSE_ENABLE_HOSTED_MODELS=0

# Keep sarcasm transformer disabled for faster responses (default behavior)
SENTSENSE_ENABLE_SARCASM_MODEL=0

# Increase batch inference throughput for CSV analysis
SENTSENSE_BATCH_SIZE=32

# Optional lighter sentiment model (default already optimized)
SENTSENSE_DEFAULT_SENTIMENT_MODEL=distilbert-base-uncased-finetuned-sst-2-english
```

If you want hosted Transformer inference instead of the default offline fallback, set
`SENTSENSE_ENABLE_HOSTED_MODELS=1`. To enable the optional sarcasm/irony Transformer, also set
`SENTSENSE_ENABLE_SARCASM_MODEL=1`.

The application still installs the packages listed in `requirements.txt` for the complete project,
but Transformer model loading and Gemini API use are optional at runtime. The offline demo does not
download models or call Gemini.

## Docker Deployment

Build image:

```bash
docker build -t sentimentsense:latest .
```

Run container:

```bash
docker run --rm -p 8501:8501 --env-file .env sentimentsense:latest
```

## Input Expectations

- **Single Text mode:** any plain text input
- **Batch CSV mode:** CSV with at least one text column; optional timestamp column for trend chart
- **Live Feed mode:** one message per line
- **Chat with Analytics mode:** ask follow-up questions over the latest structured analytics context
  - Includes optional date-range scoping (when timestamps exist)
  - Includes domain-focus override for targeted conversational insights
  - Includes quick time presets: Today, Last 7 Days, Last 30 Days, Custom
  - Shows auto-detected trend-event summary (spikes/drops)
  - Provides one-click quick questions for trend summary, top drop analysis, and action recommendations
  - Supports transcript export as JSON and Markdown with scoped analytics context
  - Optional auto-save of transcripts to local `exports/` directory
  - Includes a Saved Sessions browser to preview and download prior exports
  - Allows deleting a selected saved session with explicit confirmation

For the fastest demo, open **Batch CSV** and keep **Use the built-in demo dataset**
selected. For uploaded files, provide one text column; a timestamp column is optional.

## Manual demo checklist

1. **Single Text:** enter feedback and run analysis. Sentiment and Sarcasm are separate predictions.
   `Final Label` is a retained compatibility field, not a third sentiment class.
2. **Bundled demo:** open **Batch CSV** and run the built-in `data/big_sample_data.csv` dataset.
3. **Valid upload:** upload a CSV with a text column, select that column, and download the results.
4. **Mixed blank rows:** upload a CSV containing valid text plus null, empty, or whitespace-only
   text. Blank rows are retained with `analysis_status=skipped_blank_text`, blank prediction cells,
   and are excluded from charts and analytics.
5. **All-blank upload:** confirm the original rows are displayed, stale analytics context is cleared,
   no charts or prediction CSV are produced, and the explanatory warning is shown.
6. **Live Feed:** paste messages with blank lines. Blank lines are counted and skipped; all-blank
   input does not run inference.
7. **Chat fallback:** open Chat with Analytics without `GEMINI_API_KEY`. The local fallback remains
   usable and is identified as a local insight fallback.

## Gemini Data Flow

SentimentSense keeps prediction and reasoning separate:

User Input → `preprocess.py` → `engine.py` (sentiment/sarcasm inference) → `analytics.py` (summary/trends)
→ Structured Analytics JSON → Gemini → Conversational Insight Response → Streamlit Chat UI

Gemini receives only SentimentSense outputs (labels, sarcasm flag, confidence, aggregates, time-series, domain context).
Gemini does not perform or replace sentiment/sarcasm classification.

## Notes

- If the configured model backend cannot be used, offline analysis remains available through the
  rule-based heuristic fallback. Backend diagnostics are not displayed in the sidebar, and
  heuristic scores are backend-dependent rather than calibrated probabilities.
- If `GEMINI_API_KEY` is missing or Gemini is unavailable, chat mode remains functional using local insight fallback.
- The default SST-2 sentiment model is binary; neutral output is a rule-based fallback limitation,
  not a third learned SST-2 class.
