# SentimentSense

SentimentSense is a real-time sentiment analysis and sarcasm detection platform built with Streamlit.

## Features

- Single-text inference with sentiment + sarcasm classification
- Batch CSV processing with downloadable results
- Live feed simulator for streaming-style text analysis
- Interactive charts (distribution + sentiment trend)
- Chat with Analytics (Gemini-powered conversational insights)
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

Set in `.env`:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

4. Run the app:

```bash
streamlit run app.py
```

5. Open the local URL shown in terminal (usually `http://localhost:8501`).

## Local Model Training (Next Phase)

You can fine-tune local models for sentiment and sarcasm, then have the app use them automatically.

### Dataset format

Training CSV must include:

- `text`
- `sentiment_label` (e.g., `positive`, `negative`, `neutral`)
- `sarcasm_label` (e.g., `sarcastic`, `not_sarcastic`)

A starter file is provided at `data/train_sample.csv`.

For larger study datasets, generate synthetic data with:

```bash
python scripts/generate_big_sample_data.py --rows 8000 --output data/big_sample_data.csv
```

You can change `--rows` for any dataset size.

### Train both models

```bash
python scripts/train_models.py --data data/train_sample.csv --output-dir models --epochs 2 --batch-size 8
```

After training, local models are saved in:

- `models/sentiment`
- `models/sarcasm`

### Use local models in app

By default, the app checks `models/sentiment` and `models/sarcasm` first.
You can override with environment variables:

```env
SENTSENSE_SENTIMENT_MODEL_DIR=models/sentiment
SENTSENSE_SARCASM_MODEL_DIR=models/sarcasm
```

If local models are missing/unavailable, SentimentSense falls back to hosted transformer models, then to heuristics.

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

## Gemini Data Flow

SentimentSense keeps prediction and reasoning separate:

User Input → `preprocess.py` → `engine.py` (sentiment/sarcasm inference) → `analytics.py` (summary/trends)
→ Structured Analytics JSON → Gemini → Conversational Insight Response → Streamlit Chat UI

Gemini receives only SentimentSense outputs (labels, sarcasm flag, confidence, aggregates, time-series, domain context).
Gemini does not perform or replace sentiment/sarcasm classification.

## Notes

- First launch may take time if transformer models are downloaded.
- If model loading fails (e.g., no internet), app automatically uses rule-based fallback.
- If `GEMINI_API_KEY` is missing or Gemini is unavailable, chat mode remains functional using local insight fallback.
