from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Any

import pandas as pd

from .preprocess import clean_text


@dataclass
class Prediction:
    sentiment_label: str
    sentiment_confidence: float
    sentiment_score: float
    sarcasm_label: str
    sarcasm_confidence: float
    final_label: str
    cleaned_text: str


class SentimentEngine:
    def __init__(self) -> None:
        self._sentiment_pipe = None
        self._sarcasm_pipe = None
        self._load_errors: list[str] = []
        self._models_initialized = False
        self._local_sentiment_model = Path(os.getenv("SENTSENSE_SENTIMENT_MODEL_DIR", "models/sentiment"))
        self._local_sarcasm_model = Path(os.getenv("SENTSENSE_SARCASM_MODEL_DIR", "models/sarcasm"))

    @property
    def load_errors(self) -> list[str]:
        return self._load_errors

    def _initialize_models(self) -> None:
        sentiment_model_ref = (
            str(self._local_sentiment_model)
            if self._local_sentiment_model.exists()
            else "cardiffnlp/twitter-roberta-base-sentiment-latest"
        )
        sarcasm_model_ref = (
            str(self._local_sarcasm_model)
            if self._local_sarcasm_model.exists()
            else "cardiffnlp/twitter-roberta-base-irony"
        )

        try:
            from transformers import pipeline

            self._sentiment_pipe = pipeline(
                "text-classification",
                model=sentiment_model_ref,
                truncation=True,
            )
        except Exception as exc:
            self._load_errors.append(f"Sentiment model fallback active: {exc}")

        try:
            from transformers import pipeline

            self._sarcasm_pipe = pipeline(
                "text-classification",
                model=sarcasm_model_ref,
                truncation=True,
            )
        except Exception as exc:
            self._load_errors.append(f"Sarcasm model fallback active: {exc}")

    def analyze_text(self, text: str) -> Prediction:
        self._ensure_models_initialized()
        cleaned = clean_text(text)
        sentiment_label, sentiment_conf, sentiment_score = self._predict_sentiment(cleaned)
        sarcasm_label, sarcasm_conf = self._predict_sarcasm(cleaned)

        final_label = "sarcastic" if sarcasm_label == "sarcastic" else sentiment_label

        return Prediction(
            sentiment_label=sentiment_label,
            sentiment_confidence=sentiment_conf,
            sentiment_score=sentiment_score,
            sarcasm_label=sarcasm_label,
            sarcasm_confidence=sarcasm_conf,
            final_label=final_label,
            cleaned_text=cleaned,
        )

    def _ensure_models_initialized(self) -> None:
        if self._models_initialized:
            return
        self._initialize_models()
        self._models_initialized = True

    def analyze_dataframe(self, df: pd.DataFrame, text_col: str) -> pd.DataFrame:
        records: list[dict[str, Any]] = []
        for _, row in df.iterrows():
            text = str(row[text_col]) if pd.notna(row[text_col]) else ""
            pred = self.analyze_text(text)
            enriched = row.to_dict()
            enriched.update(pred.__dict__)
            records.append(enriched)
        return pd.DataFrame(records)

    def _predict_sentiment(self, text: str) -> tuple[str, float, float]:
        if self._sentiment_pipe is not None:
            out = self._sentiment_pipe(text[:512])[0]
            label = str(out["label"]).lower()
            score = float(out["score"])

            if "neg" in label:
                return "negative", score, -score
            if "neu" in label:
                return "neutral", score, 0.0
            return "positive", score, score

        return self._heuristic_sentiment(text)

    def _predict_sarcasm(self, text: str) -> tuple[str, float]:
        if self._sarcasm_pipe is not None:
            out = self._sarcasm_pipe(text[:512])[0]
            label = str(out["label"]).lower()
            score = float(out["score"])
            if "irony" in label or "sarcas" in label:
                return "sarcastic", score
            return "not_sarcastic", score

        return self._heuristic_sarcasm(text)

    @staticmethod
    def _heuristic_sentiment(text: str) -> tuple[str, float, float]:
        positive_words = {
            "good",
            "great",
            "awesome",
            "love",
            "excellent",
            "happy",
            "amazing",
            "best",
        }
        negative_words = {
            "bad",
            "terrible",
            "awful",
            "hate",
            "worst",
            "angry",
            "poor",
            "disappointed",
        }

        words = text.split()
        pos_hits = sum(1 for w in words if w in positive_words)
        neg_hits = sum(1 for w in words if w in negative_words)

        total = max(1, pos_hits + neg_hits)
        delta = pos_hits - neg_hits

        if delta > 0:
            confidence = min(0.85, 0.55 + abs(delta) / total * 0.3)
            return "positive", confidence, confidence
        if delta < 0:
            confidence = min(0.85, 0.55 + abs(delta) / total * 0.3)
            return "negative", confidence, -confidence
        return "neutral", 0.55, 0.0

    @staticmethod
    def _heuristic_sarcasm(text: str) -> tuple[str, float]:
        sarcasm_markers = ["yeah right", "as if", "totally", "sure", "great..."]
        exclamations = text.count("!")
        marker_hit = any(marker in text for marker in sarcasm_markers)
        quote_flip = '"' in text and any(w in text for w in ["great", "amazing", "perfect"])

        score = 0.2 + 0.2 * marker_hit + 0.2 * (exclamations >= 2) + 0.2 * quote_flip
        if score >= 0.55:
            return "sarcastic", min(score, 0.9)
        return "not_sarcastic", 1 - min(score + 0.2, 0.8)
