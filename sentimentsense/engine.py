from __future__ import annotations

import math
import os
import re
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
        self._default_sentiment_model = os.getenv(
            "SENTSENSE_DEFAULT_SENTIMENT_MODEL",
            "distilbert-base-uncased-finetuned-sst-2-english",
        )
        self._default_sarcasm_model = os.getenv("SENTSENSE_DEFAULT_SARCASM_MODEL", "cardiffnlp/twitter-roberta-base-irony")
        self._enable_sarcasm_model = os.getenv("SENTSENSE_ENABLE_SARCASM_MODEL", "0").strip().lower() in {
            "1",
            "true",
            "yes",
        }
        self._enable_hosted_models = os.getenv("SENTSENSE_ENABLE_HOSTED_MODELS", "0").strip().lower() in {
            "1",
            "true",
            "yes",
        }
        self._batch_size = int(os.getenv("SENTSENSE_BATCH_SIZE", "32"))

    @property
    def load_errors(self) -> list[str]:
        return self._load_errors

    @property
    def inference_backend(self) -> str:
        """Return the backend currently available after model initialization."""
        self._ensure_models_initialized()
        if self._sentiment_pipe is not None or self._sarcasm_pipe is not None:
            return "transformers + heuristic fallback"
        return "offline heuristic"

    @property
    def sentiment_backend(self) -> str:
        self._ensure_models_initialized()
        return "transformer model" if self._sentiment_pipe is not None else "heuristic fallback"

    @property
    def sarcasm_backend(self) -> str:
        self._ensure_models_initialized()
        return "irony transformer" if self._sarcasm_pipe is not None else "heuristic fallback"

    def _initialize_models(self) -> None:
        sentiment_model_ref = None
        if self._local_sentiment_model.exists():
            sentiment_model_ref = str(self._local_sentiment_model)
        elif self._enable_hosted_models:
            sentiment_model_ref = self._default_sentiment_model

        sarcasm_model_ref = None
        if self._local_sarcasm_model.exists():
            sarcasm_model_ref = str(self._local_sarcasm_model)
        elif self._enable_hosted_models:
            sarcasm_model_ref = self._default_sarcasm_model

        if sentiment_model_ref is not None:
            try:
                from transformers import pipeline

                self._sentiment_pipe = pipeline(
                    "text-classification",
                    model=sentiment_model_ref,
                    truncation=True,
                )
            except Exception as exc:
                self._load_errors.append(f"Sentiment model fallback active: {exc}")
        else:
            self._load_errors.append("Hosted sentiment model disabled; heuristic sentiment active.")

        if self._enable_sarcasm_model and sarcasm_model_ref is not None:
            try:
                from transformers import pipeline

                self._sarcasm_pipe = pipeline(
                    "text-classification",
                    model=sarcasm_model_ref,
                    truncation=True,
                )
            except Exception as exc:
                self._load_errors.append(f"Sarcasm model fallback active: {exc}")
        else:
            self._load_errors.append("Sarcasm transformer disabled for speed; heuristic sarcasm active.")

    def analyze_text(self, text: str) -> Prediction:
        self._ensure_models_initialized()
        cleaned = clean_text(text)
        if not cleaned:
            raise ValueError("Text must contain at least one analyzable character.")
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
        self._ensure_models_initialized()
        if df.empty:
            return df.copy()
        if text_col not in df.columns:
            raise ValueError(f"Text column '{text_col}' was not found in the uploaded data.")

        work_df = df.copy()
        raw_values = work_df[text_col].tolist()
        valid_positions = [
            position
            for position, value in enumerate(raw_values)
            if pd.notna(value) and str(value).strip() != ""
        ]
        valid_texts = [str(raw_values[position]) for position in valid_positions]
        valid_cleaned_texts = [clean_text(text) for text in valid_texts]

        if valid_cleaned_texts:
            sentiment_results = self._predict_sentiments_batch(valid_cleaned_texts)
            sarcasm_results = self._predict_sarcasm_batch(valid_cleaned_texts)
        else:
            sentiment_results = []
            sarcasm_results = []

        predictions_by_position: dict[
            int, tuple[tuple[str, float, float], tuple[str, float]]
        ] = {}
        for position, sent, sarc in zip(valid_positions, sentiment_results, sarcasm_results):
            predictions_by_position[position] = (sent, sarc)

        cleaned_texts: list[Any] = []
        sentiment_labels: list[Any] = []
        sentiment_confs: list[Any] = []
        sentiment_scores: list[Any] = []
        sarcasm_labels: list[Any] = []
        sarcasm_confs: list[Any] = []
        final_labels: list[Any] = []
        analysis_statuses: list[str] = []

        for position, value in enumerate(raw_values):
            if position not in predictions_by_position:
                cleaned_texts.append(pd.NA)
                sentiment_labels.append(pd.NA)
                sentiment_confs.append(pd.NA)
                sentiment_scores.append(pd.NA)
                sarcasm_labels.append(pd.NA)
                sarcasm_confs.append(pd.NA)
                final_labels.append(pd.NA)
                analysis_statuses.append("skipped_blank_text")
                continue

            cleaned_text = clean_text(str(value))
            sent, sarc = predictions_by_position[position]
            sentiment_label, sentiment_conf, sentiment_score = sent
            sarcasm_label, sarcasm_conf = sarc
            final_label = "sarcastic" if sarcasm_label == "sarcastic" else sentiment_label

            cleaned_texts.append(cleaned_text)
            sentiment_labels.append(sentiment_label)
            sentiment_confs.append(sentiment_conf)
            sentiment_scores.append(sentiment_score)
            sarcasm_labels.append(sarcasm_label)
            sarcasm_confs.append(sarcasm_conf)
            final_labels.append(final_label)
            analysis_statuses.append("analyzed")

        work_df["cleaned_text"] = cleaned_texts
        work_df["sentiment_label"] = sentiment_labels
        work_df["sentiment_confidence"] = sentiment_confs
        work_df["sentiment_score"] = sentiment_scores
        work_df["sarcasm_label"] = sarcasm_labels
        work_df["sarcasm_confidence"] = sarcasm_confs
        work_df["final_label"] = final_labels
        work_df["analysis_status"] = analysis_statuses
        return work_df

    def _predict_sentiments_batch(self, texts: list[str]) -> list[tuple[str, float, float]]:
        if self._sentiment_pipe is None:
            return [self._heuristic_sentiment(text) for text in texts]

        outputs = self._sentiment_pipe(texts, batch_size=self._batch_size, truncation=True)
        return [self._normalize_sentiment_output(out) for out in outputs]

    def _predict_sarcasm_batch(self, texts: list[str]) -> list[tuple[str, float]]:
        if self._sarcasm_pipe is None:
            return [self._heuristic_sarcasm(text) for text in texts]

        outputs = self._sarcasm_pipe(texts, batch_size=self._batch_size, truncation=True)
        parsed: list[tuple[str, float]] = []
        for out in outputs:
            label = str(out["label"]).lower()
            score = float(out["score"])
            if "irony" in label or "sarcas" in label:
                parsed.append(("sarcastic", score))
            else:
                parsed.append(("not_sarcastic", score))
        return parsed

    @staticmethod
    def _normalize_sentiment_output(out: dict[str, Any]) -> tuple[str, float, float]:
        if not isinstance(out, dict) or "label" not in out or "score" not in out:
            raise ValueError(
                "Sentiment transformer output must contain 'label' and 'score' fields."
            )
        try:
            label = str(out["label"]).strip().lower()
            score = float(out["score"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Sentiment transformer output contains an invalid label or score."
            ) from exc
        if not math.isfinite(score):
            raise ValueError(
                "Sentiment transformer output contains a non-finite score."
            )

        if "neg" in label:
            return "negative", score, -score
        if "neu" in label:
            return "neutral", score, 0.0
        if "pos" in label:
            return "positive", score, score

        raise ValueError(
            f"Unrecognized sentiment transformer label {label!r}; "
            "cannot infer a sentiment class from this output."
        )

    def _predict_sentiment(self, text: str) -> tuple[str, float, float]:
        if self._sentiment_pipe is not None:
            out = self._sentiment_pipe(text[:512], truncation=True)[0]
            return self._normalize_sentiment_output(out)

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
            "smooth",
            "fast",
            "improved",
            "saved",
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
            "disappointing",
            "slow",
            "failing",
            "crashes",
            "crash",
            "frustrating",
            "unstable",
            "buggy",
            "outages",
            "broken",
            "broke",
            "latency",
        }

        words = re.findall(r"[a-z]+", text.lower())
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
        lowered = text.lower()
        phrase_hit = any(marker in lowered for marker in ("yeah right", "as if", "just what we needed"))
        contrast_hit = any(
            marker in lowered for marker in ("sure,", "totally perfect", "favorite part", "waiting forever")
        )
        exaggerated_praise = any(marker in lowered for marker in ("awesome", "great job", "amazing timing"))
        negative_context = any(
            marker in lowered
            for marker in ("crash", "crashes", "broke", "broken", "waiting", "latency", "slow", "error")
        )
        exclamations = text.count("!")
        quote_flip = '"' in text and any(w in lowered for w in ("great", "amazing", "perfect"))

        score = (
            0.15
            + 0.4 * phrase_hit
            + 0.4 * contrast_hit
            + 0.4 * (exaggerated_praise and negative_context)
            + 0.1 * (exclamations >= 2)
            + 0.2 * quote_flip
        )
        if score >= 0.55:
            return "sarcastic", min(score, 0.9)
        return "not_sarcastic", 1 - min(score + 0.2, 0.8)
