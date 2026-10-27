from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import IO, Literal

import pandas as pd

TEXT_COLUMN = "text"
SENTIMENT_COLUMN = "sentiment_label"
SARCASM_COLUMN = "sarcasm_label"

VALID_SENTIMENT_LABELS = frozenset({"positive", "negative", "neutral"})
VALID_SARCASM_LABELS = frozenset({"sarcastic", "not_sarcastic"})
REQUIRED_LABELED_COLUMNS = frozenset({TEXT_COLUMN, SENTIMENT_COLUMN, SARCASM_COLUMN})
ValidationMode = Literal["sentiment", "sarcasm", "joint"]


def load_labeled_csv(
    source: str | Path | IO[str],
    *,
    text_column: str = TEXT_COLUMN,
    sentiment_column: str = SENTIMENT_COLUMN,
    sarcasm_column: str = SARCASM_COLUMN,
    task: ValidationMode = "joint",
    sentiment_label_mapping: dict[str, str] | None = None,
    sarcasm_label_mapping: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Load and validate a CSV used for supervised training or evaluation."""
    if isinstance(source, (str, Path)):
        raw_csv = Path(source).read_text(encoding="utf-8-sig")
    else:
        raw_csv = source.read()

    rows = list(csv.reader(io.StringIO(raw_csv)))
    if not rows:
        raise ValueError("CSV is empty")
    expected_fields = len(rows[0])
    malformed_rows = [
        row_number
        for row_number, row in enumerate(rows[1:], start=2)
        if len(row) != expected_fields
    ]
    if malformed_rows:
        raise ValueError(
            f"Malformed CSV rows {malformed_rows}: expected {expected_fields} fields"
        )

    df = pd.read_csv(io.StringIO(raw_csv))
    validate_labeled_dataframe(
        df,
        text_column=text_column,
        sentiment_column=sentiment_column,
        sarcasm_column=sarcasm_column,
        task=task,
        sentiment_label_mapping=sentiment_label_mapping,
        sarcasm_label_mapping=sarcasm_label_mapping,
    )
    return df


def validate_labeled_dataframe(
    df: pd.DataFrame,
    *,
    text_column: str = TEXT_COLUMN,
    sentiment_column: str = SENTIMENT_COLUMN,
    sarcasm_column: str = SARCASM_COLUMN,
    task: ValidationMode = "joint",
    sentiment_label_mapping: dict[str, str] | None = None,
    sarcasm_label_mapping: dict[str, str] | None = None,
) -> None:
    if task not in {"sentiment", "sarcasm", "joint"}:
        raise ValueError(f"Unknown validation task: {task}")

    required_columns = {text_column}
    if task in {"sentiment", "joint"}:
        required_columns.add(sentiment_column)
    if task in {"sarcasm", "joint"}:
        required_columns.add(sarcasm_column)
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    text_values = df[text_column]
    if text_values.isna().any() or text_values.astype(str).str.strip().eq("").any():
        raise ValueError(f"Column '{text_column}' contains missing or empty values")

    label_columns = []
    if task in {"sentiment", "joint"}:
        label_columns.append((sentiment_column, VALID_SENTIMENT_LABELS))
    if task in {"sarcasm", "joint"}:
        label_columns.append((sarcasm_column, VALID_SARCASM_LABELS))

    for column, valid_labels in label_columns:
        if column == sentiment_column and sentiment_label_mapping is not None:
            valid_labels = frozenset(sentiment_label_mapping)
        if column == sarcasm_column and sarcasm_label_mapping is not None:
            valid_labels = frozenset(sarcasm_label_mapping)
        values = set(df[column].dropna().astype(str))
        if df[column].isna().any():
            raise ValueError(f"Column '{column}' contains missing values")
        invalid_labels = values.difference(valid_labels)
        if invalid_labels:
            raise ValueError(f"Invalid {column} values: {sorted(invalid_labels)}")
