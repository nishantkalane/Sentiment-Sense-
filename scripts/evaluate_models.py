from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sentimentsense.dataset import (
    VALID_SARCASM_LABELS,
    VALID_SENTIMENT_LABELS,
    load_labeled_csv,
)
from sentimentsense.engine import SentimentEngine


TASKS = ("sentiment", "sarcasm", "joint")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SentimentSense on a labeled CSV.")
    parser.add_argument("--data", type=Path, required=True, help="CSV containing text and label columns.")
    parser.add_argument("--task", choices=TASKS, default="joint")
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--sentiment-col", default="sentiment_label")
    parser.add_argument("--sarcasm-col", default="sarcasm_label")
    parser.add_argument(
        "--sentiment-scope",
        choices=["three_class", "binary"],
        default="three_class",
        help="Use three-class sentiment, or explicitly exclude neutral rows for binary evaluation.",
    )
    parser.add_argument("--sentiment-label-map", help="JSON object mapping source labels to internal labels.")
    parser.add_argument("--sarcasm-label-map", help="JSON object mapping source labels to internal labels.")
    parser.add_argument("--dataset-name", default="")
    parser.add_argument("--source-url", default="")
    parser.add_argument("--dataset-type", choices=["synthetic", "external", "unknown"], default="unknown")
    parser.add_argument("--split", default="")
    parser.add_argument("--output", type=Path, help="Optional JSON report path.")
    return parser.parse_args()


def _parse_mapping(raw_mapping: str | None, valid_labels: frozenset[str], task: str) -> dict[str, str]:
    if not raw_mapping:
        return {label: label for label in valid_labels}
    try:
        mapping = json.loads(raw_mapping)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{task} label mapping must be valid JSON: {exc.msg}") from exc
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError(f"{task} label mapping must be a non-empty JSON object")
    if any(not isinstance(key, str) or not isinstance(value, str) for key, value in mapping.items()):
        raise ValueError(f"{task} label mapping keys and values must be strings")
    invalid_targets = set(mapping.values()).difference(valid_labels)
    if invalid_targets:
        raise ValueError(f"{task} label mapping has invalid target labels: {sorted(invalid_targets)}")
    if len(set(mapping.values())) != len(mapping):
        raise ValueError(f"{task} label mapping is ambiguous: multiple source labels map to one target")
    return mapping


def _map_labels(
    values: pd.Series,
    mapping: dict[str, str],
    column: str,
) -> tuple[pd.Series, int]:
    source_values = values.astype(str)
    unknown = sorted(set(source_values).difference(mapping))
    if unknown:
        raise ValueError(f"Unknown {column} labels {unknown}; provide an explicit complete mapping")
    mapped = source_values.map(mapping)
    return mapped, 0


def _prediction_rows(engine: Any, texts: list[str]) -> tuple[pd.DataFrame, list[float]]:
    predictions: list[dict[str, Any]] = []
    latencies_ms: list[float] = []
    for text in texts:
        started = time.perf_counter()
        prediction = engine.analyze_text(text)
        latencies_ms.append((time.perf_counter() - started) * 1000)
        predictions.append(prediction.__dict__ if hasattr(prediction, "__dict__") else dict(prediction))
    return pd.DataFrame(predictions), latencies_ms


def _task_metrics(
    truth: pd.Series,
    predicted: pd.Series,
    labels: frozenset[str],
) -> dict[str, Any]:
    ordered_labels = sorted(labels)
    report = classification_report(
        truth,
        predicted,
        labels=ordered_labels,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": round(float(accuracy_score(truth, predicted)), 4),
        "macro_f1": round(float(f1_score(truth, predicted, labels=ordered_labels, average="macro", zero_division=0)), 4),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(truth, predicted, labels=ordered_labels).tolist(),
        "labels": ordered_labels,
        "class_distribution": truth.value_counts().reindex(ordered_labels, fill_value=0).to_dict(),
    }


def evaluate(args: argparse.Namespace, engine: Any | None = None) -> dict[str, Any]:
    task = getattr(args, "task", "joint")
    sentiment_scope = getattr(args, "sentiment_scope", "three_class")
    sentiment_mapping = _parse_mapping(
        getattr(args, "sentiment_label_map", None), VALID_SENTIMENT_LABELS, "sentiment"
    )
    sarcasm_mapping = _parse_mapping(
        getattr(args, "sarcasm_label_map", None), VALID_SARCASM_LABELS, "sarcasm"
    )
    labeled = load_labeled_csv(
        args.data,
        text_column=args.text_col,
        sentiment_column=args.sentiment_col,
        sarcasm_column=args.sarcasm_col,
        task=task,
        sentiment_label_mapping=sentiment_mapping if task in {"sentiment", "joint"} else None,
        sarcasm_label_mapping=sarcasm_mapping if task in {"sarcasm", "joint"} else None,
    ).copy()

    if task == "joint" and sentiment_scope == "binary":
        raise ValueError("Binary sentiment scope is only valid for sentiment-only evaluation")

    if engine is None:
        engine = SentimentEngine()
    if task in {"sentiment", "joint"}:
        labeled[args.sentiment_col], _ = _map_labels(labeled[args.sentiment_col], sentiment_mapping, args.sentiment_col)
    if task in {"sarcasm", "joint"}:
        labeled[args.sarcasm_col], _ = _map_labels(labeled[args.sarcasm_col], sarcasm_mapping, args.sarcasm_col)

    if task == "sentiment" and sentiment_scope == "binary":
        excluded = int((labeled[args.sentiment_col] == "neutral").sum())
        if excluded:
            labeled = labeled[labeled[args.sentiment_col] != "neutral"].copy()
    else:
        excluded = 0

    predictions, latencies_ms = _prediction_rows(engine, labeled[args.text_col].astype(str).tolist())
    report: dict[str, Any] = {
        "data": str(args.data),
        "dataset_name": getattr(args, "dataset_name", ""),
        "source_url": getattr(args, "source_url", ""),
        "dataset_type": getattr(args, "dataset_type", "unknown"),
        "split": getattr(args, "split", ""),
        "task": task,
        "evaluation_method": "row-wise evaluation through SentimentEngine.analyze_text",
        "rows": len(labeled),
        "excluded_rows": excluded,
        "excluded_reason": "neutral sentiment rows excluded by explicit binary scope" if excluded else "",
        "backend": engine.inference_backend,
        "sentiment_backend": engine.sentiment_backend,
        "sarcasm_backend": engine.sarcasm_backend,
        "latency_ms": {
            "median": round(float(pd.Series(latencies_ms).median()), 4) if latencies_ms else 0.0,
            "p95": round(float(pd.Series(latencies_ms).quantile(0.95)), 4) if latencies_ms else 0.0,
        },
        "label_mapping": {
            "sentiment": sentiment_mapping if task in {"sentiment", "joint"} else None,
            "sarcasm": sarcasm_mapping if task in {"sarcasm", "joint"} else None,
        },
        "limitations": [],
    }
    if getattr(args, "dataset_type", "unknown") == "synthetic":
        report["limitations"].append(
            "Synthetic results do not establish generalization to real-world customer feedback."
        )
    if task == "sentiment" and sentiment_scope == "binary":
        report["limitations"].append(
            "This is positive/negative-only sentiment evaluation; neutral examples were excluded."
        )

    if task in {"sentiment", "joint"}:
        report["sentiment"] = _task_metrics(
            labeled[args.sentiment_col],
            predictions["sentiment_label"],
            VALID_SENTIMENT_LABELS if sentiment_scope == "three_class" else frozenset({"positive", "negative"}),
        )
    if task in {"sarcasm", "joint"}:
        report["sarcasm"] = _task_metrics(
            labeled[args.sarcasm_col],
            predictions["sarcasm_label"],
            VALID_SARCASM_LABELS,
        )
    return report


if __name__ == "__main__":
    arguments = parse_args()
    result = evaluate(arguments)
    output = json.dumps(result, indent=2)
    print(output)
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(output + "\n", encoding="utf-8")
