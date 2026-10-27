from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sentimentsense.dataset import (
    SARCASM_COLUMN,
    SENTIMENT_COLUMN,
    TEXT_COLUMN,
    load_labeled_csv,
    validate_labeled_dataframe,
)

TEXT_COL = TEXT_COLUMN
SENTIMENT_COL = SENTIMENT_COLUMN
SARCASM_COL = SARCASM_COLUMN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SentimentSense local sentiment and sarcasm models")
    parser.add_argument("--data", type=str, required=True, help="Path to CSV with text/sentiment/sarcasm labels")
    parser.add_argument("--base-model", type=str, default="distilbert-base-uncased")
    parser.add_argument("--output-dir", type=str, default="models")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--test-size", type=float, default=0.2)
    return parser.parse_args()


def validate_input(df: pd.DataFrame) -> None:
    validate_labeled_dataframe(df)


def validate_stratified_split(
    df: pd.DataFrame, label_col: str, test_size: float
) -> None:
    if not 0 < test_size < 1:
        raise ValueError("test_size must be greater than 0 and less than 1")

    labels = df[label_col].dropna().astype(str)
    class_counts = labels.value_counts()
    class_count = len(class_counts)
    evaluation_rows = max(1, math.ceil(len(labels) * test_size))
    training_rows = len(labels) - evaluation_rows

    if (
        class_count < 2
        or evaluation_rows < class_count
        or training_rows < class_count
        or (class_counts < 2).any()
    ):
        counts = ", ".join(
            f"{label}={count}" for label, count in class_counts.sort_index().items()
        )
        raise ValueError(
            f"Cannot create a stratified split for '{label_col}' with "
            f"test_size={test_size}: {len(labels)} rows, {class_count} classes "
            f"({counts}). Provide more examples for every class and enough "
            "rows for each class to appear in both train and evaluation splits."
        )


def validate_training_data(df: pd.DataFrame, test_size: float) -> None:
    validate_input(df)
    for label_col in (SENTIMENT_COL, SARCASM_COL):
        validate_stratified_split(df, label_col, test_size)


def require_training_dependencies() -> None:
    try:
        import datasets  # noqa: F401
        import sklearn  # noqa: F401
        import transformers  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Training dependencies are missing. Install the project "
            "requirements with 'python -m pip install -r requirements.txt' "
            "before running training."
        ) from exc


def encode_labels(series: pd.Series) -> tuple[pd.Series, dict[int, str], dict[str, int]]:
    labels = sorted(series.dropna().astype(str).unique().tolist())
    label2id = {name: idx for idx, name in enumerate(labels)}
    id2label = {idx: name for name, idx in label2id.items()}
    encoded = series.astype(str).map(label2id)
    return encoded, id2label, label2id


def train_one_task(
    df: pd.DataFrame,
    label_col: str,
    base_model: str,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    max_length: int,
    test_size: float,
) -> None:
    from datasets import Dataset
    from sklearn.model_selection import train_test_split
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    work_df = df[[TEXT_COL, label_col]].dropna().copy()
    work_df[TEXT_COL] = work_df[TEXT_COL].astype(str)

    encoded_labels, id2label, label2id = encode_labels(work_df[label_col])
    work_df["labels"] = encoded_labels

    train_df, eval_df = train_test_split(
        work_df,
        test_size=test_size,
        random_state=42,
        stratify=work_df["labels"],
    )

    train_dataset = Dataset.from_pandas(train_df[[TEXT_COL, "labels"]], preserve_index=False)
    eval_dataset = Dataset.from_pandas(eval_df[[TEXT_COL, "labels"]], preserve_index=False)

    tokenizer = AutoTokenizer.from_pretrained(base_model)

    def tokenize_fn(batch: dict) -> dict:
        return tokenizer(batch[TEXT_COL], truncation=True, max_length=max_length)

    train_dataset = train_dataset.map(tokenize_fn, batched=True)
    eval_dataset = eval_dataset.map(tokenize_fn, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )

    task_dir = output_dir / ("sentiment" if label_col == SENTIMENT_COL else "sarcasm")
    task_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(task_dir / "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    trainer.train()
    trainer.save_model(str(task_dir))
    tokenizer.save_pretrained(str(task_dir))


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.data)
    try:
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        raw_df = load_labeled_csv(dataset_path)
        validate_training_data(raw_df, args.test_size)
        require_training_dependencies()
    except (OSError, UnicodeError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_one_task(
        df=raw_df,
        label_col=SENTIMENT_COL,
        base_model=args.base_model,
        output_dir=out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        test_size=args.test_size,
    )
    train_one_task(
        df=raw_df,
        label_col=SARCASM_COL,
        base_model=args.base_model,
        output_dir=out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        test_size=args.test_size,
    )

    print("Training complete. Saved local models to:", out_dir.resolve())


if __name__ == "__main__":
    main()
