from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

TEXT_COL = "text"
SENTIMENT_COL = "sentiment_label"
SARCASM_COL = "sarcasm_label"


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
    required_cols = {TEXT_COL, SENTIMENT_COL, SARCASM_COL}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


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


if __name__ == "__main__":
    args = parse_args()
    dataset_path = Path(args.data)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    raw_df = pd.read_csv(dataset_path)
    validate_input(raw_df)

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
