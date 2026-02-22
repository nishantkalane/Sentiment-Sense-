from __future__ import annotations

import pandas as pd


def summarize_predictions(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "total": 0,
            "sentiment_counts": {},
            "sarcasm_counts": {},
            "avg_sentiment_confidence": 0.0,
            "avg_sarcasm_confidence": 0.0,
        }

    return {
        "total": len(df),
        "sentiment_counts": df["sentiment_label"].value_counts().to_dict(),
        "sarcasm_counts": df["sarcasm_label"].value_counts().to_dict(),
        "avg_sentiment_confidence": float(df["sentiment_confidence"].mean()),
        "avg_sarcasm_confidence": float(df["sarcasm_confidence"].mean()),
    }


def create_time_series(df: pd.DataFrame, time_col: str, score_col: str = "sentiment_score") -> pd.DataFrame:
    if time_col not in df.columns:
        return pd.DataFrame(columns=[time_col, score_col])

    ts = df.copy()
    ts[time_col] = pd.to_datetime(ts[time_col], errors="coerce")
    ts = ts.dropna(subset=[time_col])
    if ts.empty:
        return pd.DataFrame(columns=[time_col, score_col])

    return (
        ts.set_index(time_col)
        .resample("D")[score_col]
        .mean()
        .reset_index()
        .sort_values(time_col)
    )


def detect_trend_changes(ts_df: pd.DataFrame, max_events: int = 3) -> list[dict]:
    if ts_df.empty or "sentiment_score" not in ts_df.columns:
        return []

    ts = ts_df.copy()
    time_col = ts.columns[0]
    ts[time_col] = pd.to_datetime(ts[time_col], errors="coerce")
    ts = ts.dropna(subset=[time_col]).sort_values(time_col)
    if len(ts) < 2:
        return []

    ts["delta"] = ts["sentiment_score"].diff()
    ts = ts.dropna(subset=["delta"])
    if ts.empty:
        return []

    ts["abs_delta"] = ts["delta"].abs()
    top_changes = ts.sort_values("abs_delta", ascending=False).head(max_events)

    events: list[dict] = []
    for _, row in top_changes.iterrows():
        direction = "spike" if float(row["delta"]) > 0 else "drop"
        events.append(
            {
                "timestamp": str(row[time_col]),
                "change": float(row["delta"]),
                "direction": direction,
                "score": float(row["sentiment_score"]),
            }
        )
    return events


def extract_representative_examples(
    df: pd.DataFrame,
    text_col: str = "text",
    max_per_label: int = 2,
) -> dict:
    required = {"final_label", "sentiment_confidence", text_col}
    if not required.issubset(df.columns):
        return {}

    examples: dict[str, list[dict]] = {}
    for label, sub_df in df.groupby("final_label"):
        ranked = sub_df.sort_values("sentiment_confidence", ascending=False).head(max_per_label)
        items: list[dict] = []
        for _, row in ranked.iterrows():
            raw_text = str(row[text_col]) if pd.notna(row[text_col]) else ""
            items.append(
                {
                    "text": raw_text[:220],
                    "sentiment_confidence": float(row.get("sentiment_confidence", 0.0)),
                    "sarcasm_confidence": float(row.get("sarcasm_confidence", 0.0)),
                }
            )
        examples[str(label)] = items
    return examples
