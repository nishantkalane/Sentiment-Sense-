from __future__ import annotations

import argparse
import random
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate large synthetic sentiment+sarcasm dataset")
    parser.add_argument("--rows", type=int, default=5000, help="Number of rows to generate")
    parser.add_argument("--output", type=str, default="data/big_sample_data.csv", help="Output CSV path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def build_templates() -> dict:
    return {
        "positive": [
            "Loved the new {product} release, smooth and fast experience.",
            "Customer support resolved my {issue} quickly, great service.",
            "The {feature} is amazing and saved us hours this week.",
            "Very happy with {brand}, performance has improved a lot.",
            "Great update from {brand}; onboarding is now super clear.",
        ],
        "negative": [
            "The {product} keeps failing during {task}, very frustrating.",
            "Response time for {issue} was too slow and disappointing.",
            "After the update, {feature} became unstable and buggy.",
            "I regret choosing {brand}; too many outages this month.",
            "Poor reliability in {module}; this impacts daily operations.",
        ],
        "neutral": [
            "Used {product} for {task}; overall behavior was as expected.",
            "The latest {feature} update is available in our region now.",
            "Reviewed {module} metrics today and shared them with the team.",
            "We tested {brand} for two days; results were mixed but acceptable.",
            "There is a scheduled maintenance window for {product} tonight.",
        ],
        "sarcastic": [
            "Yeah right, {brand} is totally perfect when it crashes every hour.",
            "Awesome... another {issue} just what we needed today.",
            "Sure, waiting forever for {feature} to load is my favorite part.",
            "Great job, {product} broke exactly before our demo, amazing timing.",
            "As if this {module} latency is acceptable for production.",
        ],
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    products = ["dashboard", "mobile app", "analytics engine", "checkout flow", "chat assistant"]
    brands = ["SentiCorp", "PulseAI", "Insightly", "DataNest", "TrendOps"]
    issues = ["billing ticket", "login error", "sync failure", "API timeout", "report export"]
    features = ["trend view", "batch upload", "alerting", "sarcasm detector", "live feed"]
    modules = ["recommendation module", "notification module", "auth module", "NLP module", "reporting module"]
    tasks = ["daily reporting", "customer triage", "market analysis", "feedback review", "campaign monitoring"]

    sources = ["twitter", "reviews", "support", "survey", "forum"]
    domains = ["ecommerce", "saas", "fintech", "healthcare", "education", "travel"]

    templates = build_templates()
    label_pool = ["positive", "negative", "neutral", "sarcastic"]
    weights = [0.35, 0.30, 0.25, 0.10]

    now = datetime.utcnow()
    start = now - timedelta(days=120)

    rows = []
    for idx in range(args.rows):
        label = random.choices(label_pool, weights=weights, k=1)[0]
        text_template = random.choice(templates[label])

        text = text_template.format(
            product=random.choice(products),
            brand=random.choice(brands),
            issue=random.choice(issues),
            feature=random.choice(features),
            module=random.choice(modules),
            task=random.choice(tasks),
        )

        sarcasm_label = "sarcastic" if label == "sarcastic" else "not_sarcastic"
        sentiment_label = "negative" if label == "sarcastic" else label

        event_time = start + timedelta(minutes=random.randint(0, 120 * 24 * 60))

        rows.append(
            {
                "id": idx + 1,
                "timestamp": event_time.isoformat(timespec="seconds"),
                "source": random.choice(sources),
                "domain": random.choice(domains),
                "region": random.choice(["IN", "US", "UK", "EU", "APAC"]),
                "text": text,
                "sentiment_label": sentiment_label,
                "sarcasm_label": sarcasm_label,
            }
        )

    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} rows at: {out_path.resolve()}")


if __name__ == "__main__":
    main()
