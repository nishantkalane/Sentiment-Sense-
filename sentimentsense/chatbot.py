from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .services import GeminiService


@dataclass
class ChatResponse:
    summary: str
    rationale: str
    action: str


class InsightChatbot:
    def __init__(self, gemini_service: GeminiService | None = None) -> None:
        self.gemini_service = gemini_service or GeminiService()

    @property
    def gemini_available(self) -> bool:
        return self.gemini_service.is_available

    @property
    def gemini_error(self) -> str | None:
        return self.gemini_service.init_error

    def explain(self, text: str, sentiment_label: str, sarcasm_label: str, confidence: float) -> ChatResponse:
        sentiment_msg = {
            "positive": "The tone appears favorable and supportive.",
            "negative": "The tone appears critical or dissatisfied.",
            "neutral": "The message appears mostly informational with limited emotion.",
            "sarcastic": "The sentence likely contains a mismatch between words and intent.",
        }.get(sentiment_label, "The tone is mixed.")

        if sarcasm_label == "sarcastic":
            rationale = (
                "Sarcasm cues were detected, so literal wording may not reflect the real sentiment. "
                "Pay attention to exaggeration, contrast, and punctuation patterns."
            )
            action = "Review this item manually before using it in decision-critical reports."
        else:
            rationale = "No strong sarcasm markers were detected; literal sentiment is likely reliable."
            if sentiment_label == "negative":
                action = "Prioritize follow-up and root-cause analysis for this feedback."
            elif sentiment_label == "positive":
                action = "Highlight this feedback as a success signal and track repeated themes."
            else:
                action = "Collect additional context before taking action."

        summary = (
            f"Input: \"{text[:180]}\"\n"
            f"Interpretation: {sentiment_msg} (score: {confidence:.2f}; "
            "not necessarily a calibrated probability)."
        )

        return ChatResponse(summary=summary, rationale=rationale, action=action)

    def ask_analytics(
        self,
        user_query: str,
        analytics_payload: dict[str, Any],
        chat_history: list[dict[str, str]] | None = None,
    ) -> str:
        if self.gemini_service.is_available:
            try:
                return self.gemini_service.generate_analytics_response(
                    user_query=user_query,
                    analytics_payload=analytics_payload,
                    chat_history=chat_history,
                )
            except Exception:
                fallback_header = "Gemini is temporarily unavailable. Returning local insight fallback."
                fallback = self._fallback_analytics_response(user_query, analytics_payload)
                return f"{fallback_header}\n\n{fallback}"

        return self._fallback_analytics_response(user_query, analytics_payload)

    def _fallback_analytics_response(self, user_query: str, analytics_payload: dict[str, Any]) -> str:
        prediction = analytics_payload.get("prediction", {})
        aggregate = analytics_payload.get("aggregate", {})
        trend = analytics_payload.get("time_series", [])
        trend_changes = analytics_payload.get("trend_changes", [])
        examples = analytics_payload.get("representative_examples", {})

        sentiment_label = prediction.get("sentiment_label", "unknown")
        sarcasm_flag = prediction.get("sarcasm_label", "unknown")
        sentiment_conf = prediction.get("sentiment_confidence", 0.0)
        sarcasm_conf = prediction.get("sarcasm_confidence", 0.0)
        total_records = aggregate.get("total")
        if total_records is None:
            total_records = sum(aggregate.get("sentiment_counts", {}).values())
        if not total_records and prediction:
            total_records = 1

        trend_note = "No time trend available in current context."
        if trend:
            first = trend[0]
            last = trend[-1]
            trend_note = (
                f"Trend window has {len(trend)} points from {first.get('timestamp')} to {last.get('timestamp')}. "
                "Use batch mode with timestamps for stronger trend interpretation."
            )

        change_note = "No major spike/drop events detected from available trend points."
        if trend_changes:
            top_change = trend_changes[0]
            change_note = (
                f"Largest {top_change.get('direction', 'change')} at {top_change.get('timestamp')} "
                f"with delta {float(top_change.get('change', 0.0)):.2f}."
            )

        example_note = "No representative text examples available."
        if examples:
            label, items = next(iter(examples.items()))
            if items:
                sample_text = str(items[0].get("text", ""))
                example_note = f"Top {label} example: \"{sample_text}\""

        if total_records == 1:
            scope_note = (
                "Only one analyzed record is available; this is an item-level signal, "
                "not a population-level conclusion."
            )
        elif total_records:
            scope_note = f"{total_records} analyzed records are available in this context."
        else:
            scope_note = "No analyzed records are available in this context."

        return (
            "### Insight Summary\n"
            f"- **Question:** {user_query}\n"
            f"- **Sentiment:** {sentiment_label}\n"
            f"- **Sarcasm flag:** {sarcasm_flag}\n"
            f"- **Sentiment score:** {sentiment_conf:.2f} (backend-dependent; not necessarily a calibrated probability)\n"
            f"- **Sarcasm score:** {sarcasm_conf:.2f} (backend-dependent; not necessarily a calibrated probability)\n\n"
            "### Observed Evidence\n"
            f"- Aggregate counts: {aggregate.get('sentiment_counts', {})}\n"
            f"- Sarcasm counts: {aggregate.get('sarcasm_counts', {})}\n"
            f"- {example_note}\n"
            f"- {scope_note}\n\n"
            "### Interpretation\n"
            f"- {trend_note}\n"
            f"- {change_note}\n"
            "\n"
            "### Recommended Actions (General Guidance)\n"
            "- Prioritize items with negative sentiment and high confidence for quick response.\n"
            "- Manually review sarcastic items before escalation decisions.\n"
            "- Track day-over-day shifts to identify campaign or product impact.\n"
            "\n"
            "### Missing Information and Uncertainty\n"
            "- Scores are backend-dependent and are not necessarily calibrated probabilities.\n"
            f"- {trend_note}\n"
        )
