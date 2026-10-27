from __future__ import annotations

import math
import io
from datetime import date
import pandas as pd
import pytest
import sys
from types import SimpleNamespace

from sentimentsense.analytics import (
    create_time_series,
    extract_representative_examples,
    summarize_predictions,
)
from sentimentsense.chatbot import InsightChatbot
from sentimentsense.dataset import load_labeled_csv, validate_labeled_dataframe
from sentimentsense.engine import SentimentEngine
from scripts.evaluate_models import evaluate
from scripts import train_models
from scripts.train_models import validate_stratified_split, validate_training_data
from app import (
    ANALYSIS_MODES,
    _analyze_dataframe_for_ui,
    _analyze_text_for_ui,
    analysis_action_targets,
    apply_pending_navigation,
    clear_latest_analysis_state,
    confidence_caption,
    explain_single_prediction,
    final_label_caption,
    gemini_readiness_message,
    get_active_chat_history,
    prepare_sentiment_chart_data,
    render_sidebar,
    render_footer,
    render_header,
    read_uploaded_csv,
    request_navigation,
    serialize_analysis_results_csv,
    split_live_feed_lines,
    start_analysis_conversation,
    set_latest_analytics_context,
    workspace_label,
)


def test_engine_analyzes_positive_feedback_without_hosted_models(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SENTSENSE_ENABLE_HOSTED_MODELS", "0")
    prediction = SentimentEngine().analyze_text("Loved the smooth and fast dashboard experience")

    assert prediction.sentiment_label == "positive"
    assert prediction.sarcasm_label == "not_sarcastic"
    assert 0 <= prediction.sentiment_confidence <= 1


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ({"label": "NEGATIVE", "score": 0.91}, ("negative", 0.91, -0.91)),
        ({"label": "positive", "score": 0.84}, ("positive", 0.84, 0.84)),
        ({"label": "neutral", "score": 0.73}, ("neutral", 0.73, 0.0)),
    ],
)
def test_sentiment_normalizer_accepts_recognized_labels(output, expected) -> None:
    assert SentimentEngine._normalize_sentiment_output(output) == expected


@pytest.mark.parametrize("score", [0.0, 1.0, -1.0])
def test_sentiment_normalizer_accepts_finite_boundary_scores(score: float) -> None:
    assert SentimentEngine._normalize_sentiment_output(
        {"label": "positive", "score": score}
    ) == ("positive", score, score)


@pytest.mark.parametrize("score", [math.nan, math.inf, -math.inf])
def test_sentiment_normalizer_rejects_non_finite_scores(score: float) -> None:
    with pytest.raises(ValueError, match="non-finite score"):
        SentimentEngine._normalize_sentiment_output(
            {"label": "positive", "score": score}
        )


@pytest.mark.parametrize("score", [0.2, 0.95])
def test_sentiment_normalizer_rejects_unknown_labels(score: float) -> None:
    with pytest.raises(ValueError, match="Unrecognized sentiment transformer label"):
        SentimentEngine._normalize_sentiment_output({"label": "LABEL_0", "score": score})


@pytest.mark.parametrize(
    "output",
    [
        {},
        {"label": "positive"},
        {"score": 0.8},
        {"label": "positive", "score": "not-a-score"},
        None,
    ],
)
def test_sentiment_normalizer_rejects_malformed_outputs(output) -> None:
    with pytest.raises(ValueError, match="Sentiment transformer output"):
        SentimentEngine._normalize_sentiment_output(output)


def test_unknown_sentiment_output_does_not_create_prediction_or_change_sarcasm() -> None:
    class UnknownSentimentPipeline:
        def __call__(self, _text: str, **_kwargs):
            return [{"label": "LABEL_0", "score": 0.2}]

    class SarcasmPipeline:
        def __call__(self, _text: str, **_kwargs):
            return [{"label": "irony", "score": 0.9}]

    engine = SentimentEngine()
    engine._sentiment_pipe = UnknownSentimentPipeline()
    engine._sarcasm_pipe = SarcasmPipeline()
    engine._models_initialized = True

    with pytest.raises(ValueError, match="Unrecognized sentiment transformer label"):
        engine.analyze_text("This text has an unsupported model label.")

    assert engine._predict_sarcasm("This text has an unsupported model label.") == (
        "sarcastic",
        0.9,
    )


def test_ui_text_inference_reports_failure_without_returning_prediction(monkeypatch) -> None:
    messages: list[str] = []
    monkeypatch.setattr("app.st.error", messages.append)

    class FailingEngine:
        def analyze_text(self, _text: str):
            raise ValueError("unsupported sentiment transformer label")

    assert _analyze_text_for_ui(FailingEngine(), "message") is None
    assert len(messages) == 1
    assert "Text analysis failed" in messages[0]
    assert "No results were generated" in messages[0]
    assert "unsupported sentiment transformer label" not in messages[0]


def test_inference_error_does_not_expose_sensitive_exception_details(monkeypatch) -> None:
    messages: list[str] = []
    monkeypatch.setattr("app.st.error", messages.append)

    class FailingEngine:
        def analyze_text(self, _text: str):
            raise RuntimeError("token=FAKE_SECRET path=C:\\private\\model.bin")

    assert _analyze_text_for_ui(FailingEngine(), "message") is None
    assert messages
    assert "FAKE_SECRET" not in messages[0]
    assert "C:\\private\\model.bin" not in messages[0]
    assert "try again" in messages[0]


def test_ui_batch_and_live_inference_report_failures_without_results(monkeypatch) -> None:
    messages: list[str] = []
    monkeypatch.setattr("app.st.error", messages.append)

    class FailingEngine:
        def analyze_dataframe(self, _df: pd.DataFrame, text_col: str):
            raise ValueError(f"invalid output for {text_col}")

    frame = pd.DataFrame({"text": ["message"]})
    assert _analyze_dataframe_for_ui(FailingEngine(), frame, "text", "Batch analysis") is None
    assert _analyze_dataframe_for_ui(FailingEngine(), frame, "text", "Live-feed analysis") is None

    assert len(messages) == 2
    assert "Batch analysis failed" in messages[0]
    assert "Live-feed analysis failed" in messages[1]
    assert all("No results were generated" in message for message in messages)


def test_ui_inference_helpers_preserve_successful_results() -> None:
    prediction = SimpleNamespace(
        sentiment_label="positive",
        sarcasm_label="not_sarcastic",
        final_label="positive",
    )
    frame = pd.DataFrame({"text": ["message"]})

    class SuccessfulEngine:
        def analyze_text(self, _text: str):
            return prediction

        def analyze_dataframe(self, df: pd.DataFrame, text_col: str):
            return df.assign(sentiment_label="positive", sarcasm_label="not_sarcastic", final_label="positive")

    engine = SuccessfulEngine()
    assert _analyze_text_for_ui(engine, "message") is prediction
    result = _analyze_dataframe_for_ui(engine, frame, "text", "Batch analysis")
    assert result is not None
    assert result.loc[0, "sentiment_label"] == "positive"
    assert result.loc[0, "sarcasm_label"] == "not_sarcastic"
    assert result.loc[0, "final_label"] == "positive"


def test_transparency_helpers_describe_heuristic_and_model_scores() -> None:
    assert "heuristic/rule-based score" in confidence_caption("Sentiment", "heuristic fallback")
    assert "transformer model score" in confidence_caption("Sentiment", "transformer model")
    assert "not necessarily a calibrated probability" in confidence_caption(
        "Sarcasm", "irony transformer"
    )


def test_final_label_caption_explains_independent_prediction_fields() -> None:
    assert final_label_caption() == (
        "Final label is a compatibility field. Use Sentiment and Sarcasm "
        "separately to interpret the prediction; it is not a third sentiment class."
    )


@pytest.mark.parametrize(
    ("source_mode", "analysis_mode"),
    [
        ("single_text", "Single Text"),
        ("batch_csv", "Batch CSV"),
        ("live_feed", "Live Feed"),
    ],
)
def test_post_analysis_actions_target_chat_or_new_analysis(
    source_mode: str, analysis_mode: str
) -> None:
    assert analysis_action_targets(source_mode) == {
        "Explore insights": "Chat with Analytics",
        "New analysis": analysis_mode,
    }


def test_navigation_request_preserves_analysis_context() -> None:
    session_state: dict[str, object] = {
        "latest_analytics_payload": {"source_mode": "single_text"},
        "latest_analysis_df": pd.DataFrame({"text": ["current"]}),
        "active_chat_context_id": "analysis-context",
    }

    request_navigation(session_state, "Chat with Analytics")
    apply_pending_navigation(session_state)

    assert session_state["navigation_mode"] == "Chat with Analytics"
    assert session_state["latest_analytics_payload"] == {"source_mode": "single_text"}
    assert session_state["latest_analysis_df"].loc[0, "text"] == "current"
    assert session_state["active_chat_context_id"] == "analysis-context"


def test_navigation_request_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="Unknown navigation mode"):
        request_navigation({}, "Unknown")


@pytest.mark.parametrize("target_mode", list(ANALYSIS_MODES))
def test_pending_navigation_is_applied_repeatedly_without_context_loss(
    target_mode: str,
) -> None:
    session_state: dict[str, object] = {
        "navigation_mode": "Single Text",
        "latest_analytics_payload": {"source_mode": "batch_csv"},
        "active_chat_context_id": "batch-context",
    }

    request_navigation(session_state, target_mode)
    apply_pending_navigation(session_state)
    assert session_state["navigation_mode"] == target_mode
    assert session_state["latest_analytics_payload"] == {"source_mode": "batch_csv"}
    assert session_state["active_chat_context_id"] == "batch-context"

    request_navigation(session_state, "Chat with Analytics")
    apply_pending_navigation(session_state)
    assert session_state["navigation_mode"] == "Chat with Analytics"
    assert "pending_navigation_mode" not in session_state


def test_sidebar_navigation_uses_one_state_key_for_repeated_mode_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeSidebar:
        def header(self, _text: str) -> None:
            pass

        def caption(self, _text: str) -> None:
            pass

        def radio(self, _label: str, options: list[str], index: int, **_kwargs) -> str:
            return options[index]

        def markdown(self, _text: str) -> None:
            pass

        def expander(self, _label: str, **_kwargs):
            return self

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            pass

    class FakeEngine:
        load_errors: list[str] = []
        sentiment_backend = "heuristic fallback"
        sarcasm_backend = "heuristic fallback"
        inference_backend = "offline heuristic"

    session_state: dict[str, object] = {
        "navigation_mode": "Single Text",
        "latest_analytics_payload": {"source_mode": "batch_csv"},
        "active_chat_context_id": "batch-context",
    }
    monkeypatch.setattr("app.st.session_state", session_state)
    monkeypatch.setattr("app.st.sidebar", FakeSidebar())

    assert render_sidebar(FakeEngine()) == "Single Text"
    session_state["navigation_mode"] = "Batch CSV"
    assert render_sidebar(FakeEngine()) == "Batch CSV"
    session_state["navigation_mode"] = "Live Feed"
    assert render_sidebar(FakeEngine()) == "Live Feed"
    session_state["navigation_mode"] = "Chat with Analytics"
    assert render_sidebar(FakeEngine()) == "Chat with Analytics"
    assert session_state["latest_analytics_payload"] == {"source_mode": "batch_csv"}
    assert session_state["active_chat_context_id"] == "batch-context"


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("Single Text", "Single Text"),
        ("Batch CSV", "Batch CSV"),
        ("Live Feed", "Live Feed"),
        ("Chat with Analytics", "Chat with Analytics"),
        ("unexpected", "Workspace"),
    ],
)
def test_workspace_label_uses_existing_navigation_modes(mode: str, expected: str) -> None:
    assert workspace_label(mode) == expected


def test_shared_header_and_footer_render_required_product_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rendered: list[str] = []
    def capture(markdown: str, **_kwargs: object) -> None:
        rendered.append(markdown)

    monkeypatch.setattr("app.st.markdown", capture)

    render_header("Batch CSV")
    render_footer()

    combined = "\n".join(rendered)
    assert "SentimentSense" in combined
    assert "AI-powered sentiment analysis and text insights" in combined
    assert "Workspace / Batch CSV" in combined
    assert "Built by Nishant Kalane, Sahil Nagpure, Anubhav and Shubhankar W." in combined
    assert "Text sentiment analysis and actionable feedback insights." in combined
    assert "Built with" not in combined
    assert f"&copy; {date.today().year} SentimentSense" in combined


def test_engine_reports_task_backends_without_changing_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SENTSENSE_ENABLE_HOSTED_MODELS", "0")
    engine = SentimentEngine()

    assert engine.sentiment_backend == "heuristic fallback"
    assert engine.sarcasm_backend == "heuristic fallback"


@pytest.mark.parametrize(
    ("sentiment_loaded", "sarcasm_loaded", "expected_sentiment", "expected_sarcasm"),
    [
        (True, False, "transformer model", "heuristic fallback"),
        (False, True, "heuristic fallback", "irony transformer"),
        (False, False, "heuristic fallback", "heuristic fallback"),
    ],
)
def test_task_backend_status_matches_loaded_pipelines(
    sentiment_loaded: bool,
    sarcasm_loaded: bool,
    expected_sentiment: str,
    expected_sarcasm: str,
) -> None:
    engine = SentimentEngine()
    engine._sentiment_pipe = object() if sentiment_loaded else None
    engine._sarcasm_pipe = object() if sarcasm_loaded else None
    engine._models_initialized = True

    assert engine.sentiment_backend == expected_sentiment
    assert engine.sarcasm_backend == expected_sarcasm


def test_sidebar_initializes_before_checking_load_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeSidebar:
        def __init__(self) -> None:
            self.warning_messages: list[str] = []
            self.diagnostics: list[str] = []

        def header(self, _text: str) -> None:
            pass

        def radio(self, _label: str, options: list[str], **_kwargs) -> str:
            return options[0]

        def warning(self, message: str) -> None:
            self.warning_messages.append(message)

        def expander(self, _label: str, **_kwargs):
            return self

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            pass

        def markdown(self, _text: str) -> None:
            pass

        def write(self, _text: str) -> None:
            pass

        def caption(self, _text: str) -> None:
            pass

    class FakeEngine:
        load_errors: list[str] = []
        sentiment_backend = "heuristic fallback"
        sarcasm_backend = "heuristic fallback"

        @property
        def inference_backend(self) -> str:
            self.load_errors[:] = ["controlled initialization failure"]
            return "offline heuristic"

    sidebar = FakeSidebar()
    monkeypatch.setattr("app.st.sidebar", sidebar)
    monkeypatch.setattr("app.st.text", lambda message: sidebar.diagnostics.append(message))

    render_sidebar(FakeEngine())

    assert sidebar.warning_messages == []
    assert sidebar.diagnostics == []


def test_loaded_sarcasm_pipeline_uses_current_irony_backend_label() -> None:
    engine = SentimentEngine()
    engine._sarcasm_pipe = object()
    engine._models_initialized = True

    assert engine.sarcasm_backend == "irony transformer"


def test_engine_rejects_empty_text() -> None:
    with pytest.raises(ValueError, match="analyzable"):
        SentimentEngine().analyze_text("   ")


def test_engine_rejects_missing_text_column() -> None:
    with pytest.raises(ValueError, match="not found"):
        SentimentEngine().analyze_dataframe(pd.DataFrame({"message": ["hello"]}), text_col="text")


def test_batch_analysis_skips_null_empty_and_whitespace_text_without_inference() -> None:
    engine = SentimentEngine()
    frame = pd.DataFrame(
        {"text": ["Great update", None, "", "   ", "This is terrible"]}
    )

    result = engine.analyze_dataframe(frame, text_col="text")

    assert result.index.tolist() == frame.index.tolist()
    assert result["analysis_status"].tolist() == [
        "analyzed",
        "skipped_blank_text",
        "skipped_blank_text",
        "skipped_blank_text",
        "analyzed",
    ]
    assert result.loc[0, "sentiment_label"] == "positive"
    assert result.loc[4, "sentiment_label"] == "negative"
    assert pd.isna(result.loc[1, "sentiment_label"])
    assert pd.isna(result.loc[2, "final_label"])
    assert pd.isna(result.loc[3, "sarcasm_label"])


def test_batch_analysis_all_blank_rows_produces_no_predictions() -> None:
    engine = SentimentEngine()
    frame = pd.DataFrame({"text": [None, "", "  "]})

    result = engine.analyze_dataframe(frame, text_col="text")

    assert result["analysis_status"].tolist() == [
        "skipped_blank_text",
        "skipped_blank_text",
        "skipped_blank_text",
    ]
    assert result["sentiment_label"].isna().all()
    assert result["sarcasm_label"].isna().all()
    assert result["final_label"].isna().all()


def test_all_invalid_batch_clears_stale_analysis_state_and_preserves_rows() -> None:
    session_state = {
        "latest_analytics_payload": {"aggregate": {"total": 3}},
        "latest_analysis_df": pd.DataFrame({"text": ["previous"]}),
        "latest_analysis_meta": {"source_mode": "batch_csv"},
    }
    frame = pd.DataFrame({"text": [None, "", "  "]}, index=[7, 7, 9])
    result = SentimentEngine().analyze_dataframe(frame, text_col="text")

    clear_latest_analysis_state(session_state)

    assert session_state == {}
    assert result["text"].tolist() == [None, "", "  "]
    assert result.index.tolist() == [7, 7, 9]
    assert result["analysis_status"].tolist() == [
        "skipped_blank_text",
        "skipped_blank_text",
        "skipped_blank_text",
    ]


def test_mixed_batch_csv_export_preserves_rows_status_and_blank_predictions() -> None:
    frame = pd.DataFrame(
        {
            "text": ["Great", None, "  "],
            "sentiment_label": ["positive", pd.NA, pd.NA],
            "sentiment_confidence": [0.9, pd.NA, pd.NA],
            "sarcasm_label": ["not_sarcastic", pd.NA, pd.NA],
            "final_label": ["positive", pd.NA, pd.NA],
            "analysis_status": ["analyzed", "skipped_blank_text", "skipped_blank_text"],
        },
        index=[4, 4, 9],
    )

    csv_text = serialize_analysis_results_csv(frame)
    rows = csv_text.splitlines()

    assert rows[0].split(",") == [
        "text",
        "sentiment_label",
        "sentiment_confidence",
        "sarcasm_label",
        "final_label",
        "analysis_status",
    ]
    assert len(rows) == len(frame) + 1
    assert rows[1].split(",") == ["Great", "positive", "0.9", "not_sarcastic", "positive", "analyzed"]
    assert rows[2].endswith(",,,,skipped_blank_text")
    assert rows[3].endswith(",,,,skipped_blank_text")
    assert '"None"' not in csv_text
    assert "NA" not in csv_text
    assert "nan" not in csv_text


@pytest.mark.parametrize(
    ("raw", "expected_rows", "expected_blank_count"),
    [
        ("\nFirst\nSecond", ["First", "Second"], 1),
        ("First\nSecond\n", ["First", "Second"], 0),
        ("First\n\n \nSecond", ["First", "Second"], 2),
        (" \n\t\n", [], 2),
    ],
)
def test_split_live_feed_lines_filters_blanks_and_preserves_order(
    raw: str, expected_rows: list[str], expected_blank_count: int
) -> None:
    assert split_live_feed_lines(raw) == (expected_rows, expected_blank_count)


def test_all_blank_live_feed_clears_stale_analysis_state() -> None:
    session_state = {
        "latest_analytics_payload": {"aggregate": {"total": 2}},
        "latest_analysis_df": pd.DataFrame({"text": ["previous"]}),
        "latest_analysis_meta": {"source_mode": "live_feed"},
    }
    rows, blank_count = split_live_feed_lines("\n  \n\t")

    assert rows == []
    assert blank_count == 3
    clear_latest_analysis_state(session_state)
    assert session_state == {}


def test_nonblank_live_feed_input_does_not_trigger_blank_state_cleanup() -> None:
    session_state = {
        "latest_analytics_payload": {"aggregate": {"total": 1}},
        "latest_analysis_df": pd.DataFrame({"text": ["new result"]}),
        "latest_analysis_meta": {"source_mode": "live_feed"},
    }

    rows, blank_count = split_live_feed_lines("\nNew message\n  ")

    assert rows == ["New message"]
    assert blank_count == 2
    assert session_state["latest_analytics_payload"]["aggregate"]["total"] == 1
    assert session_state["latest_analysis_df"].loc[0, "text"] == "new result"
    assert session_state["latest_analysis_meta"]["source_mode"] == "live_feed"


def test_zero_byte_uploaded_csv_reports_user_facing_read_error() -> None:
    with pytest.raises(ValueError, match="Could not read the CSV"):
        read_uploaded_csv(io.StringIO(""))


def test_header_only_uploaded_csv_returns_empty_dataframe() -> None:
    result = read_uploaded_csv(io.StringIO("text,sentiment_label,sarcasm_label\n"))

    assert result.empty


def test_malformed_uploaded_csv_reports_user_facing_read_error() -> None:
    with pytest.raises(ValueError, match="Could not read the CSV"):
        read_uploaded_csv(io.StringIO('text,sentiment_label\n"unterminated\n'))


def test_gemini_initialization_error_is_generic(monkeypatch: pytest.MonkeyPatch) -> None:
    class FailingGeminiModule:
        @staticmethod
        def configure(**_kwargs):
            raise RuntimeError("distinctive provider secret")

    monkeypatch.setenv("GEMINI_API_KEY", "controlled-test-key")
    monkeypatch.setitem(sys.modules, "google.generativeai", FailingGeminiModule)

    from sentimentsense.services.gemini_service import GeminiService

    service = GeminiService()

    assert service.is_available is False
    assert service.init_error == (
        "Gemini could not be initialized. Local insight fallback is active."
    )
    assert "distinctive provider secret" not in service.init_error


def test_analytics_handles_empty_and_timestamped_frames() -> None:
    empty_summary = summarize_predictions(pd.DataFrame())
    assert empty_summary["total"] == 0

    frame = pd.DataFrame(
        {
            "timestamp": ["2026-01-01", "2026-01-02"],
            "sentiment_score": [0.8, -0.5],
        }
    )
    series = create_time_series(frame, "timestamp")
    assert len(series) == 2
    assert "sentiment_score" in series.columns


def test_representative_examples_group_by_sentiment_not_sarcasm() -> None:
    frame = pd.DataFrame(
        {
            "text": ["Sarcastic negative feedback"],
            "sentiment_label": ["negative"],
            "sarcasm_label": ["sarcastic"],
            "final_label": ["sarcastic"],
            "sentiment_confidence": [0.8],
            "sarcasm_confidence": [0.9],
        }
    )

    examples = extract_representative_examples(frame)

    assert set(examples) == {"negative"}
    assert "sarcastic" not in examples


def test_chatbot_fallback_reports_sentiment_and_sarcasm_separately() -> None:
    class UnavailableGemini:
        is_available = False
        init_error = None

    chatbot = InsightChatbot(gemini_service=UnavailableGemini())
    response = chatbot.ask_analytics(
        "What needs attention?",
        {
            "prediction": {
                "sentiment_label": "negative",
                "sarcasm_label": "sarcastic",
                "final_label": "sarcastic",
                "sentiment_confidence": 0.8,
                "sarcasm_confidence": 0.9,
            },
            "aggregate": {},
        },
    )

    assert "**Sentiment:** negative" in response
    assert "**Sarcasm flag:** sarcastic" in response
    assert "**Final label:**" not in response


def test_gemini_readiness_message_does_not_claim_request_health() -> None:
    message = gemini_readiness_message()

    assert "configured" in message
    assert "ready to attempt requests" in message
    assert "may still use the local fallback" in message
    assert "active" not in message.lower()


def test_single_record_fallback_separates_evidence_guidance_and_uncertainty() -> None:
    class UnavailableGemini:
        is_available = False
        init_error = None

    chatbot = InsightChatbot(gemini_service=UnavailableGemini())
    response = chatbot.ask_analytics(
        "What needs attention?",
        {
            "prediction": {
                "sentiment_label": "positive",
                "sarcasm_label": "not_sarcastic",
                "sentiment_confidence": 0.8,
                "sarcasm_confidence": 0.7,
            },
            "aggregate": {
                "total": 1,
                "sentiment_counts": {"positive": 1},
                "sarcasm_counts": {"not_sarcastic": 1},
            },
        },
    )

    assert "### Observed Evidence" in response
    assert "### Interpretation" in response
    assert "### Recommended Actions (General Guidance)" in response
    assert "### Missing Information and Uncertainty" in response
    assert "Only one analyzed record is available" in response
    assert "No time trend available in current context." in response
    assert "population-level conclusion" in response


def test_chatbot_api_failure_uses_local_fallback_without_raw_exception() -> None:
    class FailingGemini:
        is_available = True
        init_error = None

        def generate_analytics_response(self, **_kwargs):
            raise RuntimeError("secret API response details")

    chatbot = InsightChatbot(gemini_service=FailingGemini())
    response = chatbot.ask_analytics(
        "What needs attention?",
        {
            "aggregate": {
                "sentiment_counts": {"negative": 2},
                "sarcasm_counts": {"sarcastic": 1},
            }
        },
    )

    assert response.startswith(
        "Gemini is temporarily unavailable. Returning local insight fallback."
    )
    assert "secret API response details" not in response
    assert "### Insight Summary" in response
    assert "### Recommended Actions" in response


def test_chatbot_successful_gemini_response_is_returned_unchanged() -> None:
    class SuccessfulGemini:
        is_available = True
        init_error = None

        def generate_analytics_response(self, **_kwargs):
            return "Gemini response"

    chatbot = InsightChatbot(gemini_service=SuccessfulGemini())

    assert chatbot.ask_analytics("Summarize this", {}) == "Gemini response"


def test_analysis_conversations_are_archived_and_isolated() -> None:
    session_state: dict[str, object] = {}

    analysis_a = start_analysis_conversation(session_state)
    history_a = get_active_chat_history(session_state)
    history_a.extend(
        [
            {"role": "user", "content": "Question A1"},
            {"role": "assistant", "content": "Answer A1"},
        ]
    )

    analysis_b = start_analysis_conversation(session_state)
    history_b = get_active_chat_history(session_state)

    assert analysis_b != analysis_a
    assert history_b == []
    assert session_state["chat_conversations"][analysis_a] == history_a
    assert session_state["active_chat_context_id"] == analysis_b

    history_b.append({"role": "user", "content": "Question B1"})
    assert session_state["chat_conversations"][analysis_a] == [
        {"role": "user", "content": "Question A1"},
        {"role": "assistant", "content": "Answer A1"},
    ]
    assert session_state["chat_conversations"][analysis_b] == [
        {"role": "user", "content": "Question B1"}
    ]


def test_same_analysis_reuses_active_conversation_and_archives_mode_switches() -> None:
    session_state: dict[str, object] = {}

    first_id = start_analysis_conversation(session_state)
    get_active_chat_history(session_state).append({"role": "user", "content": "Current"})
    assert get_active_chat_history(session_state) == [
        {"role": "user", "content": "Current"}
    ]

    session_state["chat_history"] = get_active_chat_history(session_state)
    assert session_state["active_chat_context_id"] == first_id

    batch_id = start_analysis_conversation(session_state)
    live_id = start_analysis_conversation(session_state)

    assert first_id != batch_id != live_id
    assert session_state["chat_conversations"][first_id] == [
        {"role": "user", "content": "Current"}
    ]
    assert session_state["chat_conversations"][batch_id] == []
    assert get_active_chat_history(session_state) == []


def test_single_prediction_passes_independent_labels_to_chatbot() -> None:
    class FakeChatbot:
        def explain(self, **kwargs):
            return kwargs

    prediction = SimpleNamespace(
        sentiment_label="negative",
        sarcasm_label="sarcastic",
        sentiment_confidence=0.8,
        final_label="sarcastic",
    )

    result = explain_single_prediction(FakeChatbot(), "Sure, this is perfect.", prediction)

    assert result["sentiment_label"] == "negative"
    assert result["sarcasm_label"] == "sarcastic"


def test_sentiment_chart_data_ignores_mixed_final_label() -> None:
    frame = pd.DataFrame(
        {
            "sentiment_label": ["negative"],
            "sarcasm_label": ["sarcastic"],
            "final_label": ["sarcastic"],
        }
    )

    chart_data = prepare_sentiment_chart_data(frame)

    assert chart_data["sentiment_label"].tolist() == ["negative"]
    assert "sarcastic" not in set(chart_data["sentiment_label"])


def test_training_sample_has_expected_schema_and_labels() -> None:
    frame = load_labeled_csv("data/train_sample.csv")

    assert list(frame.columns) == ["text", "sentiment_label", "sarcasm_label"]
    assert len(frame) == 6
    assert frame["text"].str.strip().ne("").all()
    assert set(frame["sentiment_label"]) <= {"positive", "negative", "neutral"}
    assert set(frame["sarcasm_label"]) <= {"sarcastic", "not_sarcastic"}
    assert "Yeah right, this is the best service ever..." in set(frame["text"])
    assert "Sure, because waiting 2 hours is amazing" in set(frame["text"])


def test_training_sample_is_rejected_as_too_small_for_stratified_training() -> None:
    frame = load_labeled_csv("data/train_sample.csv")

    with pytest.raises(ValueError, match="Cannot create a stratified split"):
        validate_training_data(frame, test_size=0.2)


def test_stratified_split_policy_requires_each_class_in_both_splits() -> None:
    frame = pd.DataFrame(
        {
            "text": [f"Example {index}" for index in range(6)],
            "sentiment_label": ["positive", "positive", "negative", "negative", "neutral", "neutral"],
            "sarcasm_label": ["sarcastic", "sarcastic", "not_sarcastic", "not_sarcastic", "not_sarcastic", "not_sarcastic"],
        }
    )

    validate_stratified_split(frame, "sentiment_label", test_size=0.5)

    with pytest.raises(ValueError, match="enough rows for each class"):
        validate_stratified_split(frame, "sentiment_label", test_size=0.2)


def test_training_help_does_not_import_optional_training_dependencies() -> None:
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "scripts.train_models", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Train SentimentSense" in result.stdout


@pytest.mark.parametrize(
    ("data_path", "expected_message"),
    [
        ("missing.csv", "Dataset not found"),
    ],
)
def test_training_cli_reports_missing_dataset_without_traceback(
    tmp_path, monkeypatch, capsys, data_path: str, expected_message: str
) -> None:
    monkeypatch.setattr(train_models, "require_training_dependencies", lambda: None)
    monkeypatch.setattr(
        "sys.argv",
        [
            "train_models",
            "--data",
            str(tmp_path / data_path),
            "--output-dir",
            str(tmp_path / "models"),
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        train_models.main()

    captured = capsys.readouterr()
    assert exc_info.value.code == 1
    assert expected_message in captured.err
    assert "Traceback" not in captured.err
    assert not (tmp_path / "models").exists()


def test_training_cli_reports_invalid_data_without_traceback(
    tmp_path, monkeypatch, capsys
) -> None:
    data = tmp_path / "invalid.csv"
    data.write_text(
        "text,sentiment_label,sarcasm_label\n"
        "Only one row,positive,not_sarcastic\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(train_models, "require_training_dependencies", lambda: None)
    monkeypatch.setattr(
        "sys.argv",
        [
            "train_models",
            "--data",
            str(data),
            "--output-dir",
            str(tmp_path / "models"),
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        train_models.main()

    captured = capsys.readouterr()
    assert exc_info.value.code == 1
    assert "Cannot create a stratified split" in captured.err
    assert "Traceback" not in captured.err
    assert not (tmp_path / "models").exists()


def test_training_cli_reports_missing_dependencies_without_traceback(
    tmp_path, monkeypatch, capsys
) -> None:
    data = tmp_path / "valid-enough.csv"
    data.write_text(
        "text,sentiment_label,sarcasm_label\n"
        "Good one,positive,not_sarcastic\n"
        "Great one,positive,sarcastic\n"
        "Bad one,negative,not_sarcastic\n"
        "Awful one,negative,sarcastic\n"
        "Okay one,neutral,not_sarcastic\n"
        "Fine one,neutral,sarcastic\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        train_models,
        "require_training_dependencies",
        lambda: (_ for _ in ()).throw(
            RuntimeError("Training dependencies are missing.")
        ),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "train_models",
            "--data",
            str(data),
            "--output-dir",
            str(tmp_path / "models"),
            "--test-size",
            "0.5",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        train_models.main()

    captured = capsys.readouterr()
    assert exc_info.value.code == 1
    assert "Training dependencies are missing." in captured.err
    assert "Traceback" not in captured.err
    assert not (tmp_path / "models").exists()


def test_labeled_csv_rejects_missing_values_and_invalid_labels() -> None:
    frame = pd.DataFrame(
        {
            "text": ["Valid text", ""],
            "sentiment_label": ["positive", "unknown"],
            "sarcasm_label": ["not_sarcastic", "not_sarcastic"],
        }
    )

    with pytest.raises(ValueError, match="empty"):
        validate_labeled_dataframe(frame)

    frame.loc[1, "text"] = "Another text"
    with pytest.raises(ValueError, match="Invalid sentiment_label"):
        validate_labeled_dataframe(frame)


def test_labeled_csv_rejects_missing_columns_and_malformed_csv(tmp_path) -> None:
    missing_column = tmp_path / "missing.csv"
    missing_column.write_text("text,sentiment_label\nhello,positive\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Missing required columns"):
        load_labeled_csv(missing_column)

    malformed = tmp_path / "malformed.csv"
    malformed.write_text(
        "text,sentiment_label,sarcasm_label\n"
        "This, text,positive,not_sarcastic\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Malformed CSV"):
        load_labeled_csv(malformed)


def test_task_specific_validation_supports_sentiment_sarcasm_and_joint(tmp_path) -> None:
    sentiment = tmp_path / "sentiment.csv"
    sentiment.write_text(
        "message,polarity\nGood update,positive\n", encoding="utf-8"
    )
    sarcasm = tmp_path / "sarcasm.csv"
    sarcasm.write_text(
        "message,ironic\nYeah right,sarcastic\n", encoding="utf-8"
    )
    joint = tmp_path / "joint.csv"
    joint.write_text(
        "text,sentiment_label,sarcasm_label\nGood update,positive,not_sarcastic\n",
        encoding="utf-8",
    )

    assert len(
        load_labeled_csv(
            sentiment,
            text_column="message",
            sentiment_column="polarity",
            task="sentiment",
            sentiment_label_mapping={"positive": "positive"},
        )
    ) == 1
    assert len(
        load_labeled_csv(
            sarcasm,
            text_column="message",
            sarcasm_column="ironic",
            task="sarcasm",
            sarcasm_label_mapping={"sarcastic": "sarcastic"},
        )
    ) == 1
    assert len(load_labeled_csv(joint, task="joint")) == 1

    with pytest.raises(ValueError, match="Missing required columns"):
        load_labeled_csv(sentiment, text_column="message", task="joint")


def test_evaluation_rejects_incomplete_and_ambiguous_mappings(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SENTSENSE_ENABLE_HOSTED_MODELS", "0")
    data = tmp_path / "source_labels.csv"
    data.write_text(
        "text,polarity\nGood update,pos\n", encoding="utf-8"
    )
    base = {
        "data": data,
        "task": "sentiment",
        "text_col": "text",
        "sentiment_col": "polarity",
        "sarcasm_col": "sarcasm_label",
        "sentiment_scope": "three_class",
        "dataset_name": "fixture",
        "source_url": "",
        "dataset_type": "unknown",
        "split": "test",
        "output": None,
    }

    args = type("EvaluationArgs", (), {**base, "sentiment_label_map": '{"pos": "positive"}'})()
    report = evaluate(args)
    assert report["task"] == "sentiment"
    assert report["label_mapping"]["sentiment"] == {"pos": "positive"}

    args.sentiment_label_map = '{"pos": "positive", "good": "positive"}'
    with pytest.raises(ValueError, match="ambiguous"):
        evaluate(args)

    args.sentiment_label_map = '{"pos": "positive"}'
    data.write_text("text,polarity\nGood update,positive\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid polarity"):
        evaluate(args)


def test_binary_sentiment_scope_excludes_neutral_and_reports_scope(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SENTSENSE_ENABLE_HOSTED_MODELS", "0")
    data = tmp_path / "three_class.csv"
    data.write_text(
        "text,sentiment_label\n"
        "Good update,positive\n"
        "Average update,neutral\n"
        "Bad update,negative\n",
        encoding="utf-8",
    )
    args = type(
        "EvaluationArgs",
        (),
        {
            "data": data,
            "task": "sentiment",
            "text_col": "text",
            "sentiment_col": "sentiment_label",
            "sarcasm_col": "sarcasm_label",
            "sentiment_scope": "binary",
            "sentiment_label_map": None,
            "sarcasm_label_map": None,
            "dataset_name": "fixture",
            "source_url": "",
            "dataset_type": "unknown",
            "split": "test",
            "output": None,
        },
    )()

    report = evaluate(args)
    assert report["rows"] == 2
    assert report["excluded_rows"] == 1
    assert report["excluded_reason"]
    assert report["sentiment"]["labels"] == ["negative", "positive"]


def test_task_reports_use_requested_outputs_not_final_label(tmp_path) -> None:
    data = tmp_path / "joint.csv"
    data.write_text(
        "text,sentiment_label,sarcasm_label\n"
        "Good update,positive,not_sarcastic\n",
        encoding="utf-8",
    )

    class FakeEngine:
        inference_backend = "test-double"
        sentiment_backend = "heuristic fallback"
        sarcasm_backend = "heuristic fallback"

        def analyze_text(self, text: str) -> SimpleNamespace:
            return SimpleNamespace(
                sentiment_label="positive",
                sarcasm_label="not_sarcastic",
                final_label="negative",
            )

    args = type(
        "EvaluationArgs",
        (),
        {
            "data": data,
            "task": "joint",
            "text_col": "text",
            "sentiment_col": "sentiment_label",
            "sarcasm_col": "sarcasm_label",
            "sentiment_scope": "three_class",
            "sentiment_label_map": None,
            "sarcasm_label_map": None,
            "dataset_name": "fixture",
            "source_url": "",
            "dataset_type": "unknown",
            "split": "test",
            "output": None,
        },
    )()

    report = evaluate(args, engine=FakeEngine())
    assert report["sentiment"]["accuracy"] == 1.0
    assert report["sarcasm"]["accuracy"] == 1.0


@pytest.mark.parametrize(
    ("inference_backend", "sentiment_backend", "sarcasm_backend"),
    [
        ("offline heuristic", "heuristic fallback", "heuristic fallback"),
        ("transformers", "transformer model", "irony transformer"),
        ("transformers + heuristic fallback", "transformer model", "heuristic fallback"),
        ("transformers + heuristic fallback", "heuristic fallback", "irony transformer"),
    ],
)
def test_evaluation_report_records_task_specific_backends(
    tmp_path,
    inference_backend: str,
    sentiment_backend: str,
    sarcasm_backend: str,
) -> None:
    data = tmp_path / "joint.csv"
    data.write_text(
        "text,sentiment_label,sarcasm_label\n"
        "Good update,positive,not_sarcastic\n",
        encoding="utf-8",
    )

    class FakeEngine:
        def analyze_text(self, text: str) -> SimpleNamespace:
            return SimpleNamespace(
                sentiment_label="positive",
                sarcasm_label="not_sarcastic",
                final_label="positive",
            )

    engine = FakeEngine()
    engine.inference_backend = inference_backend
    engine.sentiment_backend = sentiment_backend
    engine.sarcasm_backend = sarcasm_backend

    args = type(
        "EvaluationArgs",
        (),
        {
            "data": data,
            "task": "joint",
            "text_col": "text",
            "sentiment_col": "sentiment_label",
            "sarcasm_col": "sarcasm_label",
            "sentiment_scope": "three_class",
            "sentiment_label_map": None,
            "sarcasm_label_map": None,
            "dataset_name": "fixture",
            "source_url": "",
            "dataset_type": "unknown",
            "split": "test",
            "output": None,
        },
    )()

    report = evaluate(args, engine=engine)

    assert report["backend"] == inference_backend
    assert report["sentiment_backend"] == sentiment_backend
    assert report["sarcasm_backend"] == sarcasm_backend


def test_batch_output_preserves_independent_labels_and_final_label() -> None:
    engine = SentimentEngine()
    result = engine.analyze_dataframe(
        pd.DataFrame({"text": ["Great job, waiting forever is my favorite part."]}),
        text_col="text",
    )

    assert result.loc[0, "sentiment_label"] == "positive"
    assert result.loc[0, "sarcasm_label"] == "sarcastic"
    assert result.loc[0, "final_label"] == "sarcastic"
    assert {"sentiment_label", "sarcasm_label", "final_label"}.issubset(result.columns)


def test_evaluation_rejects_invalid_labels_and_writes_provenance(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SENTSENSE_ENABLE_HOSTED_MODELS", "0")
    invalid = tmp_path / "invalid.csv"
    invalid.write_text(
        "text,sentiment_label,sarcasm_label\n"
        "A comment,positive,not_sarcastic\n"
        "Another comment,unknown,not_sarcastic\n",
        encoding="utf-8",
    )
    args = type(
        "EvaluationArgs",
        (),
        {
            "data": invalid,
            "text_col": "text",
            "sentiment_col": "sentiment_label",
            "sarcasm_col": "sarcasm_label",
            "task": "joint",
            "sentiment_scope": "three_class",
            "sentiment_label_map": None,
            "sarcasm_label_map": None,
            "dataset_type": "synthetic",
            "dataset_name": "",
            "source_url": "",
            "split": "",
            "output": None,
        },
    )()

    with pytest.raises(ValueError, match="Invalid sentiment_label"):
        evaluate(args)

    valid = tmp_path / "valid.csv"
    valid.write_text(
        "text,sentiment_label,sarcasm_label\n"
        "Loved the fast update,positive,not_sarcastic\n",
        encoding="utf-8",
    )
    args.data = valid
    report = evaluate(args)
    assert report["dataset_type"] == "synthetic"
    assert report["rows"] == 1
    assert report["limitations"]
