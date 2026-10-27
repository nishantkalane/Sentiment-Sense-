from __future__ import annotations

import io
import json
from html import escape
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd
import plotly.express as px
import streamlit as st

from sentimentsense.analytics import (
    create_time_series,
    detect_trend_changes,
    extract_representative_examples,
    summarize_predictions,
)
from sentimentsense.chatbot import InsightChatbot
from sentimentsense.engine import SentimentEngine


st.set_page_config(page_title="SentimentSense", page_icon="🧠", layout="wide")

ANALYSIS_MODES = ("Single Text", "Batch CSV", "Live Feed", "Chat with Analytics")


@st.cache_resource
def get_engine() -> SentimentEngine:
    return SentimentEngine()


@st.cache_resource
def get_chatbot() -> InsightChatbot:
    return InsightChatbot()


def render_app_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --ss-page: #F7F8FA;
            --ss-surface: #FFFFFF;
            --ss-heading: #17324D;
            --ss-body: #263746;
            --ss-secondary: #536477;
            --ss-border: #DCE3EA;
            --ss-primary: #1769AA;
            --ss-primary-hover: #12588F;
            --ss-success-bg: #E5F3EA;
            --ss-success-text: #1F5B39;
            --ss-error-bg: #FBE9E8;
            --ss-error-text: #8F2D2A;
            --ss-info-bg: #E8F1FB;
            --ss-info-text: #1E4F80;
            --ss-warning-bg: #FFF3D6;
            --ss-warning-text: #795515;
        }
        [data-testid="stAppViewContainer"] {
            background: var(--ss-page);
            color: var(--ss-body);
        }
        [data-testid="stMainBlockContainer"],
        [data-testid="stAppViewContainer"] .main .block-container {
            max-width: 1180px;
            padding-top: 1.25rem !important;
            padding-bottom: 2rem !important;
        }
        [data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] {
            min-height: calc(100vh - 36px);
        }
        [data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"]:has(.ss-product-footer) {
            display: flex;
            flex-direction: column;
        }
        [data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"]:has(.ss-product-footer)
            > [data-testid="stElementContainer"]:has(.ss-product-footer) {
            margin-top: auto;
        }
        [data-testid="stHeader"] {
            background: transparent;
            height: 2.5rem;
        }
        [data-testid="stToolbar"] {
            background: transparent;
        }
        [data-testid="stSidebarCollapseButton"] button[data-testid="stBaseButton-headerNoPadding"],
        [data-testid="stSidebarCollapsedControl"] button[data-testid="stBaseButton-headerNoPadding"],
        button[data-testid="stExpandSidebarButton"] {
            background-color: var(--ss-info-bg) !important;
            border: 1px solid var(--ss-primary) !important;
            border-radius: 0.4rem !important;
            box-shadow: none !important;
            color: var(--ss-heading) !important;
            min-height: 2rem;
            min-width: 2rem;
            opacity: 1 !important;
        }
        [data-testid="stSidebarCollapseButton"] button[data-testid="stBaseButton-headerNoPadding"] *,
        [data-testid="stSidebarCollapsedControl"] button[data-testid="stBaseButton-headerNoPadding"] *,
        button[data-testid="stExpandSidebarButton"] * {
            color: var(--ss-heading) !important;
            fill: currentColor !important;
            stroke: currentColor !important;
        }
        [data-testid="stSidebarCollapseButton"] button[data-testid="stBaseButton-headerNoPadding"]:hover,
        [data-testid="stSidebarCollapsedControl"] button[data-testid="stBaseButton-headerNoPadding"]:hover,
        button[data-testid="stExpandSidebarButton"]:hover {
            background-color: var(--ss-info-bg) !important;
            border-color: var(--ss-primary-hover) !important;
            color: var(--ss-primary-hover) !important;
        }
        [data-testid="stSidebarCollapseButton"] button[data-testid="stBaseButton-headerNoPadding"]:hover *,
        [data-testid="stSidebarCollapsedControl"] button[data-testid="stBaseButton-headerNoPadding"]:hover *,
        button[data-testid="stExpandSidebarButton"]:hover * {
            color: var(--ss-primary-hover) !important;
            fill: currentColor !important;
            stroke: currentColor !important;
        }
        [data-testid="stSidebarCollapseButton"] button[data-testid="stBaseButton-headerNoPadding"]:active,
        [data-testid="stSidebarCollapsedControl"] button[data-testid="stBaseButton-headerNoPadding"]:active,
        button[data-testid="stExpandSidebarButton"]:active {
            background-color: var(--ss-primary) !important;
            border-color: var(--ss-primary) !important;
            color: var(--ss-surface) !important;
        }
        [data-testid="stSidebarCollapseButton"] button[data-testid="stBaseButton-headerNoPadding"]:active *,
        [data-testid="stSidebarCollapsedControl"] button[data-testid="stBaseButton-headerNoPadding"]:active *,
        button[data-testid="stExpandSidebarButton"]:active * {
            color: var(--ss-surface) !important;
            fill: currentColor !important;
            stroke: currentColor !important;
        }
        [data-testid="stSidebarCollapseButton"] button[data-testid="stBaseButton-headerNoPadding"]:focus-visible,
        [data-testid="stSidebarCollapsedControl"] button[data-testid="stBaseButton-headerNoPadding"]:focus-visible,
        button[data-testid="stExpandSidebarButton"]:focus-visible {
            box-shadow: 0 0 0 2px var(--ss-surface), 0 0 0 4px var(--ss-primary) !important;
            outline: none !important;
        }
        [data-testid="stAppViewContainer"] h1,
        [data-testid="stAppViewContainer"] h2,
        [data-testid="stAppViewContainer"] h3,
        [data-testid="stAppViewContainer"] h4,
        [data-testid="stAppViewContainer"] h5,
        [data-testid="stAppViewContainer"] h6 {
            color: var(--ss-heading) !important;
            letter-spacing: -0.01em;
        }
        [data-testid="stAppViewContainer"] p,
        [data-testid="stAppViewContainer"] li,
        [data-testid="stAppViewContainer"] label,
        [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] {
            color: var(--ss-body);
        }
        [data-testid="stAppViewContainer"] small,
        [data-testid="stAppViewContainer"] [data-testid="stCaptionContainer"] {
            color: var(--ss-secondary) !important;
        }
        [data-testid="stSidebar"] {
            background: var(--ss-surface);
            border-right: 1px solid var(--ss-border);
        }
        [data-testid="stSidebar"] > div:first-child {
            background: var(--ss-surface);
        }
        [data-testid="stSidebarUserContent"] {
            margin-top: -28px;
        }
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
            color: var(--ss-body) !important;
        }
        [data-testid="stSidebar"] [role="radiogroup"] label {
            border-radius: 0.45rem;
            padding: 0.35rem 0.45rem;
        }
        [data-testid="stSidebar"] [role="radiogroup"] label:hover {
            background: var(--ss-info-bg);
        }
        [data-testid="stSidebar"] [role="radiogroup"] label p {
            color: var(--ss-body) !important;
            font-weight: 550;
        }
        [data-testid="stSidebar"] [data-testid="stExpander"] {
            border-color: var(--ss-border);
            background: var(--ss-page);
        }
        [data-testid="stMetric"] {
            background: var(--ss-surface);
            border: 1px solid var(--ss-border);
            border-radius: 0.6rem;
            padding: 0.75rem 0.9rem;
        }
        [data-testid="stMetricLabel"],
        [data-testid="stMetricValue"] {
            color: var(--ss-heading) !important;
        }
        [data-testid="stTextInput"] label,
        [data-testid="stTextArea"] label,
        [data-testid="stFileUploader"] label,
        [data-testid="stSelectbox"] label,
        [data-testid="stRadio"] label,
        [data-testid="stCheckbox"] label {
            color: var(--ss-heading) !important;
            font-weight: 650;
        }
        [data-testid="stTextInput"] input,
        [data-testid="stTextArea"] textarea,
        [data-testid="stFileUploader"] section,
        [data-baseweb="select"] > div {
            background: var(--ss-surface) !important;
            border-color: var(--ss-border) !important;
            color: var(--ss-body) !important;
        }
        [data-testid="stTextInput"] input::placeholder,
        [data-testid="stTextArea"] textarea::placeholder {
            color: var(--ss-secondary) !important;
            opacity: 1;
        }
        [data-testid="stTextInput"] input:focus,
        [data-testid="stTextArea"] textarea:focus,
        [data-baseweb="select"] > div:focus-within {
            border-color: var(--ss-primary) !important;
            box-shadow: 0 0 0 1px var(--ss-primary) !important;
        }
        [data-testid="stFileUploader"] section,
        [data-testid="stFileUploaderDropzone"] {
            border: 1px dashed var(--ss-border);
            border-radius: 0.5rem;
        }
        [data-testid="stFileUploader"] section > div,
        [data-testid="stFileUploaderDropzone"] > div {
            background: var(--ss-surface) !important;
            color: var(--ss-body) !important;
        }
        [data-testid="stFileUploader"] section span,
        [data-testid="stFileUploader"] section small,
        [data-testid="stFileUploaderDropzone"] span,
        [data-testid="stFileUploaderDropzone"] small {
            color: var(--ss-body) !important;
        }
        [data-testid="stFileUploader"] section svg,
        [data-testid="stFileUploaderDropzone"] svg {
            color: var(--ss-primary) !important;
            fill: currentColor;
            stroke: currentColor;
        }
        [data-testid="stFileUploader"] section button,
        [data-testid="stFileUploaderDropzone"] button {
            background: var(--ss-primary) !important;
            border: 1px solid var(--ss-primary) !important;
            color: var(--ss-surface) !important;
        }
        [data-testid="stFileUploader"] section button:hover,
        [data-testid="stFileUploaderDropzone"] button:hover {
            background: var(--ss-primary-hover) !important;
            border-color: var(--ss-primary-hover) !important;
            color: var(--ss-surface) !important;
        }
        [data-testid="stButton"] > button,
        [data-testid="stDownloadButton"] > button {
            background: var(--ss-surface);
            border: 1px solid var(--ss-primary);
            border-radius: 0.45rem;
            color: var(--ss-primary);
            font-weight: 650;
        }
        [data-testid="stButton"] > button:hover,
        [data-testid="stDownloadButton"] > button:hover {
            background: var(--ss-info-bg);
            border-color: var(--ss-primary-hover);
            color: var(--ss-primary-hover);
        }
        [data-testid="stButton"] > button:focus-visible,
        [data-testid="stDownloadButton"] > button:focus-visible {
            box-shadow: 0 0 0 2px var(--ss-surface), 0 0 0 4px var(--ss-primary);
        }
        [data-testid="stButton"] > button[kind="primary"] {
            background: var(--ss-primary);
            color: var(--ss-surface);
        }
        [data-testid="stButton"] > button[kind="primary"]:hover {
            background: var(--ss-primary-hover);
            color: var(--ss-surface);
        }
        [data-testid="stAlert"] {
            border: 1px solid var(--ss-border);
            border-radius: 0.5rem;
            color: var(--ss-body);
        }
        [data-testid="stAlert"] p {
            color: var(--ss-body) !important;
        }
        [data-testid="stAlert"][kind="success"] {
            background: var(--ss-success-bg);
            color: var(--ss-success-text);
        }
        [data-testid="stAlert"][kind="error"] {
            background: var(--ss-error-bg);
            color: var(--ss-error-text);
        }
        [data-testid="stAlert"][kind="info"] {
            background: var(--ss-info-bg);
            color: var(--ss-info-text);
        }
        [data-testid="stAlert"][kind="warning"] {
            background: var(--ss-warning-bg);
            color: var(--ss-warning-text);
        }
        .ss-eyebrow {
            color: var(--ss-primary);
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            margin-bottom: 0.2rem;
            text-transform: uppercase;
        }
        .ss-subtitle {
            color: var(--ss-secondary) !important;
            font-size: 1rem;
            margin-bottom: 1.25rem;
        }
        .ss-section {
            border-bottom: 1px solid var(--ss-border);
            color: var(--ss-heading);
            font-size: 1.08rem;
            font-weight: 650;
            margin: 1.5rem 0 0.75rem;
            padding-bottom: 0.35rem;
        }
        [data-testid="stExpander"] {
            background: var(--ss-surface);
            border: 1px solid var(--ss-border);
            border-radius: 0.5rem;
        }
        [data-testid="stExpander"] summary p {
            color: var(--ss-heading) !important;
            font-weight: 650;
        }
        [data-testid="stChatInput"] {
            background: var(--ss-surface) !important;
            border: 1px solid var(--ss-border) !important;
            border-radius: 0.6rem;
            box-shadow: 0 4px 18px rgba(23, 50, 77, 0.08);
        }
        [data-testid="stChatInput"] textarea {
            background: var(--ss-surface) !important;
            color: var(--ss-body) !important;
        }
        [data-testid="stChatInput"] textarea::placeholder {
            color: var(--ss-secondary) !important;
            opacity: 1;
        }
        [data-testid="stChatInput"] button {
            color: var(--ss-primary) !important;
        }
        [data-testid="stChatMessage"] {
            border: 1px solid var(--ss-border);
            border-radius: 0.6rem;
            margin-bottom: 0.6rem;
            padding: 0.35rem 0.75rem;
        }
        .ss-product-header {
            border-bottom: 1px solid var(--ss-border);
            margin-bottom: 1rem;
            padding: 0 0 0.65rem;
        }
        .ss-product-header h1 {
            color: var(--ss-heading);
            font-size: clamp(1.7rem, 3vw, 2.35rem);
            line-height: 1.15;
            margin: 0;
            padding: 0 !important;
        }
        .ss-product-tagline {
            color: var(--ss-secondary);
            font-size: 0.98rem;
            margin: 0.25rem 0 0;
        }
        .ss-product-breadcrumb {
            color: var(--ss-primary);
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            margin: 0.55rem 0 0;
            text-transform: uppercase;
        }
        .ss-product-footer {
            border-top: 1px solid var(--ss-border);
            color: var(--ss-secondary);
            margin-top: 1.5rem;
            padding: 0.9rem 0 0.35rem;
        }
        .ss-product-footer-grid {
            display: flex;
            flex-wrap: wrap;
            gap: 1rem 2rem;
            justify-content: space-between;
        }
        .ss-product-footer strong {
            color: var(--ss-heading);
        }
        .ss-product-footer p {
            color: var(--ss-secondary);
            font-size: 0.82rem;
            line-height: 1.5;
            margin: 0.2rem 0;
        }
        .ss-product-footer-meta {
            text-align: right;
        }
        @media (max-width: 640px) {
            .ss-product-header {
                margin-bottom: 0.8rem;
                padding-bottom: 0.55rem;
            }
            .ss-product-footer-grid {
                display: block;
            }
            .ss-product-footer-meta {
                margin-top: 0.75rem;
                text-align: left;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_mode_intro(title: str, description: str, eyebrow: str = "Analysis") -> None:
    st.markdown(f'<div class="ss-eyebrow">{eyebrow}</div>', unsafe_allow_html=True)
    st.subheader(title)
    st.markdown(f'<div class="ss-subtitle">{description}</div>', unsafe_allow_html=True)


def render_section_title(title: str) -> None:
    st.markdown(f'<div class="ss-section">{title}</div>', unsafe_allow_html=True)


def source_mode_label(source_mode: str) -> str:
    return {
        "single_text": "Single Text",
        "batch_csv": "Batch CSV",
        "live_feed": "Live Feed",
    }.get(source_mode, source_mode.replace("_", " ").title())


def workspace_label(mode: str) -> str:
    if mode in ANALYSIS_MODES:
        return mode
    return "Workspace"


def render_header(mode: str) -> None:
    breadcrumb = escape(workspace_label(mode))
    st.markdown(
        f"""
        <header class="ss-product-header" aria-label="SentimentSense application header">
            <div class="ss-eyebrow">Customer feedback intelligence</div>
            <h1>SentimentSense</h1>
            <p class="ss-product-tagline">AI-powered sentiment analysis and text insights</p>
            <p class="ss-product-breadcrumb" aria-label="Current workspace">Workspace / {breadcrumb}</p>
        </header>
        """,
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    current_year = date.today().year
    st.markdown(
        f"""
        <footer class="ss-product-footer" aria-label="SentimentSense footer">
            <div class="ss-product-footer-grid">
                <div>
                    <p><strong>SentimentSense</strong></p>
                    <p>Text sentiment analysis and actionable feedback insights.</p>
                    <p>Built by Nishant Kalane, Sahil Nagpure, Anubhav and Shubhankar W.</p>
                </div>
                <div class="ss-product-footer-meta">
                    <p>&copy; {current_year} SentimentSense</p>
                </div>
            </div>
        </footer>
        """,
        unsafe_allow_html=True,
    )


def build_analytics_payload(
    *,
    source_mode: str,
    prediction: dict[str, Any] | None,
    aggregate: dict[str, Any],
    time_series_df: pd.DataFrame | None = None,
    trend_changes: list[dict[str, Any]] | None = None,
    representative_examples: dict[str, Any] | None = None,
    domain_context: str = "",
) -> dict[str, Any]:
    ts_rows: list[dict[str, Any]] = []
    if time_series_df is not None and not time_series_df.empty:
        ts = time_series_df.copy()
        ts_columns = list(ts.columns)
        if len(ts_columns) >= 2:
            ts = ts.rename(columns={ts_columns[0]: "timestamp", ts_columns[1]: "sentiment_score"})
        if "timestamp" in ts.columns:
            ts["timestamp"] = pd.to_datetime(ts["timestamp"], errors="coerce").astype(str)
        ts_rows = ts.to_dict(orient="records")

    return {
        "source_mode": source_mode,
        "generated_at": datetime.utcnow().isoformat(),
        "domain_context": domain_context,
        "prediction": prediction or {},
        "aggregate": aggregate,
        "time_series": ts_rows,
        "trend_changes": trend_changes or [],
        "representative_examples": representative_examples or {},
    }


def set_latest_analytics_context(payload: dict[str, Any]) -> None:
    st.session_state["latest_analytics_payload"] = payload
    start_analysis_conversation(st.session_state)


def start_analysis_conversation(session_state: Any) -> str:
    conversations = session_state.setdefault("chat_conversations", {})
    previous_id = session_state.get("active_chat_context_id")
    if previous_id is not None:
        conversations[previous_id] = list(session_state.get("chat_history", []))
    elif session_state.get("chat_history"):
        legacy_id = uuid4().hex
        conversations[legacy_id] = list(session_state["chat_history"])

    context_id = uuid4().hex
    conversations[context_id] = []
    session_state["active_chat_context_id"] = context_id
    session_state["chat_history"] = []
    return context_id


def get_active_chat_history(session_state: Any) -> list[dict[str, str]]:
    context_id = session_state.get("active_chat_context_id")
    conversations = session_state.setdefault("chat_conversations", {})
    if context_id is None:
        context_id = start_analysis_conversation(session_state)
    history = conversations.setdefault(context_id, [])
    session_state["chat_history"] = history
    return history


def set_latest_analysis_frame(
    *,
    df: pd.DataFrame,
    source_mode: str,
    text_col: str,
    time_col: str | None,
    domain_context: str,
) -> None:
    st.session_state["latest_analysis_df"] = df.copy()
    st.session_state["latest_analysis_meta"] = {
        "source_mode": source_mode,
        "text_col": text_col,
        "time_col": time_col,
        "domain_context": domain_context,
    }


def analysis_action_targets(source_mode: str) -> dict[str, str]:
    mode_by_source = {
        "single_text": "Single Text",
        "batch_csv": "Batch CSV",
        "live_feed": "Live Feed",
    }
    return {
        "Explore insights": "Chat with Analytics",
        "New analysis": mode_by_source[source_mode],
    }


def request_navigation(session_state: Any, target_mode: str) -> None:
    if target_mode not in ANALYSIS_MODES:
        raise ValueError(f"Unknown navigation mode: {target_mode}")
    session_state["pending_navigation_mode"] = target_mode


def apply_pending_navigation(session_state: Any) -> None:
    pending_mode = session_state.pop("pending_navigation_mode", None)
    if pending_mode in ANALYSIS_MODES:
        session_state["navigation_mode"] = pending_mode


def render_post_analysis_actions(source_mode: str) -> None:
    targets = analysis_action_targets(source_mode)
    st.markdown("### What next?")
    st.caption("Your analysis results are ready. Continue with this analysis or start another one.")
    action_col, new_col = st.columns(2)
    with action_col:
        if st.button(
            targets["Explore insights"],
            type="primary",
            use_container_width=True,
            key=f"explore_{source_mode}",
        ):
            request_navigation(st.session_state, targets["Explore insights"])
            st.rerun()
    with new_col:
        if st.button(targets["New analysis"], use_container_width=True, key=f"new_{source_mode}"):
            request_navigation(st.session_state, targets["New analysis"])
            st.rerun()


def clear_latest_analysis_state(session_state: Any) -> None:
    for key in (
        "latest_analytics_payload",
        "latest_analysis_df",
        "latest_analysis_meta",
    ):
        session_state.pop(key, None)


def serialize_analysis_results_csv(df: pd.DataFrame) -> str:
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    return csv_buffer.getvalue()


def split_live_feed_lines(raw: str) -> tuple[list[str], int]:
    raw_lines = raw.splitlines()
    rows = [line.strip() for line in raw_lines if line.strip()]
    return rows, len(raw_lines) - len(rows)


def read_uploaded_csv(uploaded: Any) -> pd.DataFrame:
    try:
        return pd.read_csv(uploaded)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(f"Could not read the CSV: {exc}") from exc


def prepare_sentiment_chart_data(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, ["sentiment_label"]].copy()


def confidence_caption(task: str, backend: str) -> str:
    if backend == "heuristic fallback":
        score_type = "heuristic/rule-based score"
    else:
        score_type = f"{backend} score"
    return f"{task} {score_type}; not necessarily a calibrated probability"


def final_label_caption() -> str:
    return (
        "Final label is a compatibility field. Use Sentiment and Sarcasm "
        "separately to interpret the prediction; it is not a third sentiment class."
    )


def explain_single_prediction(chatbot: InsightChatbot, text: str, prediction: Any) -> Any:
    return chatbot.explain(
        text=text,
        sentiment_label=prediction.sentiment_label,
        sarcasm_label=prediction.sarcasm_label,
        confidence=prediction.sentiment_confidence,
    )


def _show_inference_error(scope: str, _exc: Exception) -> None:
    clear_latest_analysis_state(st.session_state)
    st.error(
        f"{scope} failed: Unable to complete the analysis. No results were generated. "
        "Check the model configuration and try again."
    )


def gemini_readiness_message() -> str:
    return (
        "Gemini is configured and ready to attempt requests. "
        "Individual requests may still use the local fallback if unavailable."
    )


def _analyze_text_for_ui(engine: SentimentEngine, text: str) -> Any | None:
    try:
        return engine.analyze_text(text)
    except Exception as exc:
        _show_inference_error("Text analysis", exc)
        return None


def _analyze_dataframe_for_ui(
    engine: SentimentEngine, df: pd.DataFrame, text_col: str, scope: str
) -> pd.DataFrame | None:
    try:
        return engine.analyze_dataframe(df, text_col=text_col)
    except Exception as exc:
        _show_inference_error(scope, exc)
        return None


def render_single_text(engine: SentimentEngine, chatbot: InsightChatbot) -> None:
    render_mode_intro(
        "Single Text",
        "Analyze one review, message, or support comment. Results use the configured model backend.",
    )
    domain_context = st.text_input(
        "Context (optional)",
        value="",
        placeholder="For example: ecommerce review or support ticket",
        help="Add context to make follow-up analytics questions more specific.",
    )
    text = st.text_area(
        "Feedback to analyze",
        height=160,
        placeholder="Paste one customer comment, review, or message...",
        help="Enter at least one non-whitespace character.",
    )

    if st.button("Analyze Text", type="primary", use_container_width=True):
        if not text.strip():
            st.warning("Please enter text to analyze.")
            return

        pred = _analyze_text_for_ui(engine, text)
        if pred is None:
            return
        render_section_title("Results")
        c1, c2, c3 = st.columns(3)
        c1.metric("Sentiment", pred.sentiment_label.title())
        c2.metric("Sarcasm", pred.sarcasm_label.replace("_", " ").title())
        c3.metric("Final Label", pred.final_label.title())
        st.caption(final_label_caption())

        st.progress(
            pred.sentiment_confidence,
            text=f"Sentiment score: {pred.sentiment_confidence:.2f} "
            f"({confidence_caption('sentiment', engine.sentiment_backend)})",
        )
        st.progress(
            pred.sarcasm_confidence,
            text=f"Sarcasm score: {pred.sarcasm_confidence:.2f} "
            f"({confidence_caption('sarcasm', engine.sarcasm_backend)})",
        )

        bot_response = explain_single_prediction(chatbot, text, pred)

        with st.expander("Interpretation", expanded=True):
            st.write(bot_response.summary)
            st.write(f"**Why:** {bot_response.rationale}")
            st.write(f"**Suggested action:** {bot_response.action}")

        one_row_df = pd.DataFrame(
            [
                {
                    "sentiment_label": pred.sentiment_label,
                    "sarcasm_label": pred.sarcasm_label,
                    "sentiment_confidence": pred.sentiment_confidence,
                    "sarcasm_confidence": pred.sarcasm_confidence,
                }
            ]
        )
        aggregate = summarize_predictions(one_row_df)
        payload = build_analytics_payload(
            source_mode="single_text",
            prediction=pred.__dict__,
            aggregate=aggregate,
            time_series_df=None,
            trend_changes=[],
            representative_examples={pred.sentiment_label: [{"text": text[:220]}]},
            domain_context=domain_context,
        )
        set_latest_analytics_context(payload)
        single_df = pd.DataFrame(
            [
                {
                    "text": text,
                    "final_label": pred.final_label,
                    "sentiment_label": pred.sentiment_label,
                    "sarcasm_label": pred.sarcasm_label,
                    "sentiment_confidence": pred.sentiment_confidence,
                    "sarcasm_confidence": pred.sarcasm_confidence,
                    "sentiment_score": pred.sentiment_score,
                }
            ]
        )
        set_latest_analysis_frame(
            df=single_df,
            source_mode="single_text",
            text_col="text",
            time_col=None,
            domain_context=domain_context,
        )
        render_post_analysis_actions("single_text")


def render_batch_mode(engine: SentimentEngine) -> None:
    render_mode_intro(
        "Batch CSV",
        "Analyze multiple feedback records while preserving original rows and marking blank text as skipped.",
    )
    domain_context = st.text_input(
        "Context (optional)",
        value="",
        placeholder="For example: support tickets or product reviews",
    )
    uploaded = st.file_uploader("Upload CSV", type=["csv"], key="batch_csv")

    demo_path = Path(__file__).parent / "data" / "big_sample_data.csv"
    use_demo = st.checkbox(
        "Use the built-in demo dataset",
        value=uploaded is None,
        help="This synthetic dataset is for repeatable demonstration only; it is not real customer data.",
    )
    if uploaded is not None and not use_demo:
        try:
            df = read_uploaded_csv(uploaded)
        except ValueError as exc:
            st.error(str(exc))
            return
    elif use_demo and demo_path.exists():
        df = pd.read_csv(demo_path)
        st.info(f"Loaded {len(df):,} demo feedback records.")
    else:
        st.info("Upload a CSV with at least one text column.")
        return

    render_section_title("Input")
    st.caption("Choose the column containing feedback text. A timestamp column is optional for trend analysis.")
    st.dataframe(df.head(), use_container_width=True)
    if df.empty:
        st.warning("Uploaded CSV is empty.")
        return

    text_col = st.selectbox("Select text column", options=list(df.columns), index=0)
    time_col = st.selectbox(
        "Optional timestamp column for trend",
        options=["<none>"] + list(df.columns),
        index=0,
    )

    if st.button("Run Batch Analysis", type="primary", use_container_width=True):
        with st.spinner("Analyzing rows..."):
            result = _analyze_dataframe_for_ui(engine, df, text_col, "Batch analysis")
        if result is None:
            return

        analyzed_result = result[result["analysis_status"] == "analyzed"].copy()
        skipped_count = len(result) - len(analyzed_result)
        if skipped_count:
            st.warning(
                f"Skipped {skipped_count} row(s) because blank text cannot be analyzed."
            )
        if analyzed_result.empty:
            clear_latest_analysis_state(st.session_state)
            st.warning("No rows with analyzable text were found. No sentiment results or charts were generated.")
            st.info("No downloadable prediction CSV is available because this batch contains no analyzable text.")
            st.dataframe(result.head(50), use_container_width=True)
            return

        st.success(f"Analysis complete: {len(analyzed_result)} row(s) processed.")
        render_section_title("Results")
        st.dataframe(result.head(50), use_container_width=True)

        summary = summarize_predictions(analyzed_result)
        c1, c2, c3 = st.columns(3)
        c1.metric("Total", summary["total"])
        c2.metric("Avg Sentiment Score", f"{summary['avg_sentiment_confidence']:.2f}")
        c3.metric("Avg Sarcasm Score", f"{summary['avg_sarcasm_confidence']:.2f}")
        st.caption(
            f"{confidence_caption('Sentiment', engine.sentiment_backend)}. "
            f"{confidence_caption('Sarcasm', engine.sarcasm_backend)}."
        )

        sentiment_chart_data = prepare_sentiment_chart_data(analyzed_result)
        pie = px.pie(
            sentiment_chart_data,
            names="sentiment_label",
            title="Sentiment Distribution",
            hole=0.45,
        )
        st.plotly_chart(pie, use_container_width=True)

        ts = pd.DataFrame()
        if time_col != "<none>":
            ts = create_time_series(analyzed_result, time_col=time_col)
            if not ts.empty:
                line = px.line(ts, x=time_col, y="sentiment_score", title="Sentiment Trend Over Time")
                st.plotly_chart(line, use_container_width=True)
            else:
                st.info("Could not parse timestamp values for trend analysis.")

        payload = build_analytics_payload(
            source_mode="batch_csv",
            prediction=None,
            aggregate=summary,
            time_series_df=ts,
            trend_changes=detect_trend_changes(ts) if not ts.empty else [],
            representative_examples=extract_representative_examples(analyzed_result, text_col=text_col),
            domain_context=domain_context,
        )
        set_latest_analytics_context(payload)
        set_latest_analysis_frame(
            df=analyzed_result,
            source_mode="batch_csv",
            text_col=text_col,
            time_col=(None if time_col == "<none>" else time_col),
            domain_context=domain_context,
        )

        csv_data = serialize_analysis_results_csv(result)
        st.download_button(
            "Download Results CSV",
            data=csv_data,
            file_name=f"sentimentsense_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
        )
        render_post_analysis_actions("batch_csv")


def render_live_feed(engine: SentimentEngine) -> None:
    render_mode_intro(
        "Live Feed",
        "Simulate a message stream by entering one feedback message per line. This is not a connection to a live external feed.",
    )
    domain_context = st.text_input(
        "Context (optional)",
        value="",
        placeholder="For example: social listening or incident feedback",
    )
    raw = st.text_area(
        "Messages",
        height=180,
        placeholder="Message 1\nMessage 2\nMessage 3",
        help="Blank lines are skipped and counted; nonblank messages keep their input order.",
    )

    if st.button("Analyze Stream", use_container_width=True):
        rows, blank_line_count = split_live_feed_lines(raw)
        if blank_line_count:
            st.warning(
                f"Skipped {blank_line_count} blank line(s) because blank text cannot be analyzed."
            )
        if not rows:
            clear_latest_analysis_state(st.session_state)
            st.warning("Add at least one line to simulate a stream.")
            return

        feed_df = pd.DataFrame({"text": rows, "ingested_at": pd.Timestamp.utcnow()})
        out = _analyze_dataframe_for_ui(engine, feed_df, "text", "Live-feed analysis")
        if out is None:
            return
        analyzed_out = out[out["analysis_status"] == "analyzed"].copy()
        skipped_count = len(out) - len(analyzed_out)
        if skipped_count:
            st.warning(
                f"Skipped {skipped_count} line(s) because blank text cannot be analyzed."
            )
        if analyzed_out.empty:
            st.warning("No lines with analyzable text were found. No sentiment results or charts were generated.")
            render_section_title("Results")
            st.dataframe(out, use_container_width=True)
            return
        st.dataframe(out, use_container_width=True)

        sentiment_chart_data = prepare_sentiment_chart_data(analyzed_out)
        bar = px.histogram(
            sentiment_chart_data,
            x="sentiment_label",
            color="sentiment_label",
            title="Live Sentiment Snapshot",
        )
        st.plotly_chart(bar, use_container_width=True)

        summary = summarize_predictions(analyzed_out)
        ts = create_time_series(analyzed_out, time_col="ingested_at")
        payload = build_analytics_payload(
            source_mode="live_feed",
            prediction=None,
            aggregate=summary,
            time_series_df=ts,
            trend_changes=detect_trend_changes(ts) if not ts.empty else [],
            representative_examples=extract_representative_examples(analyzed_out, text_col="text"),
            domain_context=domain_context,
        )
        set_latest_analytics_context(payload)
        set_latest_analysis_frame(
            df=analyzed_out,
            source_mode="live_feed",
            text_col="text",
            time_col="ingested_at",
            domain_context=domain_context,
        )
        render_post_analysis_actions("live_feed")


def _get_scope_filtered_payload(base_payload: dict[str, Any], domain_focus: str, time_scope: str) -> dict[str, Any]:
    analysis_df = st.session_state.get("latest_analysis_df")
    meta = st.session_state.get("latest_analysis_meta")

    if analysis_df is None or meta is None:
        scoped = dict(base_payload)
        scoped["domain_context"] = domain_focus
        return scoped

    working_df = analysis_df.copy()
    time_col = meta.get("time_col")
    text_col = meta.get("text_col", "text")
    source_mode = meta.get("source_mode", base_payload.get("source_mode", "unknown"))

    if time_col and time_col in working_df.columns:
        parsed = pd.to_datetime(working_df[time_col], errors="coerce")
        valid_mask = parsed.notna()
        if valid_mask.any():
            min_day = parsed[valid_mask].min().date()
            max_day = parsed[valid_mask].max().date()
            default_start = min_day
            default_end = max_day

            if time_scope == "Today":
                default_start = max_day
                default_end = max_day
            elif time_scope == "Last 7 Days":
                candidate = max_day - timedelta(days=6)
                default_start = max(min_day, candidate)
                default_end = max_day
            elif time_scope == "Last 30 Days":
                candidate = max_day - timedelta(days=29)
                default_start = max(min_day, candidate)
                default_end = max_day

            start_day, end_day = st.date_input(
                "Date range scope",
                value=(default_start, default_end),
                min_value=min_day,
                max_value=max_day,
                key="chat_date_scope",
            )
            day_series = parsed.dt.date
            day_mask = day_series.between(start_day, end_day)
            working_df = working_df.loc[day_mask].copy()

    if working_df.empty:
        return {
            "source_mode": source_mode,
            "generated_at": datetime.utcnow().isoformat(),
            "domain_context": domain_focus,
            "prediction": {},
            "aggregate": summarize_predictions(pd.DataFrame()),
            "time_series": [],
            "trend_changes": [],
            "representative_examples": {},
        }

    summary = summarize_predictions(working_df)
    ts_df = pd.DataFrame()
    if time_col and time_col in working_df.columns:
        ts_df = create_time_series(working_df, time_col=time_col)

    prediction = base_payload.get("prediction", {}) if source_mode == "single_text" else None
    return build_analytics_payload(
        source_mode=source_mode,
        prediction=prediction,
        aggregate=summary,
        time_series_df=ts_df,
        trend_changes=detect_trend_changes(ts_df) if not ts_df.empty else [],
        representative_examples=extract_representative_examples(working_df, text_col=text_col),
        domain_context=domain_focus,
    )


def render_trend_events_panel(scoped_payload: dict[str, Any]) -> None:
    trend_changes = scoped_payload.get("trend_changes", [])
    st.markdown("**Detected Trend Events**")
    if not trend_changes:
        st.caption("No significant spike/drop events detected in the current scope.")
        return

    for idx, event in enumerate(trend_changes[:3], start=1):
        direction = str(event.get("direction", "change")).title()
        timestamp = str(event.get("timestamp", "unknown time"))
        delta = float(event.get("change", 0.0))
        score = float(event.get("score", 0.0))
        st.markdown(
            f"- **{idx}. {direction}** at {timestamp}: delta `{delta:+.2f}`, resulting score `{score:+.2f}`"
        )


def _run_chat_turn(chatbot: InsightChatbot, scoped_payload: dict[str, Any], user_prompt: str) -> None:
    history = get_active_chat_history(st.session_state)
    history.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Generating insight..."):
            answer = chatbot.ask_analytics(
                user_query=user_prompt,
                analytics_payload=scoped_payload,
                chat_history=history,
            )
            st.markdown(answer)
    history.append({"role": "assistant", "content": answer})


def _build_chat_export_markdown(scoped_payload: dict[str, Any], chat_history: list[dict[str, str]]) -> str:
    lines: list[str] = []
    lines.append("# SentimentSense Chat Transcript")
    lines.append("")
    lines.append(f"Generated At: {datetime.utcnow().isoformat()} UTC")
    lines.append(f"Source Mode: {scoped_payload.get('source_mode', 'unknown')}")
    lines.append(f"Domain Context: {scoped_payload.get('domain_context', '')}")
    lines.append("")
    lines.append("## Conversation")
    lines.append("")
    for idx, msg in enumerate(chat_history, start=1):
        role = str(msg.get("role", "user")).title()
        content = str(msg.get("content", ""))
        lines.append(f"{idx}. **{role}:** {content}")

    lines.append("")
    lines.append("## Scoped Analytics JSON")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(scoped_payload, indent=2))
    lines.append("```")
    return "\n".join(lines)


def render_chat_export_controls(scoped_payload: dict[str, Any]) -> None:
    history = get_active_chat_history(st.session_state)
    if not history:
        return

    export_json = {
        "generated_at": datetime.utcnow().isoformat(),
        "scoped_payload": scoped_payload,
        "chat_history": history,
    }
    export_json_text = json.dumps(export_json, indent=2)
    export_md_text = _build_chat_export_markdown(scoped_payload, history)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    st.markdown("**Export Conversation**")
    e1, e2 = st.columns(2)
    with e1:
        st.download_button(
            "Download JSON",
            data=export_json_text,
            file_name=f"sentimentsense_chat_{ts}.json",
            mime="application/json",
            use_container_width=True,
        )
    with e2:
        st.download_button(
            "Download Markdown",
            data=export_md_text,
            file_name=f"sentimentsense_chat_{ts}.md",
            mime="text/markdown",
            use_container_width=True,
        )

    auto_save_enabled = st.toggle("Auto-save transcript to exports/", key="chat_autosave_enabled")
    if auto_save_enabled:
        _auto_save_chat_exports(export_json_text, export_md_text, history)


def _auto_save_chat_exports(export_json_text: str, export_md_text: str, history: list[dict[str, str]]) -> None:
    if not history:
        return

    last_msg = history[-1]
    signature = f"{len(history)}::{last_msg.get('role', '')}::{last_msg.get('content', '')[:120]}"
    if st.session_state.get("chat_autosave_signature") == signature:
        return

    exports_dir = Path("exports")
    exports_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = exports_dir / f"sentimentsense_chat_{ts}.json"
    md_path = exports_dir / f"sentimentsense_chat_{ts}.md"

    json_path.write_text(export_json_text, encoding="utf-8")
    md_path.write_text(export_md_text, encoding="utf-8")
    st.session_state["chat_autosave_signature"] = signature
    st.caption(f"Auto-saved: {json_path} and {md_path}")


def render_saved_sessions_browser() -> None:
    st.markdown("**Saved Sessions**")
    exports_dir = Path("exports")
    if not exports_dir.exists():
        st.caption("No saved sessions yet. Enable auto-save or export manually.")
        return

    files = [p for p in exports_dir.iterdir() if p.is_file() and p.suffix.lower() in {".json", ".md"}]
    if not files:
        st.caption("No saved sessions found in exports/.")
        return

    files = sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)
    options = {f"{p.name} ({datetime.fromtimestamp(p.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')})": p for p in files}
    selected_label = st.selectbox("Browse exports", options=list(options.keys()), key="saved_session_picker")
    selected_path = options[selected_label]

    content = selected_path.read_text(encoding="utf-8", errors="ignore")
    st.caption(f"Previewing: {selected_path}")
    st.text_area("Session Preview", value=content[:4000], height=220, disabled=True)
    b1, b2 = st.columns(2)
    with b1:
        st.download_button(
            "Download Selected Session",
            data=content,
            file_name=selected_path.name,
            mime=("application/json" if selected_path.suffix.lower() == ".json" else "text/markdown"),
            use_container_width=True,
        )
    with b2:
        confirm_key = f"confirm_delete_{selected_path.name}"
        confirm_delete = st.checkbox("Confirm delete", key=confirm_key)
        if st.button("Delete Selected Session", use_container_width=True, key=f"delete_{selected_path.name}"):
            if confirm_delete:
                selected_path.unlink(missing_ok=True)
                st.success(f"Deleted: {selected_path.name}")
                st.rerun()
            else:
                st.warning("Check 'Confirm delete' before deleting.")


def render_chat_with_analytics(chatbot: InsightChatbot) -> None:
    render_mode_intro(
        "Chat with Analytics",
        "Ask follow-up questions about the most recent successful analysis. The active conversation stays scoped to that analysis.",
        eyebrow="Explore insights",
    )

    payload = st.session_state.get("latest_analytics_payload")
    if not payload:
        st.info("No analysis is available yet. Run Single Text, Batch CSV, or Live Feed analysis to enable insights.")
        return
    history = get_active_chat_history(st.session_state)
    st.info(
        f"Active analysis: {source_mode_label(str(payload.get('source_mode', 'unknown')))}. "
        "Questions and exports use this analysis only."
    )

    domain_default = str(payload.get("domain_context", ""))
    domain_focus = st.text_input("Domain focus", value=domain_default, placeholder="e.g., customer support only")
    time_scope = st.radio(
        "Time scope",
        options=["Today", "Last 7 Days", "Last 30 Days", "Custom"],
        horizontal=True,
    )
    scoped_payload = _get_scope_filtered_payload(payload, domain_focus=domain_focus, time_scope=time_scope)

    if chatbot.gemini_available:
        st.success(gemini_readiness_message())
    else:
        st.warning("Gemini not configured. Using local insight fallback.")
        if chatbot.gemini_error:
            st.caption(chatbot.gemini_error)

    c1, c2 = st.columns([3, 1])
    with c1:
        st.caption("Try: Why is this marked sarcastic? • Summarize today's trend • What caused the negative drop?")
    with c2:
        if st.button("Clear Chat", use_container_width=True):
            history.clear()
            st.rerun()

    with st.expander("Current Analytics JSON", expanded=False):
        st.code(json.dumps(scoped_payload, indent=2), language="json")

    render_trend_events_panel(scoped_payload)

    render_section_title("Suggested questions")
    q1, q2, q3 = st.columns(3)
    selected_prompt = None
    with q1:
        if st.button("Summarize Trend", use_container_width=True, key="quick_trend"):
            selected_prompt = "Summarize the current sentiment trend in simple terms."
    with q2:
        if st.button("Explain Top Drop", use_container_width=True, key="quick_drop"):
            selected_prompt = "Explain the biggest negative sentiment drop and likely causes."
    with q3:
        if st.button("Recommend Actions", use_container_width=True, key="quick_action"):
            selected_prompt = "What actions should be taken based on this scoped sentiment analysis?"

    for msg in history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("Ask about sarcasm reasons, trend changes, spikes, or next actions...")
    active_prompt = selected_prompt or prompt
    if active_prompt:
        _run_chat_turn(chatbot, scoped_payload, active_prompt)

    render_chat_export_controls(scoped_payload)
    render_saved_sessions_browser()


def render_sidebar(engine: SentimentEngine) -> str:
    st.sidebar.header("Workspace")
    st.sidebar.caption("Choose an analysis workflow or explore the latest results.")
    apply_pending_navigation(st.session_state)
    current_mode = st.session_state.get("navigation_mode", "Single Text")
    if current_mode not in ANALYSIS_MODES:
        current_mode = "Single Text"
        st.session_state["navigation_mode"] = current_mode
    mode = st.sidebar.radio(
        "Select workflow",
        options=list(ANALYSIS_MODES),
        index=list(ANALYSIS_MODES).index(current_mode),
        key="navigation_mode",
    )

    _ = engine.inference_backend
    return mode


def main() -> None:
    engine = get_engine()
    chatbot = get_chatbot()

    render_app_styles()
    mode = render_sidebar(engine)
    render_header(mode)

    if mode == "Single Text":
        render_single_text(engine, chatbot)
    elif mode == "Batch CSV":
        render_batch_mode(engine)
    elif mode == "Live Feed":
        render_live_feed(engine)
    else:
        render_chat_with_analytics(chatbot)
    render_footer()


if __name__ == "__main__":
    main()
