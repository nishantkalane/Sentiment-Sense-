from __future__ import annotations

import io
import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

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


@st.cache_resource
def get_engine() -> SentimentEngine:
    return SentimentEngine()


@st.cache_resource
def get_chatbot() -> InsightChatbot:
    return InsightChatbot()


def render_header() -> None:
    st.title("SentimentSense")
    st.caption("Real-time sentiment analysis + sarcasm detection for text streams, feedback, and social data.")


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
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []


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


def render_single_text(engine: SentimentEngine, chatbot: InsightChatbot) -> None:
    st.subheader("Single Text Analysis")
    domain_context = st.text_input("Optional domain/context", value="", placeholder="e.g., ecommerce reviews")
    text = st.text_area("Enter text", height=120, placeholder="Type a post, review, or feedback...")

    if st.button("Analyze Text", type="primary", use_container_width=True):
        if not text.strip():
            st.warning("Please enter text to analyze.")
            return

        pred = engine.analyze_text(text)
        c1, c2, c3 = st.columns(3)
        c1.metric("Sentiment", pred.sentiment_label.title())
        c2.metric("Sarcasm", pred.sarcasm_label.replace("_", " ").title())
        c3.metric("Final Label", pred.final_label.title())

        st.progress(pred.sentiment_confidence, text=f"Sentiment confidence: {pred.sentiment_confidence:.2f}")
        st.progress(pred.sarcasm_confidence, text=f"Sarcasm confidence: {pred.sarcasm_confidence:.2f}")

        bot_response = chatbot.explain(
            text=text,
            sentiment_label=pred.final_label,
            sarcasm_label=pred.sarcasm_label,
            confidence=pred.sentiment_confidence,
        )

        with st.expander("AI Explanation Chat", expanded=True):
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
            representative_examples={pred.final_label: [{"text": text[:220]}]},
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


def render_batch_mode(engine: SentimentEngine) -> None:
    st.subheader("Batch Analysis (CSV)")
    domain_context = st.text_input("Optional domain/context", value="", placeholder="e.g., support tickets")
    uploaded = st.file_uploader("Upload CSV", type=["csv"], key="batch_csv")

    if uploaded is None:
        st.info("Upload a CSV with at least one text column.")
        return

    df = pd.read_csv(uploaded)
    st.write("Preview", df.head())
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
            result = engine.analyze_dataframe(df, text_col=text_col)

        st.success(f"Processed {len(result)} rows.")
        st.dataframe(result.head(50), use_container_width=True)

        summary = summarize_predictions(result)
        c1, c2, c3 = st.columns(3)
        c1.metric("Total", summary["total"])
        c2.metric("Avg Sentiment Conf.", f"{summary['avg_sentiment_confidence']:.2f}")
        c3.metric("Avg Sarcasm Conf.", f"{summary['avg_sarcasm_confidence']:.2f}")

        pie = px.pie(
            result,
            names="final_label",
            title="Prediction Distribution",
            hole=0.45,
        )
        st.plotly_chart(pie, use_container_width=True)

        ts = pd.DataFrame()
        if time_col != "<none>":
            ts = create_time_series(result, time_col=time_col)
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
            representative_examples=extract_representative_examples(result, text_col=text_col),
            domain_context=domain_context,
        )
        set_latest_analytics_context(payload)
        set_latest_analysis_frame(
            df=result,
            source_mode="batch_csv",
            text_col=text_col,
            time_col=(None if time_col == "<none>" else time_col),
            domain_context=domain_context,
        )

        csv_buffer = io.StringIO()
        result.to_csv(csv_buffer, index=False)
        st.download_button(
            "Download Results CSV",
            data=csv_buffer.getvalue(),
            file_name=f"sentimentsense_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
        )


def render_live_feed(engine: SentimentEngine) -> None:
    st.subheader("Live Feed Simulator")
    st.caption("Paste one message per line to simulate real-time stream analytics.")
    domain_context = st.text_input("Optional domain/context", value="", placeholder="e.g., social listening")
    raw = st.text_area("Stream input", height=160, placeholder="Message 1\nMessage 2\nMessage 3")

    if st.button("Analyze Stream", use_container_width=True):
        rows = [line.strip() for line in raw.splitlines() if line.strip()]
        if not rows:
            st.warning("Add at least one line to simulate a stream.")
            return

        feed_df = pd.DataFrame({"text": rows, "ingested_at": pd.Timestamp.utcnow()})
        out = engine.analyze_dataframe(feed_df, text_col="text")
        st.dataframe(out, use_container_width=True)

        bar = px.histogram(out, x="final_label", color="final_label", title="Live Sentiment Snapshot")
        st.plotly_chart(bar, use_container_width=True)

        summary = summarize_predictions(out)
        ts = create_time_series(out, time_col="ingested_at")
        payload = build_analytics_payload(
            source_mode="live_feed",
            prediction=None,
            aggregate=summary,
            time_series_df=ts,
            trend_changes=detect_trend_changes(ts) if not ts.empty else [],
            representative_examples=extract_representative_examples(out, text_col="text"),
            domain_context=domain_context,
        )
        set_latest_analytics_context(payload)
        set_latest_analysis_frame(
            df=out,
            source_mode="live_feed",
            text_col="text",
            time_col="ingested_at",
            domain_context=domain_context,
        )


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
    st.session_state["chat_history"].append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Generating insight..."):
            answer = chatbot.ask_analytics(
                user_query=user_prompt,
                analytics_payload=scoped_payload,
                chat_history=st.session_state["chat_history"],
            )
            st.markdown(answer)
    st.session_state["chat_history"].append({"role": "assistant", "content": answer})


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
    history = st.session_state.get("chat_history", [])
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
    st.subheader("Chat with Analytics")
    st.caption("Ask follow-up questions based on SentimentSense outputs and trends.")

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    payload = st.session_state.get("latest_analytics_payload")
    if not payload:
        st.info("No analytics context available yet. Run Single Text, Batch CSV, or Live Feed analysis first.")
        return

    domain_default = str(payload.get("domain_context", ""))
    domain_focus = st.text_input("Domain focus", value=domain_default, placeholder="e.g., customer support only")
    time_scope = st.radio(
        "Time scope",
        options=["Today", "Last 7 Days", "Last 30 Days", "Custom"],
        horizontal=True,
    )
    scoped_payload = _get_scope_filtered_payload(payload, domain_focus=domain_focus, time_scope=time_scope)

    if chatbot.gemini_available:
        st.success("Gemini analytics assistant is active.")
    else:
        st.warning("Gemini not configured. Using local insight fallback.")
        if chatbot.gemini_error:
            st.caption(chatbot.gemini_error)

    c1, c2 = st.columns([3, 1])
    with c1:
        st.caption("Try: Why is this marked sarcastic? • Summarize today's trend • What caused the negative drop?")
    with c2:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state["chat_history"] = []
            st.rerun()

    with st.expander("Current Analytics JSON", expanded=False):
        st.code(json.dumps(scoped_payload, indent=2), language="json")

    render_trend_events_panel(scoped_payload)

    st.markdown("**Quick Questions**")
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

    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("Ask about sarcasm reasons, trend changes, spikes, or next actions...")
    active_prompt = selected_prompt or prompt
    if active_prompt:
        _run_chat_turn(chatbot, scoped_payload, active_prompt)

    render_chat_export_controls(scoped_payload)
    render_saved_sessions_browser()


def render_sidebar(engine: SentimentEngine) -> str:
    st.sidebar.header("Navigation")
    mode = st.sidebar.radio("Select Mode", ["Single Text", "Batch CSV", "Live Feed", "Chat with Analytics"])

    if engine.load_errors:
        st.sidebar.warning("Running with fallback heuristics for one or more models.")
        with st.sidebar.expander("Model load diagnostics"):
            for err in engine.load_errors:
                st.text(err)

    st.sidebar.markdown("---")
    st.sidebar.write("Built with Transformers + Streamlit")
    return mode


def main() -> None:
    engine = get_engine()
    chatbot = get_chatbot()

    render_header()
    mode = render_sidebar(engine)

    if mode == "Single Text":
        render_single_text(engine, chatbot)
    elif mode == "Batch CSV":
        render_batch_mode(engine)
    elif mode == "Live Feed":
        render_live_feed(engine)
    else:
        render_chat_with_analytics(chatbot)


if __name__ == "__main__":
    main()
