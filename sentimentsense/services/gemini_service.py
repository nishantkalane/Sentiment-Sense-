# pyright: reportMissingImports=false
from __future__ import annotations

import json
import os
import importlib
from typing import Any


SYSTEM_PROMPT = """
You are an AI Analytics and Insight Assistant for SentimentSense.

Role boundaries:
- You must NOT perform sentiment classification or sarcasm classification yourself.
- You must only reason over SentimentSense outputs provided in analytics JSON.
- If required fields are missing, explicitly state limitations.

Responsibilities:
- Explain why a prediction was made using provided labels, sarcasm flags, and confidence values.
- Clarify sarcasm-related decisions.
- Summarize sentiment trends over time.
- Highlight probable causes of spikes/drops using available metadata and text evidence.
- Translate analytics into simple language for non-technical users.
- Provide actionable recommendations for business, social media, and customer-feedback workflows.

Tone:
- Professional
- Clear
- Insight-focused
- Non-technical-user friendly

Formatting:
- Use concise markdown headings and bullet points.
- Keep recommendations explicit and prioritized.
""".strip()


class GeminiService:
    def __init__(self, api_key_env: str = "GEMINI_API_KEY", model_name: str = "gemini-1.5-flash") -> None:
        self._load_dotenv_if_available()
        self.api_key = os.getenv(api_key_env, "").strip()
        self.model_name = model_name
        self._model = None
        self._init_error: str | None = None

        if not self.api_key:
            self._init_error = f"Missing environment variable: {api_key_env}"
            return

        try:
            genai = __import__("google.generativeai", fromlist=["GenerativeModel"])
            genai.configure(api_key=self.api_key)
            self._model = genai.GenerativeModel(model_name=self.model_name)
        except Exception as exc:
            self._init_error = f"Gemini initialization failed: {exc}"

    @staticmethod
    def _load_dotenv_if_available() -> None:
        try:
            dotenv_module = importlib.import_module("dotenv")
            load_dotenv = getattr(dotenv_module, "load_dotenv", None)
            if callable(load_dotenv):
                load_dotenv()
        except Exception:
            return

    @property
    def is_available(self) -> bool:
        return self._model is not None

    @property
    def init_error(self) -> str | None:
        return self._init_error

    def generate_analytics_response(
        self,
        user_query: str,
        analytics_payload: dict[str, Any],
        chat_history: list[dict[str, str]] | None = None,
    ) -> str:
        if not self.is_available:
            raise RuntimeError(self._init_error or "Gemini service unavailable")

        history_text = self._format_history(chat_history or [])
        prompt = self._build_prompt(user_query=user_query, analytics_payload=analytics_payload, history_text=history_text)

        response = self._model.generate_content(prompt)
        text = getattr(response, "text", "")
        if not text:
            return "I couldn't generate insights from the provided analytics context. Please try rephrasing your question."
        return text

    def _build_prompt(self, user_query: str, analytics_payload: dict[str, Any], history_text: str) -> str:
        payload_json = json.dumps(analytics_payload, ensure_ascii=False, indent=2)
        return (
            f"{SYSTEM_PROMPT}\n\n"
            f"Conversation history:\n{history_text}\n\n"
            f"Structured Analytics JSON:\n{payload_json}\n\n"
            f"User question:\n{user_query}\n\n"
            "Return a concise, structured answer with:\n"
            "1) Direct answer\n"
            "2) Evidence from analytics\n"
            "3) Recommended actions\n"
            "4) Any uncertainty or missing data\n"
        )

    @staticmethod
    def _format_history(chat_history: list[dict[str, str]]) -> str:
        if not chat_history:
            return "(no prior conversation)"
        lines: list[str] = []
        for item in chat_history[-8:]:
            role = item.get("role", "user")
            content = item.get("content", "")
            lines.append(f"{role.upper()}: {content}")
        return "\n".join(lines)
