from __future__ import annotations

import re
from dataclasses import dataclass


URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_PATTERN = re.compile(r"@[A-Za-z0-9_]+")
HASHTAG_PATTERN = re.compile(r"#([A-Za-z0-9_]+)")
WHITESPACE_PATTERN = re.compile(r"\s+")


@dataclass
class CleanOptions:
    lowercase: bool = True
    remove_urls: bool = True
    remove_mentions: bool = True
    keep_hashtag_text: bool = True


def clean_text(text: str, options: CleanOptions | None = None) -> str:
    options = options or CleanOptions()
    normalized = text.strip()

    if options.remove_urls:
        normalized = URL_PATTERN.sub("", normalized)
    if options.remove_mentions:
        normalized = MENTION_PATTERN.sub("", normalized)

    if options.keep_hashtag_text:
        normalized = HASHTAG_PATTERN.sub(r" \1", normalized)

    normalized = WHITESPACE_PATTERN.sub(" ", normalized).strip()
    if options.lowercase:
        normalized = normalized.lower()
    return normalized
