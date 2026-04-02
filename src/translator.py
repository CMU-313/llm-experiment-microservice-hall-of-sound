"""
Translation logic for this HTTP microservice.

Another application (separate repo) calls this service over the network; it does not
import this package. Keep the JSON response shape stable for those clients.

Ollama must be reachable at OLLAMA_HOST with the model available (default model
deepseek-r1:1.5b). Use a full URL in Docker, e.g. http://ollama:11434; locally
http://127.0.0.1:11434 or localhost:11434. Override OLLAMA_MODEL if needed.
"""
import os
import re
from typing import Optional

from ollama import Client


def _ollama_host() -> str:
    # ollama-python accepts host:port or full http URL; Docker Compose sets http://ollama:11434
    return os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")


def _ollama_model() -> str:
    return os.getenv("OLLAMA_MODEL", "deepseek-r1:1.5b")


def _chat(messages: list[dict]) -> str:
    """Call Ollama; exposed for unit tests via unittest.mock.patch."""
    client = Client(host=_ollama_host())
    response = client.chat(model=_ollama_model(), messages=messages, think=False, keep_alive=-1)
    content = getattr(response.message, "content", None) if response.message else None
    return (content or "").strip()


def _normalize_language_response(resp: str) -> Optional[str]:
    if not resp or not isinstance(resp, str):
        return None

    cleaned = resp.strip().lower()

    known_languages = [
        "english",
        "german",
        "french",
        "spanish",
        "chinese",
        "japanese",
        "korean",
        "italian",
        "portuguese",
        "russian",
        "arabic",
        "hindi",
        "vietnamese",
        "thai",
        "turkish",
        "catalan",
    ]

    for lang in known_languages:
        if lang in cleaned:
            return lang.title()

    return None


def _normalize_translation_response(resp: str) -> Optional[str]:
    if not resp or not isinstance(resp, str):
        return None

    cleaned = resp.strip()

    bad_patterns = [
        "i don't understand",
        "cannot translate",
        "can't translate",
        "unable to translate",
        "no translation",
        "error",
        "unknown",
        "n/a",
    ]
    lower_cleaned = cleaned.lower()
    if any(p in lower_cleaned for p in bad_patterns):
        return None

    cleaned = re.sub(
        r"^\s*(translation|output|answer)\s*:\s*", "", cleaned, flags=re.IGNORECASE
    )

    if not cleaned:
        return None

    return cleaned


def translate_content(content: str) -> tuple[bool, str]:
    """
    Classify whether `content` is English; if not, translate to English via Ollama.

    Returns (is_english, text) where `text` is either the original (if English) or
    the English translation. On classification/LLM failure, returns the original
    content as English-safe fallback; on failed translation, returns a placeholder.
    """
    if not content.strip():
        return True, content

    def get_language(post: str) -> str:
        return _chat(
            [
                {
                    "role": "user",
                    "content": (
                        "Identify the language of the following text. "
                        "Reply with only the language name in English and nothing else.\n\n"
                        f"Text: {post}"
                    ),
                }
            ]
        )

    def get_translation(post: str) -> str:
        return _chat(
            [
                {
                    "role": "user",
                    "content": (
                        "Translate the following text into English. "
                        "Reply with only the English translation and nothing else.\n\n"
                        f"Text: {post}"
                    ),
                }
            ]
        )

    try:
        language_raw = get_language(content)
        language = _normalize_language_response(language_raw)

        if language is None:
            return True, content

        if language.lower() == "english":
            return True, content

        translation_raw = get_translation(content)
        translation = _normalize_translation_response(translation_raw)

        if translation is None:
            return False, "Translation unavailable"

        return False, translation

    except Exception:
        return True, content
