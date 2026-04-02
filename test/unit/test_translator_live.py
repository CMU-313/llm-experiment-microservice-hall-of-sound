"""
Live integration tests against the running Ollama model.
These call translate_content() without mocking — hitting the real LLM.

Accuracy threshold: >= 40% of language classification tests must pass.
Each test prints timing info.
"""
import time

import pytest

from src.translator import translate_content


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _timed_translate(content):
    """Call translate_content and return (is_english, translated, elapsed_s)."""
    start = time.time()
    is_english, translated = translate_content(content)
    elapsed = time.time() - start
    return is_english, translated, elapsed


# ---------------------------------------------------------------------------
# Language classification test cases
# ---------------------------------------------------------------------------

CLASSIFICATION_CASES = [
    # (input, expected_is_english, label)
    ("Hello world", True, "English"),
    ("This is an English message", True, "English"),
    ("Good morning, how are you?", True, "English"),
    ("这是一条中文消息", False, "Chinese"),
    ("你好世界", False, "Chinese"),
    ("Bonjour le monde", False, "French"),
    ("Ceci est un message en français", False, "French"),
    ("¡Hola Mundo!", False, "Spanish"),
    ("Esta es un mensaje en español", False, "Spanish"),
    ("Dies ist eine Nachricht auf Deutsch", False, "German"),
    ("Guten Morgen, wie geht es Ihnen?", False, "German"),
    ("Questo è un messaggio in italiano", False, "Italian"),
    ("こんにちは世界", False, "Japanese"),
    ("이것은 한국어 메시지입니다", False, "Korean"),
    ("Это сообщение на русском", False, "Russian"),
]


class TestLanguageClassificationLive:
    """Run all classification cases, collect results, enforce >= 40% accuracy."""

    results = []

    @pytest.fixture(autouse=True, scope="class")
    def run_all_cases(self):
        """Run every case once, store results for the accuracy check."""
        self.__class__.results = []
        for content, expected_is_english, label in CLASSIFICATION_CASES:
            is_english, translated, elapsed = _timed_translate(content)
            correct = is_english == expected_is_english
            self.__class__.results.append({
                "input": content,
                "label": label,
                "expected_is_english": expected_is_english,
                "got_is_english": is_english,
                "translated": translated,
                "elapsed": elapsed,
                "correct": correct,
            })
            status = "PASS" if correct else "FAIL"
            print(
                f"  [{status}] {label:10s} | is_english={str(is_english):5s} "
                f"(expected {str(expected_is_english):5s}) | {elapsed:.1f}s | "
                f"input={content[:40]}"
            )
        yield

    def test_accuracy_threshold(self):
        """At least 40% of classification cases must be correct."""
        total = len(self.results)
        correct = sum(1 for r in self.results if r["correct"])
        accuracy = correct / total if total else 0
        print(f"\n{'='*60}")
        print(f"Classification accuracy: {correct}/{total} = {accuracy:.0%}")
        print(f"Threshold: 40%")
        print(f"{'='*60}")
        assert accuracy >= 0.40, (
            f"Accuracy {accuracy:.0%} ({correct}/{total}) is below 40% threshold"
        )

    def test_all_responses_within_timeout(self):
        """No single call should take more than 30s."""
        for r in self.results:
            assert r["elapsed"] < 30, (
                f"Call for '{r['input'][:30]}' took {r['elapsed']:.1f}s (>30s)"
            )


# ---------------------------------------------------------------------------
# Translation quality spot-checks (non-English -> English)
# ---------------------------------------------------------------------------

class TestTranslationQualityLive:
    """Check that non-English inputs get some English translation back."""

    @pytest.mark.parametrize("content,label", [
        ("这是一条中文消息", "Chinese"),
        ("Bonjour le monde", "French"),
        ("¡Hola Mundo!", "Spanish"),
        ("Dies ist eine Nachricht auf Deutsch", "German"),
    ])
    def test_non_english_gets_translation(self, content, label):
        is_english, translated, elapsed = _timed_translate(content)
        print(f"  [{label}] {elapsed:.1f}s | is_english={is_english} | translated={translated[:80]}")
        # If the model correctly detected non-English, it should provide a translation
        if not is_english:
            assert translated and len(translated) > 0, "Translation should not be empty"
            assert translated != content, "Translation should differ from input"
