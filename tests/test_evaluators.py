"""Tests for the pure-Python WER logic in evaluators.py — no whisper/torch needed."""
import pytest
from ttseval.evaluators import _normalise, _word_error_rate


class TestNormalise:
    def test_lowercases(self):
        assert _normalise("Hello World") == "hello world"

    def test_strips_punctuation(self):
        assert _normalise("Hello, World!") == "hello world"

    def test_collapses_whitespace(self):
        assert _normalise("hello   world") == "hello world"

    def test_empty_string(self):
        assert _normalise("") == ""

    def test_numbers_preserved(self):
        assert _normalise("step 1") == "step 1"


class TestWordErrorRate:
    def test_identical_strings_zero_wer(self):
        assert _word_error_rate("hello world", "hello world") == 0.0

    def test_empty_reference_returns_zero(self):
        assert _word_error_rate("", "anything") == 0.0

    def test_all_wrong_returns_one(self):
        assert _word_error_rate("hello world", "foo bar") == 1.0

    def test_one_substitution(self):
        # "hello world" vs "hello earth" — 1 sub out of 2 words
        assert _word_error_rate("hello world", "hello earth") == pytest.approx(0.5)

    def test_one_deletion(self):
        # ref="hello world", hyp="hello" — 1 deletion out of 2
        assert _word_error_rate("hello world", "hello") == pytest.approx(0.5)

    def test_one_insertion(self):
        # ref="hello", hyp="hello world" — 1 insertion out of 1
        assert _word_error_rate("hello", "hello world") == pytest.approx(1.0)

    def test_perfect_long_sentence(self):
        s = "the quick brown fox jumps over the lazy dog"
        assert _word_error_rate(s, s) == 0.0

    def test_wer_not_capped_at_one(self):
        # More insertions than reference words — WER can exceed 1.0
        wer = _word_error_rate("hi", "hi there how are you")
        assert wer > 1.0
