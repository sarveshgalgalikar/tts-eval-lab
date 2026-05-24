"""Tests for report.py — pure logic, no TTS models or torch required."""
import pytest
from ttseval.pipeline import EngineSummary, SentenceResult
from ttseval.report import leaderboard_markdown, build_claude_prompt


def _make_summary(engine, avg_wer, avg_mos, avg_rtf, n_sentences=2):
    sentences = [
        SentenceResult(sentence=f"Sentence {i}", wer=avg_wer, mos=avg_mos, rtf=avg_rtf)
        for i in range(n_sentences)
    ]
    return EngineSummary(
        engine=engine,
        avg_wer=avg_wer,
        avg_mos=avg_mos,
        avg_rtf=avg_rtf,
        sentence_results=sentences,
    )


class TestLeaderboardMarkdown:
    def test_empty_results_returns_placeholder(self):
        md = leaderboard_markdown({})
        assert "No results" in md

    def test_single_engine_renders_table(self):
        results = {"pyttsx3": _make_summary("pyttsx3", 0.1, 3.5, 0.8)}
        md = leaderboard_markdown(results)
        assert "pyttsx3" in md
        assert "MOS" in md
        assert "WER" in md
        assert "RTF" in md

    def test_best_mos_ranks_first(self):
        results = {
            "bad":  _make_summary("bad",  avg_wer=0.5, avg_mos=1.0, avg_rtf=1.0),
            "good": _make_summary("good", avg_wer=0.1, avg_mos=4.5, avg_rtf=0.5),
        }
        md = leaderboard_markdown(results)
        lines = md.splitlines()
        data_lines = [l for l in lines if "|" in l and "Rank" not in l and "---" not in l]
        assert data_lines[0].split("|")[2].strip() == "**good**"

    def test_rank_1_assigned_to_top_engine(self):
        results = {
            "engine_a": _make_summary("engine_a", avg_wer=0.05, avg_mos=4.8, avg_rtf=0.3),
            "engine_b": _make_summary("engine_b", avg_wer=0.4,  avg_mos=2.0, avg_rtf=2.0),
        }
        md = leaderboard_markdown(results)
        first_data_row = [l for l in md.splitlines() if "|" in l and "Rank" not in l and "---" not in l][0]
        assert "| 1 |" in first_data_row
        assert "engine_a" in first_data_row

    def test_three_engines_all_present(self):
        results = {
            "a": _make_summary("a", 0.1, 4.0, 0.5),
            "b": _make_summary("b", 0.2, 3.5, 0.8),
            "c": _make_summary("c", 0.3, 3.0, 1.2),
        }
        md = leaderboard_markdown(results)
        for name in ("a", "b", "c"):
            assert name in md

    def test_scores_rounded_to_three_decimals(self):
        results = {"eng": _make_summary("eng", avg_wer=0.123456, avg_mos=3.141592, avg_rtf=0.999999)}
        md = leaderboard_markdown(results)
        assert "0.123" in md
        assert "3.142" in md

    def test_equal_engines_all_ranked(self):
        results = {
            "x": _make_summary("x", 0.1, 3.0, 1.0),
            "y": _make_summary("y", 0.1, 3.0, 1.0),
        }
        md = leaderboard_markdown(results)
        assert "| 1 |" in md
        assert "| 2 |" in md


class TestBuildClaudePrompt:
    def test_empty_results_returns_placeholder(self):
        prompt = build_claude_prompt({})
        assert "Run the pipeline" in prompt

    def test_prompt_contains_engine_name(self):
        results = {"kokoro": _make_summary("kokoro", 0.05, 4.2, 0.4)}
        prompt = build_claude_prompt(results)
        assert "kokoro" in prompt

    def test_prompt_contains_analysis_sections(self):
        results = {"pyttsx3": _make_summary("pyttsx3", 0.15, 3.1, 1.2)}
        prompt = build_claude_prompt(results)
        assert "naturalness" in prompt.lower() or "mos" in prompt.lower()
        assert "intelligib" in prompt.lower() or "wer" in prompt.lower()

    def test_prompt_contains_leaderboard(self):
        results = {"eng": _make_summary("eng", 0.1, 3.5, 0.8)}
        prompt = build_claude_prompt(results)
        assert "Leaderboard" in prompt or "leaderboard" in prompt

    def test_prompt_includes_sentence_breakdown(self):
        results = {"eng": _make_summary("eng", 0.1, 3.5, 0.8, n_sentences=3)}
        prompt = build_claude_prompt(results)
        assert "Sentence" in prompt
