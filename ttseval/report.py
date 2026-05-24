"""Render leaderboard markdown and build a prompt for qualitative analysis on claude.ai."""
from .pipeline import EngineSummary


def leaderboard_markdown(results: dict[str, EngineSummary]) -> str:
    if not results:
        return "_No results yet. Run the pipeline first._"

    # Rank by composite score: lower WER is better, higher MOS is better, lower RTF is better.
    # Normalise each metric to [0,1] then compute rank_score = MOS_norm - WER_norm - RTF_norm.
    summaries = list(results.values())

    wer_vals = [s.avg_wer for s in summaries]
    mos_vals = [s.avg_mos for s in summaries]
    rtf_vals = [s.avg_rtf for s in summaries]

    def _norm(val, lo, hi):
        return (val - lo) / (hi - lo) if hi != lo else 0.5

    wer_lo, wer_hi = min(wer_vals), max(wer_vals)
    mos_lo, mos_hi = min(mos_vals), max(mos_vals)
    rtf_lo, rtf_hi = min(rtf_vals), max(rtf_vals)

    def rank_score(s: EngineSummary) -> float:
        return (
            _norm(s.avg_mos, mos_lo, mos_hi)
            - _norm(s.avg_wer, wer_lo, wer_hi)
            - _norm(s.avg_rtf, rtf_lo, rtf_hi)
        )

    ranked = sorted(summaries, key=rank_score, reverse=True)

    lines = [
        "## TTS Leaderboard",
        "",
        "| Rank | Engine | MOS ↑ | WER ↓ | RTF ↓ |",
        "|------|--------|-------|-------|-------|",
    ]
    for i, s in enumerate(ranked, 1):
        lines.append(
            f"| {i} | **{s.engine}** | {s.avg_mos:.3f} | {s.avg_wer:.3f} | {s.avg_rtf:.3f} |"
        )

    return "\n".join(lines)


def build_claude_prompt(results: dict[str, EngineSummary]) -> str:
    if not results:
        return "_Run the pipeline first to generate a prompt._"

    lb = leaderboard_markdown(results)

    detail_lines = []
    for eng_name, summary in results.items():
        detail_lines.append(f"\n### {eng_name}")
        detail_lines.append(
            f"Average — MOS: {summary.avg_mos:.3f} | WER: {summary.avg_wer:.3f} | RTF: {summary.avg_rtf:.3f}"
        )
        detail_lines.append("")
        detail_lines.append("| Sentence | MOS | WER | RTF |")
        detail_lines.append("|----------|-----|-----|-----|")
        for r in summary.sentence_results:
            snippet = r.sentence[:50] + ("…" if len(r.sentence) > 50 else "")
            detail_lines.append(
                f"| {snippet} | {r.mos:.3f} | {r.wer:.3f} | {r.rtf:.3f} |"
            )

    prompt = f"""You are an expert in speech synthesis quality evaluation.
I ran an objective benchmark of {len(results)} TTS engine(s) on {len(next(iter(results.values())).sentence_results)} sentences.
Here are the results:

{lb}

Per-sentence breakdown:
{''.join(detail_lines)}

Please provide a qualitative analysis that covers:
1. Which engine sounds most natural and why, based on MOS scores.
2. Which engine is most intelligible and why, based on WER scores.
3. Trade-offs between quality and speed (RTF).
4. Any surprising or noteworthy patterns in the per-sentence data.
5. Your recommendation for different use cases (e.g., real-time vs. offline, accessibility tools, etc.).
"""
    return prompt
