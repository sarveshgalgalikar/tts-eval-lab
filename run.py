"""CLI entry point: python run.py --engines pyttsx3 kokoro --whisper base"""
import argparse
import sys

from ttseval.pipeline import run_pipeline, DEFAULT_SENTENCES
from ttseval.report import leaderboard_markdown, build_claude_prompt
from ttseval.tts_engines import ENGINES


def main():
    parser = argparse.ArgumentParser(
        description="TTS Eval Lab — benchmark local TTS engines"
    )
    parser.add_argument(
        "--engines",
        nargs="+",
        choices=list(ENGINES.keys()),
        default=["pyttsx3"],
        metavar="ENGINE",
        help=f"Engines to benchmark. Choices: {list(ENGINES.keys())}",
    )
    parser.add_argument(
        "--whisper",
        default="base",
        choices=["tiny", "base", "small", "medium", "large-v2"],
        help="Whisper model size for WER scoring (default: base)",
    )
    parser.add_argument(
        "--sentences",
        default=None,
        help="Path to a text file with one sentence per line (uses built-in set if omitted)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Write leaderboard markdown to this file",
    )
    parser.add_argument(
        "--prompt-output",
        default=None,
        dest="prompt_output",
        help="Write claude.ai prompt to this file",
    )
    args = parser.parse_args()

    sentences = None
    if args.sentences:
        with open(args.sentences) as f:
            sentences = [line.strip() for line in f if line.strip()]

    def _progress(eng, idx, total):
        print(f"  [{eng}] sentence {idx}/{total}", end="\r", flush=True)

    print(f"Running engines: {args.engines}")
    print(f"Sentences: {len(sentences) if sentences else len(DEFAULT_SENTENCES)}")
    print(f"Whisper model: {args.whisper}\n")

    try:
        results = run_pipeline(
            engine_names=args.engines,
            sentences=sentences,
            whisper_model=args.whisper,
            progress_cb=_progress,
        )
    except Exception as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        sys.exit(1)

    print()  # newline after progress line
    lb = leaderboard_markdown(results)
    print(lb)

    if args.output:
        with open(args.output, "w") as f:
            f.write(lb + "\n")
        print(f"\nLeaderboard written to {args.output}")

    prompt = build_claude_prompt(results)
    if args.prompt_output:
        with open(args.prompt_output, "w") as f:
            f.write(prompt)
        print(f"Claude prompt written to {args.prompt_output}")
    else:
        print("\n--- Claude Prompt (paste into claude.ai) ---")
        print(prompt)


if __name__ == "__main__":
    main()
