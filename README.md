---
title: TTS Eval Lab
emoji: 🎙️
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
license: mit
---

# TTS Eval Lab

A free, fully-local pipeline that benchmarks text-to-speech engines on three objective metrics and produces a leaderboard plus a ready-to-paste prompt for qualitative analysis on [claude.ai](https://claude.ai).

**No paid APIs. Everything runs on your machine (or on the free HF CPU tier).**

## Metrics

| Metric | Direction | Description |
|--------|-----------|-------------|
| MOS ↑ | Higher is better | UTMOS naturalness prediction (1–5 scale) |
| WER ↓ | Lower is better | Whisper word-error rate (intelligibility) |
| RTF ↓ | Lower is better | Real-time factor = generation time ÷ audio duration |

## Supported Engines

| Engine | Key | Notes |
|--------|-----|-------|
| pyttsx3 | `pyttsx3` | Offline, no model download. Needs `espeak` on Linux. |
| Kokoro | `kokoro` | High-quality, ~82M params, Apache 2.0 |
| XTTS-v2 (Coqui) | `xtts` | Multi-lingual, ~1.8GB model download |

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Fastest smoke test (no model download)
python run.py --engines pyttsx3

# Full benchmark
python run.py --engines kokoro pyttsx3 --whisper base

# Web UI
python app.py
```

> **Linux note:** pyttsx3 requires espeak: `sudo apt-get install espeak`

## CLI Options

```
python run.py --engines pyttsx3 kokoro xtts
              --whisper base         # tiny | base | small | medium | large-v2
              --sentences my_sents.txt  # one sentence per line (optional)
              --output leaderboard.md
              --prompt-output claude_prompt.txt
```

## Adding a New Engine

1. Subclass `BaseTTS` in `ttseval/tts_engines.py`.
2. Lazy-import its library inside `__init__`.
3. Implement `synthesize(text, out_path) -> (sample_rate, elapsed_seconds)`, writing a WAV via `soundfile`.
4. Register it in `ENGINES`.
5. Add it to `ENGINE_CHOICES` in `app.py` and to this README table.

## How the Claude Step Works

After the benchmark, `report.build_claude_prompt()` emits a structured text block with the leaderboard and per-sentence data. Copy it and paste into [claude.ai](https://claude.ai) for a qualitative analysis covering naturalness, intelligibility, speed trade-offs, and use-case recommendations.

## Architecture

```
ttseval/
  tts_engines.py   # synthesis wrappers — each subclasses BaseTTS
  evaluators.py    # WhisperWER (intelligibility) + UTMOSScorer (naturalness)
  pipeline.py      # runs engines × sentences, aggregates per-engine summaries
  report.py        # markdown leaderboard + build_claude_prompt()
app.py             # Gradio UI (Hugging Face Spaces entry point)
run.py             # CLI entry point
```

## License

MIT
