"""Generate all architecture and flow diagrams for TTS Eval Lab docs."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "images")
os.makedirs(OUT, exist_ok=True)

# ── Colour palette ────────────────────────────────────────────────────────────
C = {
    "bg":       "#0F1117",
    "panel":    "#1E2130",
    "border":   "#2E3250",
    "blue":     "#4C9BE8",
    "purple":   "#9B72CF",
    "green":    "#56C596",
    "orange":   "#F4A261",
    "red":      "#E76F51",
    "yellow":   "#E9C46A",
    "text":     "#E8EAF6",
    "subtext":  "#9FA8C4",
    "arrow":    "#5A6490",
}

def _fig(w, h):
    fig = plt.figure(figsize=(w, h), facecolor=C["bg"])
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    ax.set_facecolor(C["bg"])
    return fig, ax

def box(ax, x, y, w, h, color, label, sublabel=None, radius=0.3, alpha=1.0):
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle=f"round,pad=0,rounding_size={radius}",
                          facecolor=color, edgecolor=C["border"],
                          linewidth=1.5, alpha=alpha, zorder=3)
    ax.add_patch(rect)
    ty = y + h / 2 + (0.15 if sublabel else 0)
    ax.text(x + w/2, ty, label, ha="center", va="center",
            color=C["text"], fontsize=9, fontweight="bold", zorder=4)
    if sublabel:
        ax.text(x + w/2, y + h/2 - 0.2, sublabel, ha="center", va="center",
                color=C["subtext"], fontsize=7, zorder=4)

def arrow(ax, x1, y1, x2, y2, color=None, label=None):
    color = color or C["arrow"]
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=1.8, mutation_scale=14),
                zorder=5)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx+0.08, my, label, color=C["subtext"], fontsize=7,
                ha="left", va="center", zorder=6)

def title_text(ax, x, y, t, size=13):
    ax.text(x, y, t, color=C["text"], fontsize=size,
            fontweight="bold", ha="center", va="center", zorder=6)

def sub_text(ax, x, y, t, size=8, color=None):
    ax.text(x, y, t, color=color or C["subtext"], fontsize=size,
            ha="center", va="center", zorder=6)


# ══════════════════════════════════════════════════════════════════════════════
# 1. SYSTEM OVERVIEW FLOWCHART
# ══════════════════════════════════════════════════════════════════════════════
def diagram_system_overview():
    fig, ax = _fig(14, 9)
    title_text(ax, 7, 8.6, "TTS Eval Lab — System Overview", 14)
    sub_text(ax, 7, 8.2, "End-to-end flow from user input to leaderboard + Claude prompt")

    # Inputs column
    box(ax, 0.4, 6.2, 2.4, 0.8, C["purple"], "Engine Selection",  "pyttsx3 / kokoro / xtts")
    box(ax, 0.4, 5.0, 2.4, 0.8, C["purple"], "Sentence Set",       "built-in or custom")
    box(ax, 0.4, 3.8, 2.4, 0.8, C["purple"], "Whisper Model Size", "tiny / base / small …")

    # Pipeline box
    box(ax, 3.4, 4.2, 3.0, 3.2, C["blue"], "pipeline.run_pipeline()", sublabel=None, radius=0.4)
    ax.text(3.4+1.5, 4.2+1.6, "pipeline.run_pipeline()", ha="center", va="center",
            color=C["text"], fontsize=8.5, fontweight="bold", zorder=4)
    for i, step in enumerate(["① synthesize text → WAV",
                               "② score WER (Whisper)",
                               "③ score MOS (UTMOS)",
                               "④ compute RTF"]):
        ax.text(3.6, 4.2+1.15 - i*0.28, step, color=C["subtext"], fontsize=7.5, zorder=4)

    # Scorer boxes
    box(ax, 7.2, 5.6, 2.4, 0.7, C["green"],  "WhisperWER",   "faster-whisper (CPU)")
    box(ax, 7.2, 4.6, 2.4, 0.7, C["green"],  "UTMOSScorer",  "torch.hub UTMOS22")
    box(ax, 7.2, 3.6, 2.4, 0.7, C["orange"], "RTF",          "elapsed ÷ audio_duration")

    # Results
    box(ax, 10.2, 5.0, 3.2, 0.8, C["panel"], "EngineSummary", "avg_wer · avg_mos · avg_rtf")

    # Output column
    box(ax, 10.2, 3.4, 3.2, 0.8, C["yellow"], "Leaderboard",  "Markdown table (ranked)")
    box(ax, 10.2, 2.2, 3.2, 0.8, C["red"],    "Claude Prompt","paste → claude.ai")

    # Arrows — inputs to pipeline
    for y in [6.6, 5.4, 4.2]:
        arrow(ax, 2.8, y, 3.4, 5.8)

    # Pipeline to scorers
    arrow(ax, 6.4, 5.95, 7.2, 5.95)
    arrow(ax, 6.4, 5.95, 7.2, 4.95)
    arrow(ax, 6.4, 5.95, 7.2, 3.95)

    # Scorers to results
    for y in [5.95, 4.95, 3.95]:
        arrow(ax, 9.6, y, 10.2, 5.4)

    # Results to outputs
    arrow(ax, 11.8, 5.0, 11.8, 4.2)
    arrow(ax, 11.8, 3.4, 11.8, 3.0)

    # UI Layer banner
    rect = FancyBboxPatch((0.2, 1.0), 13.6, 0.9,
                          boxstyle="round,pad=0,rounding_size=0.3",
                          facecolor=C["panel"], edgecolor=C["border"],
                          linewidth=1.2, alpha=0.8, zorder=2)
    ax.add_patch(rect)
    ax.text(7, 1.45, "🖥  Gradio UI  (app.py)  ←→  CLI  (run.py)", ha="center",
            color=C["subtext"], fontsize=9, zorder=4)

    arrow(ax, 11.8, 2.2, 11.8, 1.9, color=C["yellow"])
    arrow(ax, 6.4, 4.2, 6.4, 1.9, color=C["blue"])

    fig.savefig(f"{OUT}/01_system_overview.png", dpi=150, bbox_inches="tight",
                facecolor=C["bg"])
    plt.close(fig)
    print("✓ 01_system_overview.png")


# ══════════════════════════════════════════════════════════════════════════════
# 2. MODULE ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════
def diagram_module_architecture():
    fig, ax = _fig(12, 8)
    title_text(ax, 6, 7.6, "Module Architecture & Dependencies", 13)

    modules = [
        # (x, y, w, h, color, name, file)
        (0.4,  5.4, 2.8, 1.1, C["blue"],   "tts_engines",  "ttseval/tts_engines.py"),
        (0.4,  3.6, 2.8, 1.1, C["green"],  "evaluators",   "ttseval/evaluators.py"),
        (4.2,  4.5, 2.8, 1.1, C["purple"], "pipeline",     "ttseval/pipeline.py"),
        (4.2,  2.8, 2.8, 1.1, C["orange"], "report",       "ttseval/report.py"),
        (8.0,  5.4, 3.2, 1.1, C["red"],    "app",          "app.py  (Gradio UI)"),
        (8.0,  3.8, 3.2, 1.1, C["yellow"], "run",          "run.py  (CLI)"),
        (4.2,  0.8, 2.8, 1.1, C["panel"],  "__init__",     "ttseval/__init__.py"),
    ]
    for x, y, w, h, color, name, fname in modules:
        box(ax, x, y, w, h, color, name, fname)

    # Dependency arrows
    deps = [
        (3.2, 6.0, 4.2, 5.3),   # engines → pipeline
        (3.2, 4.2, 4.2, 5.0),   # evaluators → pipeline
        (7.0, 5.0, 8.0, 5.9),   # pipeline → app
        (7.0, 5.0, 8.0, 4.3),   # pipeline → run
        (7.0, 3.4, 8.0, 4.1),   # report → run
        (7.0, 3.4, 8.0, 5.7),   # report → app (slight offset)
        (5.6, 4.5, 5.6, 3.9),   # pipeline → report (implicit via results)
        (4.2, 1.5, 3.2, 5.6),   # __init__ re-exports engines
        (4.2, 1.5, 3.2, 3.8),   # __init__ re-exports evaluators
        (5.6, 2.8, 5.6, 1.9),   # report ← __init__
    ]
    for x1, y1, x2, y2 in deps:
        arrow(ax, x1, y1, x2, y2)

    # External deps panel
    ext = [("pyttsx3", 0.4), ("kokoro", 1.2), ("coqui-tts", 2.0)]
    for label, ox in ext:
        box(ax, ox, 7.0, 0.75, 0.45, C["panel"], label, radius=0.15, alpha=0.7)
    ax.text(1.6, 7.6, "External TTS Libs", color=C["subtext"], fontsize=7.5,
            ha="center", zorder=4)

    ext2 = [("faster-whisper", 0.4), ("torch+UTMOS", 1.8)]
    for label, ox in ext2:
        box(ax, ox, 2.4, 1.3, 0.45, C["panel"], label, radius=0.15, alpha=0.7)
    ax.text(1.6, 2.1, "Scorer Libs", color=C["subtext"], fontsize=7.5,
            ha="center", zorder=4)

    fig.savefig(f"{OUT}/02_module_architecture.png", dpi=150, bbox_inches="tight",
                facecolor=C["bg"])
    plt.close(fig)
    print("✓ 02_module_architecture.png")


# ══════════════════════════════════════════════════════════════════════════════
# 3. PIPELINE EXECUTION FLOW (detailed step-by-step)
# ══════════════════════════════════════════════════════════════════════════════
def diagram_pipeline_flow():
    fig, ax = _fig(10, 13)
    title_text(ax, 5, 12.5, "Pipeline Execution — Step by Step", 13)

    steps = [
        (C["purple"], "START",              "run_pipeline(engine_names, sentences, …)"),
        (C["blue"],   "Load Models",        "get_wer_scorer()  ·  get_mos_scorer()\n(lru_cache — load once)"),
        (C["blue"],   "For each engine",    "Pyttsx3TTS()  /  KokoroTTS()  /  XTTS()"),
        (C["blue"],   "For each sentence",  "iterate over sentence list"),
        (C["orange"], "Synthesize",         "engine.synthesize(text, /tmp/xxx.wav)\n→ (sample_rate, elapsed_sec)"),
        (C["orange"], "Compute RTF",        "audio_duration = len(wav) / sample_rate\nrtf = elapsed / audio_duration"),
        (C["green"],  "Score WER",          "WhisperWER.score(wav_path, reference)\n→ word error rate  [0, ∞)"),
        (C["green"],  "Score MOS",          "UTMOSScorer.score(wav_path)\n→ naturalness  [1, 5]  or -1 if unavail"),
        (C["yellow"], "Store Result",       "SentenceResult(sentence, wer, mos, rtf)"),
        (C["yellow"], "Aggregate",          "avg_wer · avg_mos · avg_rtf  over all sentences"),
        (C["red"],    "EngineSummary",      "engine · avg_wer · avg_mos · avg_rtf · sentence_results"),
        (C["purple"], "RETURN",             "dict[engine_name → EngineSummary]"),
    ]

    ystart = 11.8
    ystep  = 0.88
    bw, bh = 6.2, 0.68

    for i, (color, label, detail) in enumerate(steps):
        y = ystart - i * ystep
        bx = (10 - bw) / 2
        box(ax, bx, y - bh/2, bw, bh, color, label, detail, radius=0.25)
        if i < len(steps) - 1:
            arrow(ax, 5, y - bh/2, 5, y - bh/2 - (ystep - bh))

    # Loop annotations
    ax.annotate("", xy=(1.7, ystart - 2*ystep + 0.1),
                xytext=(1.7, ystart - (len(steps)-3)*ystep - 0.1),
                arrowprops=dict(arrowstyle="-|>", color=C["blue"],
                                lw=1.5, connectionstyle="arc3,rad=-0.3"), zorder=5)
    ax.text(1.1, ystart - 4.5*ystep, "next\nengine", color=C["blue"],
            fontsize=7, ha="center", va="center")

    ax.annotate("", xy=(8.3, ystart - 3*ystep + 0.1),
                xytext=(8.3, ystart - (len(steps)-4)*ystep - 0.1),
                arrowprops=dict(arrowstyle="-|>", color=C["orange"],
                                lw=1.5, connectionstyle="arc3,rad=0.3"), zorder=5)
    ax.text(9.0, ystart - 5.5*ystep, "next\nsentence", color=C["orange"],
            fontsize=7, ha="center", va="center")

    fig.savefig(f"{OUT}/03_pipeline_flow.png", dpi=150, bbox_inches="tight",
                facecolor=C["bg"])
    plt.close(fig)
    print("✓ 03_pipeline_flow.png")


# ══════════════════════════════════════════════════════════════════════════════
# 4. TTS ENGINES MODULE
# ══════════════════════════════════════════════════════════════════════════════
def diagram_tts_engines():
    fig, ax = _fig(13, 8)
    title_text(ax, 6.5, 7.5, "TTS Engines Module  —  tts_engines.py", 13)

    # BaseTTS abstract
    box(ax, 4.5, 5.8, 4.0, 1.3, C["purple"], "BaseTTS  (ABC)",
        "name: str\nsynthesize(text, out_path) → (sr, elapsed)",
        radius=0.35)

    engines = [
        (0.4,  3.2, C["blue"],   "Pyttsx3TTS",     "name = 'pyttsx3'",
         ["pyttsx3.init()",  "engine.save_to_file()", "works offline, no download"]),
        (4.5,  3.2, C["green"],  "KokoroTTS",      "name = 'kokoro'",
         ["kokoro.generate()", "voice = 'af'", "82M params, Apache 2.0"]),
        (8.6,  3.2, C["orange"], "CoquiXTTSTTS",   "name = 'xtts'",
         ["TTS.api.TTS()", "multi-lingual XTTS-v2", "optional, ~1.8 GB"]),
    ]
    for x, y, color, name, sub, details in engines:
        box(ax, x, y, 3.8, 1.3, color, name, sub, radius=0.3)
        for j, d in enumerate(details):
            ax.text(x + 0.25, y - 0.28 - j*0.22, f"• {d}",
                    color=C["subtext"], fontsize=7, zorder=4)
        # Inheritance arrow
        ax.annotate("", xy=(x+1.9, y+1.3), xytext=(6.5, 5.8),
                    arrowprops=dict(arrowstyle="-|>", color=C["border"],
                                   lw=1.4, linestyle="dashed"), zorder=5)

    # _probe_engines box
    box(ax, 3.6, 0.5, 5.8, 1.0, C["panel"],
        "_probe_engines()  →  ENGINES dict",
        "tries __import__(pkg) for each candidate — silently skips unavailable libs",
        radius=0.3)

    arrow(ax, 2.3, 3.2, 4.0, 1.5)
    arrow(ax, 6.4, 3.2, 6.4, 1.5)
    arrow(ax, 10.5, 3.2, 8.0, 1.5)

    ax.text(6.5, 0.2, 'ENGINES = {"pyttsx3": Pyttsx3TTS, "kokoro": KokoroTTS, …}',
            color=C["yellow"], fontsize=8, ha="center", zorder=4,
            fontfamily="monospace")

    fig.savefig(f"{OUT}/04_tts_engines.png", dpi=150, bbox_inches="tight",
                facecolor=C["bg"])
    plt.close(fig)
    print("✓ 04_tts_engines.png")


# ══════════════════════════════════════════════════════════════════════════════
# 5. EVALUATORS MODULE
# ══════════════════════════════════════════════════════════════════════════════
def diagram_evaluators():
    fig, ax = _fig(13, 8.5)
    title_text(ax, 6.5, 8.1, "Evaluators Module  —  evaluators.py", 13)

    # WhisperWER
    box(ax, 0.5, 5.8, 5.5, 1.8, C["blue"], "WhisperWER", radius=0.35)
    for i, line in enumerate([
        "WhisperModel(model_size, device='cpu', compute_type='int8')",
        "transcribe(wav) → segments → hypothesis string",
        "_normalise(ref) + _normalise(hyp)",
        "→ _word_error_rate(ref, hyp)  [DP edit distance]",
    ]):
        ax.text(0.8, 7.3 - i*0.28, f"• {line}", color=C["subtext"], fontsize=7.5, zorder=4)

    # UTMOSScorer
    box(ax, 7.0, 5.8, 5.5, 1.8, C["green"], "UTMOSScorer", radius=0.35)
    for i, line in enumerate([
        "torch.hub.load('tarepan/SpeechMOS:v1.2.0', 'utmos22_strong')",
        "read WAV → float32 tensor",
        "model(wav_tensor, sample_rate)",
        "→ MOS score [1.0 – 5.0]  or  -1.0 if torch missing",
    ]):
        ax.text(7.3, 7.3 - i*0.28, f"• {line}", color=C["subtext"], fontsize=7.5, zorder=4)

    # lru_cache
    box(ax, 1.5, 4.2, 3.5, 0.75, C["panel"], "get_wer_scorer(model_size)",
        "@lru_cache — single WhisperWER instance", radius=0.25)
    box(ax, 8.0, 4.2, 3.5, 0.75, C["panel"], "get_mos_scorer()",
        "@lru_cache — single UTMOSScorer instance", radius=0.25)

    arrow(ax, 3.25, 5.8, 3.25, 4.95)
    arrow(ax, 9.75, 5.8, 9.75, 4.95)

    # WER internals
    box(ax, 0.5, 2.2, 5.5, 1.6, C["panel"], "WER Internals", radius=0.3)
    for i, line in enumerate([
        "_normalise(text)  →  lowercase · strip punctuation · collapse spaces",
        "_word_error_rate(ref, hyp)  →  DP Levenshtein on word list",
        "                            →  edits / len(ref_words)",
    ]):
        ax.text(0.8, 3.5 - i*0.28, line, color=C["subtext"], fontsize=7.5, zorder=4)

    # Metric legend
    for x, label, score, color in [
        (1.0, "MOS", "1.0 – 5.0  (↑ better)", C["green"]),
        (5.0, "WER", "0.0 – ∞    (↓ better)", C["blue"]),
        (9.0, "RTF", "computed in pipeline",   C["orange"]),
    ]:
        box(ax, x, 0.4, 3.2, 0.7, color, label, score, radius=0.2)

    arrow(ax, 3.25, 4.2, 3.25, 3.8)

    fig.savefig(f"{OUT}/05_evaluators.png", dpi=150, bbox_inches="tight",
                facecolor=C["bg"])
    plt.close(fig)
    print("✓ 05_evaluators.png")


# ══════════════════════════════════════════════════════════════════════════════
# 6. LEADERBOARD RANKING ALGORITHM
# ══════════════════════════════════════════════════════════════════════════════
def diagram_ranking():
    fig, ax = _fig(12, 7)
    title_text(ax, 6, 6.6, "Leaderboard Ranking Algorithm  —  report.py", 13)

    steps = [
        (1.0, 5.2, C["blue"],   "Raw Scores",
         "avg_wer, avg_mos, avg_rtf\nfor each engine"),
        (4.6, 5.2, C["purple"], "Normalise  [0, 1]",
         "_norm(val, lo, hi) = (val−lo) / (hi−lo)\napplied per metric across all engines"),
        (8.2, 5.2, C["orange"], "Composite Score",
         "rank_score =\n  MOS_norm − WER_norm − RTF_norm"),
    ]
    for x, y, color, label, detail in steps:
        box(ax, x, y, 2.8, 1.2, color, label, detail, radius=0.3)

    arrow(ax, 3.8, 5.8, 4.6, 5.8)
    arrow(ax, 7.4, 5.8, 8.2, 5.8)

    box(ax, 3.6, 3.2, 4.8, 1.0, C["green"], "Sort descending by rank_score",
        "Rank 1 = highest composite score", radius=0.3)
    arrow(ax, 9.6, 5.2, 6.0, 4.2)

    box(ax, 3.6, 1.6, 4.8, 1.0, C["yellow"], "Markdown Table Output",
        "| Rank | Engine | MOS ↑ | WER ↓ | RTF ↓ |", radius=0.3)
    arrow(ax, 6.0, 3.2, 6.0, 2.6)

    # Example bar chart
    ax2 = fig.add_axes([0.07, 0.04, 0.22, 0.22])
    ax2.set_facecolor(C["panel"])
    for spine in ax2.spines.values():
        spine.set_edgecolor(C["border"])
    ax2.tick_params(colors=C["subtext"], labelsize=6)
    engines = ["pyttsx3", "kokoro", "xtts"]
    scores  = [0.2, 0.75, 0.55]
    colors  = [C["blue"], C["green"], C["orange"]]
    bars = ax2.bar(engines, scores, color=colors, width=0.5)
    ax2.set_ylim(0, 1)
    ax2.set_title("rank_score example", color=C["subtext"], fontsize=6.5)
    ax2.yaxis.label.set_color(C["subtext"])

    fig.savefig(f"{OUT}/06_ranking_algorithm.png", dpi=150, bbox_inches="tight",
                facecolor=C["bg"])
    plt.close(fig)
    print("✓ 06_ranking_algorithm.png")


# ══════════════════════════════════════════════════════════════════════════════
# 7. DEPLOYMENT ARCHITECTURE (HF Spaces vs Local)
# ══════════════════════════════════════════════════════════════════════════════
def diagram_deployment():
    fig, ax = _fig(14, 8)
    title_text(ax, 7, 7.5, "Deployment Architecture", 13)

    # HF Spaces side
    rect = FancyBboxPatch((0.3, 0.5), 6.0, 6.5,
                          boxstyle="round,pad=0,rounding_size=0.4",
                          facecolor=C["panel"], edgecolor=C["blue"],
                          linewidth=2, alpha=0.5, zorder=1)
    ax.add_patch(rect)
    ax.text(3.3, 6.7, "🤗  Hugging Face Spaces", color=C["blue"],
            fontsize=10, fontweight="bold", ha="center", zorder=4)

    box(ax, 0.7, 5.4, 5.2, 0.7, C["panel"], "python:3.13 Docker base image", radius=0.2)
    box(ax, 0.7, 4.5, 5.2, 0.7, C["blue"],  "packages.txt  →  apt-get install espeak-ng", radius=0.2)
    box(ax, 0.7, 3.6, 5.2, 0.7, C["blue"],  "requirements.txt  →  pip install …", radius=0.2)
    box(ax, 0.7, 2.7, 5.2, 0.7, C["green"], "app.py launches  →  Gradio on :7860", radius=0.2)
    box(ax, 0.7, 1.6, 5.2, 0.75, C["panel"],
        "Available engine: pyttsx3 only",
        "kokoro & XTTS need Python <3.13 or separate container", radius=0.2)

    arrow(ax, 3.3, 5.4, 3.3, 5.2)
    arrow(ax, 3.3, 4.5, 3.3, 4.3)
    arrow(ax, 3.3, 3.6, 3.3, 3.4)
    arrow(ax, 3.3, 2.7, 3.3, 2.375)

    # Local side
    rect2 = FancyBboxPatch((7.7, 0.5), 6.0, 6.5,
                           boxstyle="round,pad=0,rounding_size=0.4",
                           facecolor=C["panel"], edgecolor=C["green"],
                           linewidth=2, alpha=0.5, zorder=1)
    ax.add_patch(rect2)
    ax.text(10.7, 6.7, "💻  Local Machine", color=C["green"],
            fontsize=10, fontweight="bold", ha="center", zorder=4)

    box(ax, 8.1, 5.4, 5.2, 0.7, C["panel"], "Python 3.10 – 3.12  (venv / conda)", radius=0.2)
    box(ax, 8.1, 4.5, 5.2, 0.7, C["green"],  "pip install -r requirements.txt", radius=0.2)
    box(ax, 8.1, 3.6, 5.2, 0.7, C["green"],  "python app.py  →  localhost:7860", radius=0.2)
    box(ax, 8.1, 2.7, 5.2, 0.7, C["orange"], "python run.py --engines kokoro pyttsx3", radius=0.2)
    box(ax, 8.1, 1.6, 5.2, 0.75, C["panel"],
        "All engines available",
        "kokoro / XTTS / pyttsx3  +  full MOS scoring", radius=0.2)

    arrow(ax, 10.7, 5.4, 10.7, 5.2)
    arrow(ax, 10.7, 4.5, 10.7, 4.3)
    arrow(ax, 10.7, 3.6, 10.7, 3.4)
    arrow(ax, 10.7, 2.7, 10.7, 2.375)

    # GitHub sync arrow
    ax.annotate("", xy=(7.7, 4.0), xytext=(6.3, 4.0),
                arrowprops=dict(arrowstyle="<->", color=C["yellow"],
                                lw=2, mutation_scale=14), zorder=5)
    ax.text(7.0, 4.35, "git push", color=C["yellow"], fontsize=8,
            ha="center", va="center", zorder=6)
    ax.text(7.0, 3.75, "GitHub → HF", color=C["yellow"], fontsize=7,
            ha="center", va="center", zorder=6)

    fig.savefig(f"{OUT}/07_deployment.png", dpi=150, bbox_inches="tight",
                facecolor=C["bg"])
    plt.close(fig)
    print("✓ 07_deployment.png")


# ══════════════════════════════════════════════════════════════════════════════
# 8. DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════
def diagram_data_structures():
    fig, ax = _fig(13, 7)
    title_text(ax, 6.5, 6.6, "Core Data Structures", 13)

    # SentenceResult
    box(ax, 0.4, 4.0, 3.8, 2.2, C["blue"], "SentenceResult", radius=0.35)
    for i, (fname, ftype, note) in enumerate([
        ("sentence", "str",   "original text"),
        ("wer",      "float", "0.0 = perfect"),
        ("mos",      "float", "1–5, -1 = N/A"),
        ("rtf",      "float", "< 1.0 = faster than real-time"),
    ]):
        y = 5.7 - i * 0.38
        ax.text(0.7, y, fname, color=C["yellow"],   fontsize=8, zorder=4, fontfamily="monospace")
        ax.text(2.2, y, ftype,  color=C["subtext"], fontsize=8, zorder=4)
        ax.text(2.8, y, note,   color=C["subtext"], fontsize=7, zorder=4)

    # EngineSummary
    box(ax, 4.8, 3.0, 4.0, 3.2, C["purple"], "EngineSummary", radius=0.35)
    for i, (fname, ftype, note) in enumerate([
        ("engine",           "str",            "engine key name"),
        ("avg_wer",          "float",          "mean over sentences"),
        ("avg_mos",          "float",          "mean over sentences"),
        ("avg_rtf",          "float",          "mean over sentences"),
        ("sentence_results", "List[Sentence…]","one per sentence"),
    ]):
        y = 5.8 - i * 0.42
        ax.text(5.1, y, fname,  color=C["yellow"],   fontsize=8, zorder=4, fontfamily="monospace")
        ax.text(7.1, y, ftype,  color=C["subtext"],  fontsize=7.5, zorder=4)
        ax.text(7.1, y-0.18, note, color=C["subtext"], fontsize=6.5, zorder=4)

    # dict return
    box(ax, 9.2, 4.0, 3.4, 2.2, C["green"], "Pipeline Output", radius=0.35)
    ax.text(9.5, 5.7, "dict[str, EngineSummary]",
            color=C["yellow"], fontsize=7.5, zorder=4, fontfamily="monospace")
    ax.text(9.5, 5.3, '"pyttsx3" → EngineSummary',
            color=C["subtext"], fontsize=7.5, zorder=4, fontfamily="monospace")
    ax.text(9.5, 4.95, '"kokoro"  → EngineSummary',
            color=C["subtext"], fontsize=7.5, zorder=4, fontfamily="monospace")
    ax.text(9.5, 4.55, '"xtts"    → EngineSummary',
            color=C["subtext"], fontsize=7.5, zorder=4, fontfamily="monospace")

    # Composition arrow
    arrow(ax, 4.2, 5.1, 4.8, 5.0)
    ax.text(4.45, 5.35, "1..*", color=C["subtext"], fontsize=8, zorder=4)

    # Output arrow
    arrow(ax, 8.8, 5.1, 9.2, 5.1)

    # Downstream
    box(ax, 3.0, 1.0, 3.0, 0.8, C["orange"], "leaderboard_markdown(results)",
        "→ Markdown string", radius=0.2)
    box(ax, 7.0, 1.0, 3.0, 0.8, C["red"],    "build_claude_prompt(results)",
        "→ prompt string", radius=0.2)
    arrow(ax, 10.0, 4.0, 8.5, 1.8)
    arrow(ax, 10.0, 4.0, 4.5, 1.8)

    fig.savefig(f"{OUT}/08_data_structures.png", dpi=150, bbox_inches="tight",
                facecolor=C["bg"])
    plt.close(fig)
    print("✓ 08_data_structures.png")


if __name__ == "__main__":
    diagram_system_overview()
    diagram_module_architecture()
    diagram_pipeline_flow()
    diagram_tts_engines()
    diagram_evaluators()
    diagram_ranking()
    diagram_deployment()
    diagram_data_structures()
    print(f"\nAll diagrams saved to {OUT}/")
