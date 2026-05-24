"""Gradio web UI — Hugging Face Spaces entry point."""
import gradio as gr

from ttseval.pipeline import run_pipeline, DEFAULT_SENTENCES
from ttseval.report import leaderboard_markdown, build_claude_prompt
from ttseval.tts_engines import ENGINES

ENGINE_CHOICES = list(ENGINES.keys()) or ["pyttsx3"]
WHISPER_CHOICES = ["tiny", "base", "small", "medium"]


def _run(engine_checkboxes, whisper_model, custom_sentences_text, progress=gr.Progress()):
    if not engine_checkboxes:
        return "⚠️ Select at least one engine.", ""

    sentences = None
    if custom_sentences_text.strip():
        sentences = [s.strip() for s in custom_sentences_text.strip().splitlines() if s.strip()]

    total_steps = len(engine_checkboxes) * (len(sentences) if sentences else len(DEFAULT_SENTENCES))
    step = 0

    def _cb(eng, idx, total):
        nonlocal step
        step += 1
        progress(step / total_steps, desc=f"{eng} — sentence {idx}/{total}")

    try:
        results = run_pipeline(
            engine_names=engine_checkboxes,
            sentences=sentences,
            whisper_model=whisper_model,
            progress_cb=_cb,
        )
    except Exception as exc:
        return f"❌ Error: {exc}", ""

    lb = leaderboard_markdown(results)
    prompt = build_claude_prompt(results)
    return lb, prompt


with gr.Blocks(title="TTS Eval Lab", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# 🎙️ TTS Eval Lab\n"
        "Benchmark free, local TTS engines on **MOS** (naturalness), **WER** (intelligibility), "
        "and **RTF** (speed). No paid APIs — everything runs on your machine."
    )

    with gr.Row():
        with gr.Column(scale=1):
            engine_cb = gr.CheckboxGroup(
                choices=ENGINE_CHOICES,
                value=["pyttsx3"],
                label="TTS Engines",
            )
            whisper_dd = gr.Dropdown(
                choices=WHISPER_CHOICES,
                value="base",
                label="Whisper model (WER scorer)",
            )
            custom_sentences = gr.Textbox(
                lines=6,
                placeholder="One sentence per line (leave blank to use the built-in set)",
                label="Custom sentences (optional)",
            )
            run_btn = gr.Button("▶ Run Benchmark", variant="primary")

        with gr.Column(scale=2):
            leaderboard_out = gr.Markdown(label="Leaderboard")
            claude_prompt_out = gr.Textbox(
                lines=18,
                label="Paste into claude.ai for qualitative analysis",
                show_copy_button=True,
            )

    run_btn.click(
        fn=_run,
        inputs=[engine_cb, whisper_dd, custom_sentences],
        outputs=[leaderboard_out, claude_prompt_out],
    )

    gr.Markdown(
        "---\n"
        "**Metrics:** MOS ↑ = UTMOS naturalness (1–5) · WER ↓ = Whisper word error rate · "
        "RTF ↓ = real-time factor (generation time ÷ audio duration)"
    )


if __name__ == "__main__":
    demo.launch()
