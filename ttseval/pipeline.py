"""Pipeline: run engines × sentences, aggregate per-engine summaries."""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from .tts_engines import ENGINES, BaseTTS
from .evaluators import get_wer_scorer, get_mos_scorer

DEFAULT_SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "She sells seashells by the seashore.",
    "How much wood would a woodchuck chuck if a woodchuck could chuck wood?",
    "Peter Piper picked a peck of pickled peppers.",
    "The rain in Spain stays mainly in the plain.",
    "To be or not to be, that is the question.",
    "All that glitters is not gold.",
    "A journey of a thousand miles begins with a single step.",
]


@dataclass
class SentenceResult:
    sentence: str
    wer: float
    mos: float
    rtf: float


@dataclass
class EngineSummary:
    engine: str
    avg_wer: float
    avg_mos: float
    avg_rtf: float
    sentence_results: List[SentenceResult] = field(default_factory=list)


def run_pipeline(
    engine_names: List[str],
    sentences: Optional[List[str]] = None,
    whisper_model: str = "base",
    progress_cb: Optional[Callable] = None,
) -> dict:
    """Run every requested engine over every sentence.

    progress_cb(engine_name, sentence_idx, total) is called after each WAV is scored.
    """
    if sentences is None:
        sentences = DEFAULT_SENTENCES

    wer_scorer = get_wer_scorer(whisper_model)
    mos_scorer = get_mos_scorer()

    results = {}

    for eng_name in engine_names:
        if eng_name not in ENGINES:
            raise ValueError(f"Unknown engine '{eng_name}'. Available: {list(ENGINES)}")

        engine: BaseTTS = ENGINES[eng_name]()
        sent_results: List[SentenceResult] = []

        for idx, sentence in enumerate(sentences):
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                wav_path = tmp.name

            try:
                sr, elapsed = engine.synthesize(sentence, wav_path)

                import soundfile as sf
                data, _ = sf.read(wav_path)
                audio_duration = len(data) / sr
                rtf = elapsed / audio_duration if audio_duration > 0 else 0.0

                wer = wer_scorer.score(wav_path, sentence)
                mos = mos_scorer.score(wav_path)

                sent_results.append(SentenceResult(sentence=sentence, wer=wer, mos=mos, rtf=rtf))
            finally:
                os.unlink(wav_path)

            if progress_cb:
                progress_cb(eng_name, idx + 1, len(sentences))

        avg_wer = sum(r.wer for r in sent_results) / len(sent_results)
        avg_mos = sum(r.mos for r in sent_results) / len(sent_results)
        avg_rtf = sum(r.rtf for r in sent_results) / len(sent_results)

        results[eng_name] = EngineSummary(
            engine=eng_name,
            avg_wer=avg_wer,
            avg_mos=avg_mos,
            avg_rtf=avg_rtf,
            sentence_results=sent_results,
        )

    return results
