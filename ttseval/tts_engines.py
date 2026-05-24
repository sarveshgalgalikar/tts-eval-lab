"""TTS engine wrappers. Each subclass implements synthesize() and is registered in ENGINES."""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Type


class BaseTTS(ABC):
    name: str = ""

    @abstractmethod
    def synthesize(self, text: str, out_path: str) -> Tuple[int, float]:
        """Synthesize text to a WAV at out_path. Returns (sample_rate, elapsed_seconds)."""


class Pyttsx3TTS(BaseTTS):
    name = "pyttsx3"

    def __init__(self):
        import pyttsx3
        self._engine = pyttsx3.init()

    def synthesize(self, text: str, out_path: str) -> Tuple[int, float]:
        t0 = time.perf_counter()
        self._engine.save_to_file(text, out_path)
        self._engine.runAndWait()
        elapsed = time.perf_counter() - t0

        import soundfile as sf
        _, sr = sf.read(out_path)
        return sr, elapsed


class KokoroTTS(BaseTTS):
    name = "kokoro"

    def __init__(self):
        from kokoro import generate
        self._generate = generate

    def synthesize(self, text: str, out_path: str) -> Tuple[int, float]:
        import soundfile as sf

        t0 = time.perf_counter()
        audio, sr = self._generate(text, voice="af", speed=1.0)
        elapsed = time.perf_counter() - t0

        sf.write(out_path, audio, sr)
        return sr, elapsed


class CoquiXTTSTTS(BaseTTS):
    name = "xtts"

    def __init__(self):
        from TTS.api import TTS
        self._tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")

    def synthesize(self, text: str, out_path: str) -> Tuple[int, float]:
        t0 = time.perf_counter()
        self._tts.tts_to_file(
            text=text, file_path=out_path,
            speaker="Claribel Dervla", language="en",
        )
        elapsed = time.perf_counter() - t0
        import soundfile as sf
        _, sr = sf.read(out_path)
        return sr, elapsed


def _probe_engines() -> Dict[str, Type[BaseTTS]]:
    """Return only engines whose top-level import succeeds."""
    candidates: List[Tuple[str, Type[BaseTTS], str]] = [
        ("pyttsx3", Pyttsx3TTS,   "pyttsx3"),
        ("kokoro",  KokoroTTS,    "kokoro"),
        ("xtts",    CoquiXTTSTTS, "TTS"),
    ]
    available: Dict[str, Type[BaseTTS]] = {}
    for key, cls, pkg in candidates:
        try:
            __import__(pkg)
            available[key] = cls
        except ImportError:
            pass
    return available


ENGINES: Dict[str, Type[BaseTTS]] = _probe_engines()
