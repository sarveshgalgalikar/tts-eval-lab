"""TTS engine wrappers. Each subclass implements synthesize() and is registered in ENGINES."""
import time
from abc import ABC, abstractmethod
from pathlib import Path


class BaseTTS(ABC):
    name: str = ""

    @abstractmethod
    def synthesize(self, text: str, out_path: str) -> tuple[int, float]:
        """Synthesize text to a WAV file at out_path.

        Returns (sample_rate, elapsed_seconds).
        """


class Pyttsx3TTS(BaseTTS):
    name = "pyttsx3"

    def __init__(self):
        import pyttsx3
        self._engine = pyttsx3.init()

    def synthesize(self, text: str, out_path: str) -> tuple[int, float]:
        t0 = time.perf_counter()
        self._engine.save_to_file(text, out_path)
        self._engine.runAndWait()
        elapsed = time.perf_counter() - t0

        import soundfile as sf
        data, sr = sf.read(out_path)
        return sr, elapsed


class KokoroTTS(BaseTTS):
    name = "kokoro"

    def __init__(self):
        from kokoro import KPipeline
        self._pipeline = KPipeline(lang_code="a")  # American English

    def synthesize(self, text: str, out_path: str) -> tuple[int, float]:
        import numpy as np
        import soundfile as sf

        t0 = time.perf_counter()
        samples = []
        sr = 24000
        for _, _, audio in self._pipeline(text, voice="af_heart"):
            samples.append(audio)
        elapsed = time.perf_counter() - t0

        audio_np = np.concatenate(samples) if samples else np.zeros(sr)
        sf.write(out_path, audio_np, sr)
        return sr, elapsed


class CoquiXTTSTTS(BaseTTS):
    name = "xtts"

    def __init__(self):
        from TTS.api import TTS
        self._tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")

    def synthesize(self, text: str, out_path: str) -> tuple[int, float]:
        t0 = time.perf_counter()
        self._tts.tts_to_file(
            text=text,
            file_path=out_path,
            speaker="Claribel Dervla",
            language="en",
        )
        elapsed = time.perf_counter() - t0

        import soundfile as sf
        _, sr = sf.read(out_path)
        return sr, elapsed


ENGINES: dict[str, type[BaseTTS]] = {
    "pyttsx3": Pyttsx3TTS,
    "kokoro": KokoroTTS,
    "xtts": CoquiXTTSTTS,
}
