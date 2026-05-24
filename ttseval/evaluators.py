"""Scoring utilities. Use get_wer_scorer() / get_mos_scorer() — never instantiate directly."""
import functools
import re


class WhisperWER:
    def __init__(self, model_size: str = "base"):
        from faster_whisper import WhisperModel
        self._model = WhisperModel(model_size, device="cpu", compute_type="int8")

    def score(self, wav_path: str, reference: str) -> float:
        segments, _ = self._model.transcribe(wav_path)
        hypothesis = " ".join(s.text.strip() for s in segments)
        return _word_error_rate(_normalise(reference), _normalise(hypothesis))


class UTMOSScorer:
    def __init__(self):
        try:
            import torch
            self._model = torch.hub.load(
                "tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True
            )
            self._model.eval()
            self._available = True
        except Exception:
            self._available = False

    def score(self, wav_path: str) -> float:
        if not self._available:
            return -1.0  # sentinel: MOS unavailable
        import torch
        import soundfile as sf

        wav, sr = sf.read(wav_path, dtype="float32")
        wav_t = torch.from_numpy(wav).unsqueeze(0)
        with torch.no_grad():
            mos = self._model(wav_t, sr)
        return float(mos)


@functools.lru_cache(maxsize=1)
def get_wer_scorer(model_size: str = "base") -> WhisperWER:
    return WhisperWER(model_size)


@functools.lru_cache(maxsize=1)
def get_mos_scorer() -> UTMOSScorer:
    return UTMOSScorer()


def _normalise(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _word_error_rate(ref: str, hyp: str) -> float:
    ref_words = ref.split()
    hyp_words = hyp.split()
    if not ref_words:
        return 0.0

    # Dynamic programming edit distance
    d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]
    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j

    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)

    return d[len(ref_words)][len(hyp_words)] / len(ref_words)
