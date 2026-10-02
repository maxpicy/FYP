# intelligibility.py: Whisper WER, with the field's Whisper English normaliser as an option.

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"[^a-z']+")


def whisper_text_normalizer(model_id: str = "openai/whisper-base.en"):
    from transformers.models.whisper.english_normalizer import (
        BasicTextNormalizer, EnglishTextNormalizer)
    try:
        from transformers import WhisperTokenizer
        tok = WhisperTokenizer.from_pretrained(model_id)
        spell = getattr(tok, "english_spelling_normalizer", None)
        if spell:
            return EnglishTextNormalizer(spell)
    except Exception as e:
        logger.warning("Whisper spelling map unavailable for %s (%s); using BasicTextNormalizer",
                       model_id, e)
    return BasicTextNormalizer()


def _normalize(text: str) -> list[str]:
    text = text.lower().strip()
    text = _WORD_RE.sub(" ", text)
    return [w for w in text.split() if w]


def _edit_distance(a: list[str], b: list[str]) -> tuple[int, int, int, int]:
    n, m = len(a), len(b)
    if n == 0:
        return 0, 0, m, m
    if m == 0:
        return 0, n, 0, n

    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    op = np.zeros((n + 1, m + 1), dtype=np.int8)
    dp[:, 0] = np.arange(n + 1)
    dp[0, :] = np.arange(m + 1)
    op[:, 0] = 2
    op[0, :] = 3
    op[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                dp[i, j] = dp[i - 1, j - 1]
                op[i, j] = 0
            else:
                sub_cost = dp[i - 1, j - 1] + 1
                del_cost = dp[i - 1, j] + 1
                ins_cost = dp[i, j - 1] + 1
                best = min(sub_cost, del_cost, ins_cost)
                dp[i, j] = best
                op[i, j] = (
                    1 if best == sub_cost
                    else 2 if best == del_cost
                    else 3
                )

    i, j = n, m
    S = D = I = 0
    while i > 0 or j > 0:
        o = op[i, j]
        if o == 0:
            i -= 1
            j -= 1
        elif o == 1:
            S += 1
            i -= 1
            j -= 1
        elif o == 2:
            D += 1
            i -= 1
        else:
            I += 1
            j -= 1
    return S, D, I, int(dp[n, m])


def word_error_rate(reference: str, hypothesis: str,
                    text_normalizer=None) -> dict:
    if text_normalizer is not None:
        ref = text_normalizer(reference).split()
        hyp = text_normalizer(hypothesis).split()
    else:
        ref = _normalize(reference)
        hyp = _normalize(hypothesis)
    S, D, I, dist = _edit_distance(ref, hyp)
    n = len(ref)
    return {
        "wer": (dist / n) if n > 0 else float("nan"),
        "substitutions": S,
        "deletions": D,
        "insertions": I,
        "ref_word_count": n,
        "normalizer": "whisper" if text_normalizer is not None else "simple",
    }


class IntelligibilityScorer:
    DEFAULT_MODEL_ID = "openai/whisper-tiny.en"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "cuda",
        normalizer: str = "simple",
    ):
        if normalizer not in ("simple", "whisper"):
            raise ValueError(f"normalizer must be 'simple' or 'whisper', got {normalizer!r}")
        self.model_id = model_id
        self.device = device
        self.normalizer = normalizer
        self._model = None
        self._processor = None
        self._text_norm = None

    def _load(self):
        if self._model is not None:
            return
        from transformers import (
            WhisperForConditionalGeneration,
            WhisperProcessor,
        )
        logger.info("Loading Whisper %s on %s", self.model_id, self.device)
        self._processor = WhisperProcessor.from_pretrained(self.model_id)
        self._model = WhisperForConditionalGeneration.from_pretrained(
            self.model_id,
        ).to(self.device).eval()
        if self.normalizer == "whisper":
            self._text_norm = whisper_text_normalizer(self.model_id)

    def transcribe(self, wav_path: str | Path) -> str:
        import soundfile as sf
        import torch

        self._load()
        audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        if sr != 16000 and len(audio) > 0:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

        inputs = self._processor(
            audio, sampling_rate=16000, return_tensors="pt",
        ).to(self.device)
        gen_kw = {}
        if not self.model_id.endswith(".en"):
            gen_kw = {"language": "en", "task": "transcribe"}
        with torch.no_grad():
            ids = self._model.generate(**inputs, max_new_tokens=256, **gen_kw)
        text = self._processor.batch_decode(ids, skip_special_tokens=True)[0]
        return text.strip()

    def score(self, wav_path: str | Path, reference_text: str) -> dict:
        hypothesis = self.transcribe(wav_path)
        result = word_error_rate(reference_text, hypothesis,
                                 text_normalizer=self._text_norm)
        result["hypothesis"] = hypothesis
        result["reference"] = reference_text
        return result
