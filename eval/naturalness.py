# naturalness.py: UTMOS22 naturalness (torch.hub tarepan/SpeechMOS).

import logging
from typing import Union

import numpy as np

logger = logging.getLogger(__name__)

UTMOS_REPO = "tarepan/SpeechMOS:v1.2.0"
UTMOS_ENTRY = "utmos22_strong"


class NaturalnessScorer:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            import torch
            logger.info("Loading UTMOS (%s)...", UTMOS_REPO)
            self._model = torch.hub.load(
                UTMOS_REPO, UTMOS_ENTRY, trust_repo=True).to(self.device).eval()

    def score(self, wav: Union[str, np.ndarray, "np.ndarray"],
              sr: int = None) -> dict:
        import torch
        self._ensure_model()
        if isinstance(wav, str):
            import soundfile as sf
            audio, sr = sf.read(wav, dtype="float32")
        else:
            if sr is None:
                raise ValueError("sr required when passing an array")
            audio = np.asarray(wav, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=-1)
        if audio.size == 0:
            return {"utmos": float("nan")}
        x = torch.from_numpy(np.ascontiguousarray(audio))[None].to(self.device)
        with torch.no_grad():
            val = float(self._model(x, sr).item())
        return {"utmos": val}
