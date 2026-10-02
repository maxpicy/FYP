# plan_extract.py: The word-level prosody plan: pitch / duration / energy per word, quantised to bins, and
# its token form.

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

PLAN_SPEC_VERSION = "p2cot-plan-v1"

try:
    import config as _cfg
except Exception:
    _cfg = None


def _cfg_get(name: str, default):
    return getattr(_cfg, name, default) if _cfg is not None else default

FRAME_RATE_HZ = 21.53
N_PITCH_BINS = int(_cfg_get("PITCH_BINS", 16))
N_DUR_BINS = int(_cfg_get("DURATION_BINS", 16))
N_ENERGY_BINS = int(_cfg_get("ENERGY_BINS", 8))
PITCH_UNVOICED_BIN = N_PITCH_BINS

MIN_VOICED_FRAC = 0.25
MIN_VOICED_FRAMES = 2

MIN_WORDS_FOR_SPEAKER_BINS = 48

DEFAULT_MIN_WORD_CONF = 0.50
DEFAULT_MIN_COVERAGE = 0.95

HARD_FLAGS = frozenset({"low_coverage", "low_conf"})

DUR_EDGES_FRAMES = np.concatenate([
    np.array([1.5, 2.5, 3.5, 4.5]),
    np.geomspace(5.5, 40.0, N_DUR_BINS - 5),
])

GLOBAL_PITCH_EDGES_OCT = np.linspace(-0.7, 0.7, N_PITCH_BINS - 1)
GLOBAL_ENERGY_EDGES_DB = np.linspace(-15.0, 9.0, N_ENERGY_BINS - 1)

PLAN_START_TOKEN = _cfg_get("PLAN_START_TOKEN", "<PLAN>")
PLAN_END_TOKEN = _cfg_get("PLAN_END_TOKEN", "</PLAN>")
PLAN_WORD_TOKEN = _cfg_get("PLAN_WORD_SEP_TOKEN", "[PW]")
_P_PRE = _cfg_get("PLAN_PITCH_PREFIX", "[P:")
_D_PRE = _cfg_get("PLAN_DURATION_PREFIX", "[D:")
_E_PRE = _cfg_get("PLAN_ENERGY_PREFIX", "[E:")
_SUF = _cfg_get("PLAN_TOKEN_SUFFIX", "]")

PITCH_UNVOICED_SYMBOL = "U"

_PLAN_TOKEN_RE = re.compile("|".join([
    f"(?P<w>{re.escape(PLAN_WORD_TOKEN)})",
    f"{re.escape(_P_PRE)}(?P<p>[0-9]+|{PITCH_UNVOICED_SYMBOL}){re.escape(_SUF)}",
    f"{re.escape(_D_PRE)}(?P<d>[0-9]+){re.escape(_SUF)}",
    f"{re.escape(_E_PRE)}(?P<e>[0-9]+){re.escape(_SUF)}",
]))


def plan_pitch_token(b: int) -> str:
    if b == PITCH_UNVOICED_BIN:
        return f"{_P_PRE}{PITCH_UNVOICED_SYMBOL}{_SUF}"
    return f"{_P_PRE}{int(b)}{_SUF}"


def plan_dur_token(b: int) -> str:
    return f"{_D_PRE}{int(b)}{_SUF}"


def plan_energy_token(b: int) -> str:
    return f"{_E_PRE}{int(b)}{_SUF}"


def plan_vocab() -> list[str]:
    return ([PLAN_START_TOKEN, PLAN_END_TOKEN, PLAN_WORD_TOKEN]
            + [plan_pitch_token(i) for i in range(N_PITCH_BINS)]
            + [plan_dur_token(i) for i in range(N_DUR_BINS)]
            + [plan_energy_token(i) for i in range(N_ENERGY_BINS)])


def verify_config_vocab() -> Optional[str]:
    if _cfg is None or not hasattr(_cfg, "get_plan_tokens"):
        return None
    cfg_tokens = set(_cfg.get_plan_tokens())
    missing = [t for t in plan_vocab() if t not in cfg_tokens]
    if missing:
        raise PlanExtractError(
            "config.get_plan_tokens() does not cover the plan token surface "
            f"emitted by scripts/plan_extract.py; missing {missing[:8]}"
            f"{'...' if len(missing) > 8 else ''}. One of the two must change — "
            "a mismatch tokenizes plans to <unk> and trains silently.")
    return "ok"


class PlanExtractError(Exception):
    pass


class PlanAlignmentError(PlanExtractError):
    pass


class PlanQuantiseError(PlanExtractError):
    pass


class PlanTokenError(PlanExtractError):
    pass


@dataclass
class PlanResult:
    words: list[dict]
    coverage: float
    time_coverage: float
    mean_conf: float
    min_conf: float
    duration_s: float
    n_words_expected: int
    ok: bool
    flags: list[str] = field(default_factory=list)
    reason: str = ""

    def __iter__(self):
        return iter(self.words)

    def __len__(self) -> int:
        return len(self.words)

    def __getitem__(self, i):
        return self.words[i]

_F0_BACKEND_WARNED = False


def compute_f0(
    audio: np.ndarray,
    sr: int,
    *,
    f0_min_hz: float = 60.0,
    f0_max_hz: float = 500.0,
    frame_period_ms: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    global _F0_BACKEND_WARNED
    x = np.asarray(audio, dtype=np.float64).ravel()
    if x.size == 0:
        return np.zeros(0), np.zeros(0)
    try:
        import pyworld as pw
    except Exception as e:
        if not _F0_BACKEND_WARNED:
            print(f"WARNING plan_extract: pyworld unavailable ({e}); falling back "
                  f"to librosa.pyin. F0 values will differ slightly from the "
                  f"pyworld reference — do not mix backends within one corpus.",
                  file=sys.stderr, flush=True)
            _F0_BACKEND_WARNED = True
        return _f0_librosa(x, sr, f0_min_hz, f0_max_hz, frame_period_ms)

    f0, t = pw.dio(np.ascontiguousarray(x), sr, f0_floor=float(f0_min_hz),
                   f0_ceil=float(f0_max_hz), frame_period=float(frame_period_ms))
    f0 = pw.stonemask(np.ascontiguousarray(x), f0, t, sr)
    return np.asarray(f0, dtype=np.float64), np.asarray(t, dtype=np.float64)


def _f0_librosa(x, sr, f0_min_hz, f0_max_hz, frame_period_ms):
    import librosa
    hop = max(1, int(round(sr * frame_period_ms / 1000.0)))
    frame_length = max(4 * hop, 1024)
    f0, _voiced, _p = librosa.pyin(
        x.astype(np.float32), fmin=float(f0_min_hz), fmax=float(f0_max_hz), sr=sr,
        frame_length=frame_length, hop_length=hop,
        no_trough_prob=0.1, switch_prob=0.1,
    )
    f0 = np.nan_to_num(np.asarray(f0, dtype=np.float64), nan=0.0)
    t = np.arange(f0.size, dtype=np.float64) * hop / sr
    return f0, t


def _as_span(s) -> tuple[str, float, float, float]:
    if isinstance(s, dict):
        return (str(s["word"]), float(s["t_start"]), float(s["t_end"]),
                float(s.get("conf", 1.0)))
    s = tuple(s)
    if len(s) == 3:
        return (str(s[0]), float(s[1]), float(s[2]), 1.0)
    if len(s) == 4:
        return (str(s[0]), float(s[1]), float(s[2]), float(s[3]))
    raise PlanExtractError(
        f"span must be (word, t_start, t_end[, conf]) or a dict; got {s!r}")


def word_features_from_spans(
    audio: np.ndarray,
    sr: int,
    spans: Sequence,
    *,
    frame_rate: float = FRAME_RATE_HZ,
    f0_min_hz: float = 60.0,
    f0_max_hz: float = 500.0,
    f0_frame_period_ms: float = 10.0,
    f0_track: Optional[tuple[np.ndarray, np.ndarray]] = None,
) -> list[dict]:
    audio = np.asarray(audio, dtype=np.float64).ravel()
    if sr <= 0:
        raise PlanExtractError(f"sr must be positive, got {sr}")
    if frame_rate <= 0:
        raise PlanExtractError(f"frame_rate must be positive, got {frame_rate}")
    parsed = [_as_span(s) for s in spans]
    for w, t0, t1, _c in parsed:
        if not (t1 > t0):
            raise PlanExtractError(
                f"word {w!r} has non-positive duration ({t0:.4f}..{t1:.4f}s); the "
                f"aligner produced an invalid span and the plan for this row "
                f"would be garbage")

    if f0_track is None:
        f0, f0_t = compute_f0(audio, sr, f0_min_hz=f0_min_hz, f0_max_hz=f0_max_hz,
                              frame_period_ms=f0_frame_period_ms)
    else:
        f0, f0_t = f0_track
    f0 = np.asarray(f0, dtype=np.float64)
    f0_t = np.asarray(f0_t, dtype=np.float64)

    voiced_all = f0[f0 > 0]
    if voiced_all.size > 3:
        med = float(np.median(voiced_all))
        lo, hi = 0.5 * med, 2.0 * med
    else:
        lo, hi = 0.0, float("inf")

    out: list[dict] = []
    n = audio.size
    for i, (w, t0, t1, conf) in enumerate(parsed):
        i0 = max(0, min(n, int(round(t0 * sr))))
        i1 = max(i0, min(n, int(round(t1 * sr))))
        seg = audio[i0:i1]
        rms = float(np.sqrt(np.mean(seg ** 2))) if seg.size else 0.0

        sel = (f0_t >= t0) & (f0_t < t1)
        wf0 = f0[sel]
        n_frames = int(wf0.size)
        voiced = wf0[wf0 > 0]
        voiced_frac = float(voiced.size / n_frames) if n_frames else 0.0
        gated = voiced[(voiced >= lo) & (voiced <= hi)]
        if (gated.size >= MIN_VOICED_FRAMES and voiced_frac >= MIN_VOICED_FRAC):
            f0_mean = float(np.exp(np.mean(np.log(gated))))
        else:
            f0_mean = float("nan")

        dur_s = float(t1 - t0)
        out.append({
            "word": w,
            "idx": i,
            "t_start": round(float(t0), 6),
            "t_end": round(float(t1), 6),
            "dur_s": dur_s,
            "dur_frames": max(1, int(round(dur_s * frame_rate))),
            "f0_mean_hz": f0_mean,
            "voiced_frac": voiced_frac,
            "n_voiced_frames": int(gated.size),
            "energy_rms": rms,
            "conf": float(conf),
            "silent": bool(rms < 1e-6),
        })
    return out

_MMS_FA_CHARS = set("abcdefghijklmnopqrstuvwxyz'")


def normalise_transcript(text: str) -> tuple[list[str], list[str]]:
    words, dropped = [], []
    for raw in str(text).split():
        s = unicodedata.normalize("NFKD", raw).lower()
        s = s.replace("’", "'").replace("ʼ", "'").replace("`", "'")
        s = "".join(c for c in s if c in _MMS_FA_CHARS)
        s = s.strip("'")
        if s:
            words.append(s)
        else:
            dropped.append(raw)
    return words, dropped


class PlanExtractor:
    def __init__(
        self,
        *,
        frame_rate: float = FRAME_RATE_HZ,
        device: str = "cpu",
        f0_min_hz: float = 60.0,
        f0_max_hz: float = 500.0,
        f0_frame_period_ms: float = 10.0,
        min_word_conf: float = DEFAULT_MIN_WORD_CONF,
        min_coverage: float = DEFAULT_MIN_COVERAGE,
        align_fn: Optional[Callable[[np.ndarray, list[str]], list]] = None,
    ):
        self.frame_rate = frame_rate
        self.device = device
        self.f0_min_hz = f0_min_hz
        self.f0_max_hz = f0_max_hz
        self.f0_frame_period_ms = f0_frame_period_ms
        self.min_word_conf = min_word_conf
        self.min_coverage = min_coverage
        self.align_fn = align_fn
        self._bundle = None
        self._model = None
        self._tokenizer = None
        self._aligner = None

    @property
    def align_sr(self) -> int:
        return 16000

    def _ensure_aligner(self):
        if self._aligner is not None:
            return
        import torch
        import torchaudio
        self._bundle = torchaudio.pipelines.MMS_FA
        if self._bundle.sample_rate != self.align_sr:
            raise PlanAlignmentError(
                f"MMS_FA expects {self._bundle.sample_rate} Hz but this extractor "
                f"resamples to {self.align_sr}; word times would be scaled wrong")
        self._model = self._bundle.get_model(with_star=False).to(self.device).eval()
        self._tokenizer = self._bundle.get_tokenizer()
        self._aligner = self._bundle.get_aligner()
        self._torch = torch

    def align_words(self, audio: np.ndarray, words: list[str]) -> list[tuple]:
        if self.align_fn is not None:
            return [_as_span(s) for s in self.align_fn(audio, words)]
        if not words:
            raise PlanAlignmentError("no alignable words in the transcript")
        self._ensure_aligner()
        torch = self._torch
        wav = torch.as_tensor(np.asarray(audio, dtype=np.float32)).reshape(1, -1)
        wav = wav.to(self.device)
        with torch.inference_mode():
            emission, _ = self._model(wav)
        tokens = self._tokenizer(words)
        n_tok = sum(len(t) for t in tokens)
        if emission.shape[1] < n_tok:
            raise PlanAlignmentError(
                f"emission has {emission.shape[1]} frames for {n_tok} characters "
                f"({len(words)} words, {audio.size / self.align_sr:.2f}s audio): "
                f"transcript is too long for the audio (mismatched pair?)")
        try:
            spans = self._aligner(emission[0], tokens)
        except Exception as e:
            raise PlanAlignmentError(
                f"forced_align failed on {len(words)} words / {n_tok} chars: {e}")
        ratio = wav.shape[1] / emission.shape[1]
        out = []
        for w, ts in zip(words, spans):
            if not ts:
                raise PlanAlignmentError(f"aligner returned no span for word {w!r}")
            t0 = ratio * ts[0].start / self.align_sr
            t1 = ratio * ts[-1].end / self.align_sr
            tot = sum(s.end - s.start for s in ts)
            conf = (sum(s.score * (s.end - s.start) for s in ts) / tot) if tot else 0.0
            out.append((w, float(t0), float(t1), float(conf)))
        return out

    def extract(
        self,
        wav: Any,
        sr: Optional[int] = None,
        transcript: str = "",
        *,
        strict: bool = False,
    ) -> PlanResult:
        audio, sr = _load_audio(wav, sr, target_sr=self.align_sr)
        duration_s = audio.size / float(sr)
        words, dropped = normalise_transcript(transcript)
        n_expected = len(words) + len(dropped)
        if n_expected == 0:
            raise PlanExtractError("empty transcript: nothing to align")
        if not words:
            raise PlanExtractError(
                f"transcript has {n_expected} words but none survive MMS_FA "
                f"normalisation (digits/non-Latin only?): {transcript!r}")
        if duration_s < 0.05:
            raise PlanExtractError(f"audio is {duration_s:.4f}s — too short to align")
        if float(np.max(np.abs(audio))) < 1e-6:
            raise PlanExtractError("audio is digital silence — no plan is extractable")

        spans = self.align_words(audio, words)
        over = [s for s in spans if s[2] > duration_s + 0.05]
        if over:
            raise PlanAlignmentError(
                f"{len(over)} word span(s) end past the {duration_s:.3f}s audio "
                f"(first: {over[0][0]!r} -> {over[0][2]:.3f}s) — the aligner's "
                f"frame-to-second conversion is wrong for this row")
        feats = word_features_from_spans(
            audio, sr, spans, frame_rate=self.frame_rate,
            f0_min_hz=self.f0_min_hz, f0_max_hz=self.f0_max_hz,
            f0_frame_period_ms=self.f0_frame_period_ms,
        )

        confs = [w["conf"] for w in feats] or [0.0]
        coverage = len(feats) / n_expected
        time_cov = sum(w["dur_s"] for w in feats) / max(duration_s, 1e-9)
        mean_conf = float(np.mean(confs))
        min_conf = float(np.min(confs))

        flags: list[str] = []
        if dropped:
            flags.append("dropped_words")
        if coverage < self.min_coverage:
            flags.append("low_coverage")
        if mean_conf < self.min_word_conf:
            flags.append("low_conf")
        if min_conf < 0.5 * self.min_word_conf:
            flags.append("low_conf_word")
        reason = "" if not flags else (
            f"coverage={coverage:.3f} (bar {self.min_coverage}), "
            f"mean_conf={mean_conf:.3f} / min_conf={min_conf:.3f} "
            f"(bar {self.min_word_conf}), dropped={dropped[:5]}")
        res = PlanResult(
            words=feats, coverage=coverage, time_coverage=time_cov,
            mean_conf=mean_conf, min_conf=min_conf, duration_s=duration_s,
            n_words_expected=n_expected,
            ok=not (set(flags) & HARD_FLAGS), flags=flags, reason=reason,
        )
        if strict and not res.ok:
            raise PlanExtractError(f"plan rejected [{','.join(flags)}]: {reason}")
        return res

_EXTRACTOR_CACHE: dict[Any, PlanExtractor] = {}


def extract_word_plan(
    wav: Any,
    sr: Optional[int] = None,
    transcript: str = "",
    *,
    extractor: Optional[PlanExtractor] = None,
    strict: bool = False,
    **kwargs,
) -> PlanResult:
    if extractor is not None:
        if kwargs:
            raise TypeError(f"extractor= and {sorted(kwargs)} are mutually exclusive")
    else:
        try:
            key = tuple(sorted(kwargs.items()))
            extractor = _EXTRACTOR_CACHE.get(key)
            if extractor is None:
                extractor = _EXTRACTOR_CACHE[key] = PlanExtractor(**kwargs)
        except TypeError:
            extractor = PlanExtractor(**kwargs)
    return extractor.extract(wav, sr, transcript, strict=strict)


def _load_audio(wav: Any, sr: Optional[int], *, target_sr: int) -> tuple[np.ndarray, int]:
    if isinstance(wav, (str, Path)):
        import soundfile as sf
        audio, file_sr = sf.read(str(wav), dtype="float32", always_2d=False)
        sr = int(file_sr)
    else:
        audio = np.asarray(wav)
        if sr is None:
            raise PlanExtractError("sr is required when passing an audio array")
        sr = int(sr)
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        import librosa
        audio = librosa.resample(audio.astype(np.float32), orig_sr=sr,
                                 target_sr=target_sr).astype(np.float64)
        sr = target_sr
    return audio, sr


def _quantile_edges(values: np.ndarray, n_bins: int) -> Optional[np.ndarray]:
    qs = np.arange(1, n_bins) / float(n_bins)
    edges = np.quantile(np.asarray(values, dtype=np.float64), qs)
    if not np.all(np.diff(edges) > 0):
        return None
    return edges


def _bin_index(value: float, edges: np.ndarray) -> int:
    v = float(value)
    if not np.isfinite(v):
        raise PlanQuantiseError(
            f"cannot bin a non-finite value ({v!r}): NaN/inf would land in the "
            f"top bin and be indistinguishable from a measured extreme. The "
            f"feature that produced it (f0_mean_hz / energy_rms / dur_frames) "
            f"is broken for this word — fix or drop the row upstream.")
    return int(np.searchsorted(np.asarray(edges), v, side="right"))


def _bin_centres(edges: np.ndarray) -> np.ndarray:
    e = np.asarray(edges, dtype=np.float64)
    w = float(np.median(np.diff(e))) if e.size > 1 else 1.0
    c = np.empty(e.size + 1, dtype=np.float64)
    c[0] = e[0] - w / 2.0
    c[-1] = e[-1] + w / 2.0
    if e.size > 1:
        c[1:-1] = (e[:-1] + e[1:]) / 2.0
    return c


def _log2_hz(hz: float) -> float:
    return float(np.log2(max(float(hz), 1e-6)))


def _db(rms: float) -> float:
    return float(20.0 * np.log10(max(float(rms), 1e-8)))


def fit_speaker_stats(
    list_of_word_lists: Iterable[Sequence[dict]],
    *,
    n_pitch_bins: int = N_PITCH_BINS,
    n_energy_bins: int = N_ENERGY_BINS,
    min_words: int = MIN_WORDS_FOR_SPEAKER_BINS,
    speaker_id: Any = None,
) -> dict:
    pitch_vals, energy_vals = [], []
    n_words = 0
    for words in list_of_word_lists:
        for w in words:
            n_words += 1
            f0 = w.get("f0_mean_hz")
            if f0 is not None and float(f0) == float(f0) and float(f0) > 0:
                pitch_vals.append(_log2_hz(f0))
            rms = w.get("energy_rms")
            if rms is not None and float(rms) == float(rms) and float(rms) > 0:
                energy_vals.append(_db(rms))

    stats: dict[str, Any] = {
        "speaker_id": speaker_id,
        "n_words": n_words,
        "n_voiced_words": len(pitch_vals),
        "n_pitch_bins": int(n_pitch_bins),
        "n_energy_bins": int(n_energy_bins),
        "min_words": int(min_words),
        "spec_version": PLAN_SPEC_VERSION,
        "pitch_edges": None,
        "energy_edges": None,
        "f0_median_hz": None,
        "energy_median_db": None,
        "sufficient": False,
        "reason": "",
    }
    if n_words < min_words:
        stats["reason"] = (f"{n_words} words < min_words={min_words}: "
                           f"{n_pitch_bins} quantile edges would be noise")
        return stats
    if len(pitch_vals) < min_words:
        stats["reason"] = (f"only {len(pitch_vals)} voiced words < "
                           f"min_words={min_words}")
        return stats
    if len(energy_vals) < n_energy_bins:
        stats["reason"] = (f"only {len(energy_vals)} words with positive energy "
                           f"< {n_energy_bins} energy bins (silent/DC audio?)")
        return stats
    p_edges = _quantile_edges(np.array(pitch_vals), n_pitch_bins)
    e_edges = _quantile_edges(np.array(energy_vals), n_energy_bins)
    if p_edges is None or e_edges is None:
        stats["reason"] = ("degenerate spread: quantile edges are not strictly "
                           f"increasing (pitch_ok={p_edges is not None}, "
                           f"energy_ok={e_edges is not None})")
        return stats
    stats["pitch_edges"] = [float(v) for v in p_edges]
    stats["energy_edges"] = [float(v) for v in e_edges]
    stats["f0_median_hz"] = float(2.0 ** np.median(pitch_vals))
    stats["energy_median_db"] = float(np.median(energy_vals))
    stats["sufficient"] = True
    return stats


def quantise_plan(
    words: Sequence[dict],
    speaker_stats: Optional[dict] = None,
    frame_rate: float = FRAME_RATE_HZ,
    *,
    dur_edges: np.ndarray = DUR_EDGES_FRAMES,
) -> list[dict]:
    words = list(words)
    use_speaker = bool(speaker_stats and speaker_stats.get("sufficient"))
    if use_speaker:
        p_edges = np.asarray(speaker_stats["pitch_edges"], dtype=np.float64)
        e_edges = np.asarray(speaker_stats["energy_edges"], dtype=np.float64)
        if p_edges.size != N_PITCH_BINS - 1 or e_edges.size != N_ENERGY_BINS - 1:
            raise PlanQuantiseError(
                f"speaker_stats has {p_edges.size + 1} pitch / {e_edges.size + 1} "
                f"energy bins but the plan token surface is "
                f"{N_PITCH_BINS}/{N_ENERGY_BINS}")
        src = "speaker"
        p_ref = e_ref = 0.0
    else:
        p_edges = np.asarray(GLOBAL_PITCH_EDGES_OCT, dtype=np.float64)
        e_edges = np.asarray(GLOBAL_ENERGY_EDGES_DB, dtype=np.float64)
        src = "global"
        voiced = [_log2_hz(w["f0_mean_hz"]) for w in words
                  if _is_voiced_word(w)]
        loud = [_db(w["energy_rms"]) for w in words
                if float(w.get("energy_rms", 0.0)) > 0]
        p_ref = float(np.median(voiced)) if voiced else 0.0
        e_ref = float(np.median(loud)) if loud else 0.0

    out = []
    for w in words:
        q = dict(w)
        dur_s = float(w["dur_s"])
        expected = max(1, int(round(dur_s * frame_rate)))
        stored = int(w.get("dur_frames", expected))
        if abs(stored - expected) > 1:
            raise PlanQuantiseError(
                f"word {w.get('word')!r}: dur_frames={stored} was extracted at a "
                f"different frame rate than quantise_plan's {frame_rate} Hz "
                f"(expected {expected} for {dur_s:.4f}s)")
        q["dur_frames"] = expected
        q["dur_val"] = float(expected)
        q["dur_bin"] = _bin_index(np.log(max(expected, 1e-6)),
                                  np.log(np.asarray(dur_edges, dtype=np.float64)))

        if _is_voiced_word(w):
            pv = _log2_hz(w["f0_mean_hz"]) - p_ref
            q["pitch_val"] = pv
            q["pitch_bin"] = _bin_index(pv, p_edges)
        else:
            q["pitch_val"] = None
            q["pitch_bin"] = PITCH_UNVOICED_BIN

        ev = _db(w.get("energy_rms", 0.0)) - e_ref
        q["energy_val"] = ev
        q["energy_bin"] = _bin_index(ev, e_edges)

        q["bin_source"] = src
        out.append(q)
    return out


def _is_voiced_word(w: dict) -> bool:
    f0 = w.get("f0_mean_hz")
    if f0 is None:
        return False
    f0 = float(f0)
    return f0 == f0 and f0 > 0


def dequantise_plan(
    quantised: Sequence[dict],
    speaker_stats: Optional[dict] = None,
    *,
    dur_edges: np.ndarray = DUR_EDGES_FRAMES,
) -> list[dict]:
    use_speaker = bool(speaker_stats and speaker_stats.get("sufficient"))
    p_edges = (np.asarray(speaker_stats["pitch_edges"], dtype=np.float64)
               if use_speaker else np.asarray(GLOBAL_PITCH_EDGES_OCT))
    e_edges = (np.asarray(speaker_stats["energy_edges"], dtype=np.float64)
               if use_speaker else np.asarray(GLOBAL_ENERGY_EDGES_DB))
    p_c = _bin_centres(p_edges)
    e_c = _bin_centres(e_edges)
    d_c = np.exp(_bin_centres(np.log(np.asarray(dur_edges, dtype=np.float64))))

    out = []
    for w in quantised:
        d = dict(w)
        pb = int(w["pitch_bin"])
        d["pitch_hat"] = None if pb == PITCH_UNVOICED_BIN else float(p_c[pb])
        d["energy_hat"] = float(e_c[int(w["energy_bin"])])
        d["dur_hat_frames"] = float(d_c[int(w["dur_bin"])])
        out.append(d)
    return out


def plan_to_tokens(quantised: Sequence[dict], *, wrap: bool = False) -> str:
    parts: list[str] = []
    for w in quantised:
        pb, db, eb = int(w["pitch_bin"]), int(w["dur_bin"]), int(w["energy_bin"])
        if not (0 <= pb <= PITCH_UNVOICED_BIN):
            raise PlanTokenError(f"pitch_bin {pb} out of range 0..{PITCH_UNVOICED_BIN}")
        if not (0 <= db < N_DUR_BINS):
            raise PlanTokenError(f"dur_bin {db} out of range 0..{N_DUR_BINS - 1}")
        if not (0 <= eb < N_ENERGY_BINS):
            raise PlanTokenError(f"energy_bin {eb} out of range 0..{N_ENERGY_BINS - 1}")
        parts.append(PLAN_WORD_TOKEN)
        if pb != PITCH_UNVOICED_BIN:
            parts.append(plan_pitch_token(pb))
        parts += [plan_dur_token(db), plan_energy_token(eb)]
    body = "".join(parts)
    return f"{PLAN_START_TOKEN}{body}{PLAN_END_TOKEN}" if wrap else body


def tokens_to_plan(text: str) -> list[dict]:
    s = str(text).replace(PLAN_START_TOKEN, " ").replace(PLAN_END_TOKEN, " ")
    if _PLAN_TOKEN_RE.sub(" ", s).strip():
        leftover = _PLAN_TOKEN_RE.sub(" ", s).strip()
        raise PlanTokenError(f"unexpected content in plan string: {leftover!r}")

    toks = [(m.lastgroup, m.group(m.lastgroup)) for m in _PLAN_TOKEN_RE.finditer(s)]
    words: list[dict] = []
    i, n = 0, len(toks)
    while i < n:
        if toks[i][0] != "w":
            raise PlanTokenError(
                f"plan word {len(words)} starts with {toks[i][1]!r} not "
                f"{PLAN_WORD_TOKEN}")
        i += 1
        pb = PITCH_UNVOICED_BIN
        if i < n and toks[i][0] == "p":
            raw = toks[i][1]
            pb = PITCH_UNVOICED_BIN if raw == PITCH_UNVOICED_SYMBOL else int(raw)
            i += 1
        if i + 1 >= n or toks[i][0] != "d" or toks[i + 1][0] != "e":
            raise PlanTokenError(
                f"plan word {len(words)} is malformed: expected "
                f"{_D_PRE}*{_SUF}{_E_PRE}*{_SUF}, got {[t[0] for t in toks[i:i + 2]]}")
        db, eb = int(toks[i][1]), int(toks[i + 1][1])
        i += 2
        if not (0 <= pb <= PITCH_UNVOICED_BIN and 0 <= db < N_DUR_BINS
                and 0 <= eb < N_ENERGY_BINS):
            raise PlanTokenError(
                f"plan word {len(words)}: bin out of range (P={pb}, D={db}, E={eb})")
        words.append({"pitch_bin": pb, "dur_bin": db, "energy_bin": eb})
    return words


def dur_bin_centres(dur_edges: np.ndarray = DUR_EDGES_FRAMES) -> np.ndarray:
    return np.exp(_bin_centres(np.log(np.asarray(dur_edges, dtype=np.float64))))


def plan_frames(quantised_or_tokens) -> int:
    if isinstance(quantised_or_tokens, str):
        words = tokens_to_plan(quantised_or_tokens)
        d_c = dur_bin_centres()
        return int(round(sum(float(d_c[int(w["dur_bin"])]) for w in words)))
    return int(sum(int(w["dur_frames"]) for w in quantised_or_tokens))


class PlanWordList(list):
    def __init__(self, words, result: Optional[PlanResult] = None):
        super().__init__(words)
        self.result = result

_REQUIRED = object()


def parse_plan_string(text: Any) -> list[dict]:
    d_c = dur_bin_centres()
    out = []
    for w in tokens_to_plan(text):
        pb = int(w["pitch_bin"])
        db = int(w["dur_bin"])
        out.append({
            "pitch_bin": None if pb == PITCH_UNVOICED_BIN else pb,
            "dur_bin": db,
            "dur_frames": float(d_c[db]),
            "energy_bin": int(w["energy_bin"]),
        })
    return out


def extract_plan_words(
    wav: Any,
    transcript: str = "",
    *,
    speaker_stats: Any = _REQUIRED,
    sr: Optional[int] = None,
    frame_rate: float = FRAME_RATE_HZ,
    strict: bool = False,
    **kwargs,
) -> PlanWordList:
    if speaker_stats is _REQUIRED:
        raise TypeError(
            "extract_plan_words(...) requires speaker_stats: PAS compares BINS, "
            "so the realized side must be quantised with the same edges as the "
            "plan. Pass fit_speaker_stats(...) output, or speaker_stats=None "
            "explicitly to use the global (utterance-relative) fallback on both "
            "sides.")
    res = extract_word_plan(wav, sr, transcript, strict=strict, **kwargs)
    if not res.ok:
        print(f"[plan_extract] WARNING: alignment REJECTED for {wav!r} "
              f"(flags={res.flags}, coverage={res.coverage:.3f}, "
              f"reason={res.reason!r}); returning its words anyway — check "
              f"`.result.ok` before believing any score built from them.",
              file=sys.stderr, flush=True)
    words = []
    for w in quantise_plan(res.words, speaker_stats, frame_rate):
        pb = int(w["pitch_bin"])
        words.append({
            "word": w.get("word", ""),
            "pitch_bin": None if pb == PITCH_UNVOICED_BIN else pb,
            "dur_bin": int(w["dur_bin"]),
            "dur_frames": float(w["dur_frames"]),
            "energy_bin": int(w["energy_bin"]),
            "bin_source": w.get("bin_source"),
        })
    return PlanWordList(words, res)


def _iter_jsonl(path: str):
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise PlanExtractError(f"{path}:{ln} is not valid JSON: {e}")


def _row_field(row: dict, keys: Sequence[str], required: bool = True):
    for k in keys:
        if row.get(k) not in (None, ""):
            return row[k]
    if required:
        raise PlanExtractError(
            f"row is missing all of {list(keys)}; keys present: {sorted(row)[:12]}")
    return None


def cmd_extract(args) -> int:
    ex = PlanExtractor(frame_rate=args.frame_rate, device=args.device,
                       min_word_conf=args.min_conf, min_coverage=args.min_coverage)
    reasons = Counter()
    n_in = n_ok = n_flag = 0
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fo:
        for row in _iter_jsonl(args.manifest):
            if args.limit and n_in >= args.limit:
                break
            n_in += 1
            wav = _row_field(row, ["wav", "audio_path", "wav_path", "path"])
            text = _row_field(row, ["text", "prompt", "transcript"])
            try:
                res = ex.extract(wav, None, text)
            except PlanExtractError as e:
                reasons[type(e).__name__] += 1
                if args.verbose:
                    print(f"  SKIP {wav}: {e}", file=sys.stderr)
                continue
            for f in res.flags:
                reasons[f] += 1
            n_ok += 1
            n_flag += 0 if res.ok else 1
            fo.write(json.dumps({
                "id": row.get("id", n_in),
                "speaker_id": row.get("speaker_id"),
                "wav": wav,
                "text": text,
                "words": res.words,
                "coverage": res.coverage,
                "time_coverage": res.time_coverage,
                "mean_conf": res.mean_conf,
                "min_conf": res.min_conf,
                "duration_s": res.duration_s,
                "ok": res.ok,
                "flags": res.flags,
                "frame_rate": args.frame_rate,
                "spec_version": PLAN_SPEC_VERSION,
            }, ensure_ascii=False) + "\n")
            if n_ok % 200 == 0:
                print(f"  {n_ok}/{n_in}", flush=True)
    print(f"PLAN_EXTRACT_DONE rows_in={n_in} extracted={n_ok} flagged={n_flag} "
          f"-> {args.out}")
    print(f"  reasons: {dict(reasons)}")
    return 0


def cmd_quantise(args) -> int:
    rows = [r for r in _iter_jsonl(args.features)]
    if not rows:
        raise PlanExtractError(f"{args.features} has no rows to quantise")
    for r in rows:
        fr = r.get("frame_rate")
        if fr is not None and abs(float(fr) - args.frame_rate) > 1e-6:
            raise PlanQuantiseError(
                f"row {r.get('id')} was extracted at {fr} Hz but --frame_rate is "
                f"{args.frame_rate}: re-extract or re-run with the matching rate")
    n_nospk = sum(1 for r in rows if r.get("speaker_id") in (None, ""))
    if n_nospk:
        print(f"WARNING: {n_nospk}/{len(rows)} rows have no speaker_id and are "
              f"pooled into ONE pseudo-speaker; their pitch/energy bins encode "
              f"register, not relative prosody.", file=sys.stderr, flush=True)
    by_spk: dict[Any, list] = defaultdict(list)
    for r in rows:
        if r.get("ok", True) or args.keep_flagged:
            by_spk[r.get("speaker_id")].append(r["words"])
    stats = {str(spk): fit_speaker_stats(wl, min_words=args.min_words,
                                         speaker_id=spk)
             for spk, wl in by_spk.items()}
    n_suff = sum(1 for s in stats.values() if s["sufficient"])
    print(f"speakers: {len(stats)}  within-speaker bins: {n_suff}  "
          f"global fallback: {len(stats) - n_suff}")

    n_out = n_global = 0
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fo:
        for r in rows:
            if not r.get("ok", True) and not args.keep_flagged:
                continue
            st = stats.get(str(r.get("speaker_id")))
            q = quantise_plan(r["words"], st, frame_rate=args.frame_rate)
            n_global += 1 if (q and q[0]["bin_source"] == "global") else 0
            n_out += 1
            fo.write(json.dumps({
                "id": r.get("id"), "speaker_id": r.get("speaker_id"),
                "wav": r.get("wav"), "text": r.get("text"),
                "plan_tokens": plan_to_tokens(q),
                "plan_frames": plan_frames(q),
                "bin_source": q[0]["bin_source"] if q else None,
                "coverage": r.get("coverage"), "mean_conf": r.get("mean_conf"),
                "flags": r.get("flags", []),
                "words": q if args.dump_words else None,
                "spec_version": PLAN_SPEC_VERSION,
            }, ensure_ascii=False) + "\n")
    if args.stats_out:
        Path(args.stats_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.stats_out, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
    print(f"PLAN_QUANTISE_DONE rows={n_out} global_fallback_rows={n_global} "
          f"-> {args.out}")
    if not args.dump_words:
        print("  NOTE: --dump_words was not given, so `words` is null and this "
              "file CANNOT be assembled into a training corpus (the per-word "
              "dur_frames the injection schedule needs live there). Re-run with "
              "--dump_words for the corpus pass; this output is fine for "
              "statistics and the token surface.", flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("extract", help="manifest JSONL -> per-word features JSONL")
    e.add_argument("--manifest", required=True,
                   help="JSONL with wav/audio_path + text/prompt [+ speaker_id]")
    e.add_argument("--out", required=True)
    e.add_argument("--frame_rate", type=float, default=FRAME_RATE_HZ)
    e.add_argument("--device", default="cpu")
    e.add_argument("--min_conf", type=float, default=DEFAULT_MIN_WORD_CONF)
    e.add_argument("--min_coverage", type=float, default=DEFAULT_MIN_COVERAGE)
    e.add_argument("--limit", type=int, default=0)
    e.add_argument("--verbose", action="store_true")
    e.set_defaults(func=cmd_extract)

    q = sub.add_parser("quantise", help="features JSONL -> plan-token JSONL")
    q.add_argument("--features", required=True)
    q.add_argument("--out", required=True)
    q.add_argument("--stats_out", default="")
    q.add_argument("--frame_rate", type=float, default=FRAME_RATE_HZ)
    q.add_argument("--min_words", type=int, default=MIN_WORDS_FOR_SPEAKER_BINS)
    q.add_argument("--keep_flagged", action="store_true",
                   help="also quantise rows the extractor flagged (default: drop)")
    q.add_argument("--dump_words", action="store_true",
                   help="include full per-word dicts in the output (large)")
    q.set_defaults(func=cmd_quantise)

    args = ap.parse_args(argv)
    return args.func(args)

if __name__ == "__main__":
    raise SystemExit(main())
