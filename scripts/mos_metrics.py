# mos_metrics.py: Optional extra no-reference quality predictors.

from __future__ import annotations

import pathlib
import sys
import warnings

import numpy as np

_CACHE: dict = {}


def _purge_speechmos():
    for k in [k for k in sys.modules
              if k == "speechmos" or k.startswith("speechmos.")]:
        del sys.modules[k]


def _init_scorers():
    if _CACHE.get("init"):
        return
    import torch

    _purge_speechmos()
    try:
        from speechmos import dnsmos as _d, plcmos as _p
        _CACHE["dnsmos_mod"], _CACHE["plcmos_mod"] = _d, _p
    except Exception as e:
        print(f"  [mos_metrics] pip speechmos unavailable: {e}")

    _purge_speechmos()
    hub = pathlib.Path(torch.hub.get_dir()) / "tarepan_SpeechMOS_main"
    added = False
    if hub.is_dir():
        sys.path.insert(0, str(hub))
        added = True
    try:
        from eval.naturalness import NaturalnessScorer
        s = NaturalnessScorer(device="cuda" if torch.cuda.is_available()
                              else "cpu")
        s._ensure_model()
        _CACHE["utmos_scorer"] = s
    except Exception as e:
        print(f"  [mos_metrics] UTMOS unavailable: {e}")
    finally:
        if added and sys.path and sys.path[0] == str(hub):
            sys.path.pop(0)
    _CACHE["init"] = True


def _resample(x: np.ndarray, sr: int, target: int) -> np.ndarray:
    if sr == target:
        return x
    n = int(round(len(x) * target / sr))
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x)


def _mono(x: np.ndarray) -> np.ndarray:
    return x.mean(axis=1) if x.ndim > 1 else x


def dnsmos_scores(x: np.ndarray, sr: int) -> dict:
    _init_scorers()
    dnsmos = _CACHE.get("dnsmos_mod")
    if dnsmos is None:
        raise RuntimeError("pip speechmos not importable")
    y = _resample(_mono(x).astype(np.float64), sr, 16000)
    peak = np.abs(y).max()
    if peak > 0:
        y = y / peak * 0.9
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = dnsmos.run(y, 16000)
    return {"dnsmos_sig": float(r["ovrl_mos"] if "sig_mos" not in r
                                else r["sig_mos"]),
            "dnsmos_bak": float(r.get("bak_mos", float("nan"))),
            "dnsmos_ovrl": float(r.get("ovrl_mos", float("nan"))),
            "dnsmos_p808": float(r.get("p808_mos", float("nan")))}


def plcmos_score(x: np.ndarray, sr: int) -> float:
    _init_scorers()
    plcmos = _CACHE.get("plcmos_mod")
    if plcmos is None:
        raise RuntimeError("pip speechmos not importable")
    y = _resample(_mono(x).astype(np.float64), sr, 16000)
    peak = np.abs(y).max()
    if peak > 0:
        y = y / peak * 0.9
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = plcmos.run(y, 16000)
    return float(r["plcmos"] if isinstance(r, dict) else r)


def _squim():
    if "squim_obj" not in _CACHE:
        import torch
        from torchaudio.pipelines import SQUIM_OBJECTIVE, SQUIM_SUBJECTIVE
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _CACHE["squim_obj"] = SQUIM_OBJECTIVE.get_model().to(dev).eval()
        _CACHE["squim_subj"] = SQUIM_SUBJECTIVE.get_model().to(dev).eval()
        _CACHE["squim_dev"] = dev
    return _CACHE["squim_obj"], _CACHE["squim_subj"], _CACHE["squim_dev"]


def squim_scores(x: np.ndarray, sr: int, nmr: np.ndarray | None = None) -> dict:
    import torch
    obj, subj, dev = _squim()
    y = _resample(_mono(x).astype(np.float32), sr, 16000)
    peak = np.abs(y).max()
    if peak > 0:
        y = y / peak * 0.95
    t = torch.from_numpy(y).float().unsqueeze(0).to(dev)
    out = {}
    with torch.no_grad():
        try:
            stoi, pesq, sisdr = obj(t)
            out.update(squim_stoi=float(stoi[0]), squim_pesq=float(pesq[0]),
                       squim_sisdr=float(sisdr[0]))
        except Exception:
            out.update(squim_stoi=float("nan"), squim_pesq=float("nan"),
                       squim_sisdr=float("nan"))
        if nmr is not None:
            r = _resample(_mono(nmr).astype(np.float32), sr, 16000)
            rp = np.abs(r).max()
            if rp > 0:
                r = r / rp * 0.95
            rt = torch.from_numpy(r).float().unsqueeze(0).to(dev)
            try:
                out["squim_mos"] = float(subj(t, rt)[0])
            except Exception:
                out["squim_mos"] = float("nan")
    return out


def _xvector():
    if "sv" not in _CACHE:
        import torch
        from transformers import AutoFeatureExtractor, WavLMForXVector
        mid = "microsoft/wavlm-base-plus-sv"
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _CACHE["sv_fe"] = AutoFeatureExtractor.from_pretrained(mid)
        _CACHE["sv"] = WavLMForXVector.from_pretrained(mid).to(dev).eval()
        _CACHE["sv_dev"] = dev
    return _CACHE["sv_fe"], _CACHE["sv"], _CACHE["sv_dev"]


def speaker_embedding(x: np.ndarray, sr: int) -> np.ndarray:
    import torch
    fe, model, dev = _xvector()
    y = _resample(_mono(x).astype(np.float32), sr, 16000)
    inp = fe([y], sampling_rate=16000, return_tensors="pt", padding=True)
    inp = {k: v.to(dev) for k, v in inp.items()}
    with torch.no_grad():
        e = model(**inp).embeddings
    e = torch.nn.functional.normalize(e, dim=-1)[0]
    return e.cpu().numpy()


def speaker_similarity(a: np.ndarray, sr_a: int,
                       b: np.ndarray, sr_b: int) -> float:
    ea, eb = speaker_embedding(a, sr_a), speaker_embedding(b, sr_b)
    return float(np.dot(ea, eb))


def utmos_score(x: np.ndarray, sr: int) -> float:
    _init_scorers()
    s = _CACHE.get("utmos_scorer")
    if s is None:
        raise RuntimeError("UTMOS scorer unavailable")
    y = _resample(_mono(x).astype(np.float32), sr, 16000)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(s.score(y, 16000)["utmos"])


def distillmos_score(x: np.ndarray, sr: int) -> float:
    import torch
    if "distill" not in _CACHE:
        import distillmos
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        m = distillmos.ConvTransformerSQAModel().to(dev).eval()
        _CACHE["distill"], _CACHE["distill_dev"] = m, dev
    m, dev = _CACHE["distill"], _CACHE["distill_dev"]
    y = _resample(_mono(x).astype(np.float32), sr, 16000)
    with torch.no_grad():
        return float(m(torch.from_numpy(y).float().unsqueeze(0).to(dev)))


def audiobox_scores(x: np.ndarray, sr: int) -> dict:
    import tempfile
    import os
    import soundfile as sf
    if "abox" not in _CACHE:
        from audiobox_aesthetics.infer import initialize_predictor
        _CACHE["abox"] = initialize_predictor()
    y = _mono(x).astype(np.float32)
    fd, p = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        sf.write(p, y, sr)
        r = _CACHE["abox"].forward([{"path": p}])[0]
    finally:
        try:
            os.unlink(p)
        except OSError:
            pass
    return {f"ab_{k}": float(v) for k, v in r.items()}

ALL_KEYS = ["utmos", "distillmos", "dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl",
            "dnsmos_p808", "plcmos", "squim_stoi", "squim_pesq", "squim_sisdr",
            "squim_mos", "ab_CE", "ab_CU", "ab_PC", "ab_PQ"]


def score_all(x: np.ndarray, sr: int, nmr: np.ndarray | None = None) -> dict:
    out = {}
    for name, fn in (("utmos", lambda: {"utmos": utmos_score(x, sr)}),
                     ("dnsmos", lambda: dnsmos_scores(x, sr)),
                     ("plcmos", lambda: {"plcmos": plcmos_score(x, sr)}),
                     ("squim", lambda: squim_scores(x, sr, nmr)),
                     ("distillmos",
                      lambda: {"distillmos": distillmos_score(x, sr)}),
                     ("audiobox", lambda: audiobox_scores(x, sr))):
        try:
            out.update(fn())
        except Exception as e:
            print(f"    [{name}] FAILED: {type(e).__name__}: {e}")
    for k in ALL_KEYS:
        out.setdefault(k, float("nan"))
    return out
