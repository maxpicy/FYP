# score_naturalness_panel.py: UTMOS and spectral / F0 diagnostics over a folder of wavs.

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def hf_percent(wav, sr, cutoff_hz=6000.0):
    x = np.asarray(wav, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    n = 2048
    if len(x) < n:
        return 0.0
    win = np.hanning(n)
    mags = np.stack([np.abs(np.fft.rfft(x[i:i + n] * win))
                     for i in range(0, len(x) - n, n // 2)])
    p = (mags ** 2).mean(axis=0)
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    return float(p[freqs >= cutoff_hz].sum() / (p.sum() + 1e-12) * 100.0)


def f0_track(wav, sr):
    import librosa
    x = np.asarray(wav, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    hop = int(sr * 0.010)
    f0, _, _ = librosa.pyin(x, fmin=60, fmax=400, sr=sr, hop_length=hop,
                            frame_length=2048)
    return f0, 1.0 / 0.010


def flutter_percent(f0, fs, fast_hz=8.0):
    v = np.isfinite(f0)
    if not v.any():
        return float("nan")
    runs, start = [], None
    for i, ok in enumerate(v):
        if ok and start is None:
            start = i
        elif not ok and start is not None:
            runs.append((start, i)); start = None
    if start is not None:
        runs.append((start, len(v)))
    lo, hi = max(runs, key=lambda r: r[1] - r[0])
    seg = f0[lo:hi]
    if len(seg) < int(0.4 * fs):
        return float("nan")
    seg = seg - np.polyval(np.polyfit(np.arange(len(seg)), seg, 1),
                           np.arange(len(seg)))
    spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) ** 2
    freqs = np.fft.rfftfreq(len(seg), 1.0 / fs)
    band = freqs >= 1.0
    fast = freqs > fast_hz
    return float(spec[fast].sum() / (spec[band].sum() + 1e-12) * 100.0)


def am_probe(wav, sr, band_lo=4.0, band_hi=30.0):
    x = np.asarray(wav, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    hop = int(sr * 0.010)
    nfr = len(x) // hop
    if nfr < 60:
        return float("nan"), float("nan")
    rms = np.sqrt(np.mean(
        x[: nfr * hop].reshape(nfr, hop) ** 2, axis=1))
    fs = 100.0
    thr = rms.max() * (10.0 ** (-30.0 / 20.0))
    active = rms > thr
    runs, start = [], None
    for i, ok in enumerate(active):
        if ok and start is None:
            start = i
        elif not ok and start is not None:
            runs.append((start, i)); start = None
    if start is not None:
        runs.append((start, len(active)))
    if not runs:
        return float("nan"), float("nan")
    lo, hi = max(runs, key=lambda r: r[1] - r[0])
    seg = rms[lo:hi]
    if len(seg) < int(0.5 * fs):
        return float("nan"), float("nan")
    seg = seg - np.polyval(np.polyfit(np.arange(len(seg)), seg, 1),
                           np.arange(len(seg)))
    spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) ** 2
    freqs = np.fft.rfftfreq(len(seg), 1.0 / fs)
    denom = spec[freqs >= 1.0].sum() + 1e-12
    am_pct = float(spec[(freqs >= band_lo) & (freqs <= band_hi)].sum()
                   / denom * 100.0)
    pk_mask = (freqs >= 2.0) & (freqs <= 40.0)
    peak_hz = float(freqs[pk_mask][np.argmax(spec[pk_mask])])
    return am_pct, peak_hz


def low_am_percent(wav, sr, f_lo=100.0, f_hi=500.0, m_lo=4.0, m_hi=7.5,
                   n_fft=1024, hop=256):
    x = np.asarray(wav, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if len(x) < n_fft * 4:
        return float("nan")
    win = np.hanning(n_fft)
    frames = np.lib.stride_tricks.sliding_window_view(x, n_fft)[::hop] * win
    S = np.abs(np.fft.rfft(frames, axis=1))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    fs_env = sr / hop
    full = S.sum(axis=1)
    thr = full.max() * (10.0 ** (-30.0 / 20.0))
    active = full > thr
    runs, start = [], None
    for i, ok in enumerate(active):
        if ok and start is None:
            start = i
        elif not ok and start is not None:
            runs.append((start, i)); start = None
    if start is not None:
        runs.append((start, len(active)))
    if not runs:
        return float("nan")
    lo, hi = max(runs, key=lambda r: r[1] - r[0])
    if hi - lo < int(0.5 * fs_env):
        return float("nan")
    env = S[lo:hi, (freqs >= f_lo) & (freqs < f_hi)].sum(axis=1)
    env = env - np.polyval(np.polyfit(np.arange(len(env)), env, 1),
                           np.arange(len(env)))
    spec = np.abs(np.fft.rfft(env * np.hanning(len(env)))) ** 2
    mf = np.fft.rfftfreq(len(env), 1.0 / fs_env)
    denom = spec[mf >= 1.0].sum() + 1e-12
    return float(spec[(mf >= m_lo) & (mf < m_hi)].sum() / denom * 100.0)


def flatmod_percent(wav, sr, m_lo=12.0, m_hi=25.0, f_lo=300.0, f_hi=4000.0,
                    n_fft=1024, hop=256):
    x = np.asarray(wav, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if len(x) < n_fft * 4:
        return float("nan")
    win = np.hanning(n_fft)
    frames = np.lib.stride_tricks.sliding_window_view(x, n_fft)[::hop] * win
    S = np.abs(np.fft.rfft(frames, axis=1)) ** 2 + 1e-12
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    sel = (freqs >= f_lo) & (freqs < f_hi)
    Sb = S[:, sel]
    flat = np.exp(np.mean(np.log(Sb), axis=1)) / np.mean(Sb, axis=1)
    fs_env = sr / hop
    full = S.sum(axis=1)
    thr = full.max() * (10.0 ** (-30.0 / 20.0))
    active = full > thr
    runs, start = [], None
    for i, ok in enumerate(active):
        if ok and start is None:
            start = i
        elif not ok and start is not None:
            runs.append((start, i)); start = None
    if start is not None:
        runs.append((start, len(active)))
    if not runs:
        return float("nan")
    lo, hi = max(runs, key=lambda r: r[1] - r[0])
    seg = np.log(flat[lo:hi] + 1e-9)
    if len(seg) < int(0.5 * fs_env):
        return float("nan")
    seg = seg - np.polyval(np.polyfit(np.arange(len(seg)), seg, 1),
                           np.arange(len(seg)))
    spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) ** 2
    mf = np.fft.rfftfreq(len(seg), 1.0 / fs_env)
    denom = spec[mf >= 1.0].sum() + 1e-12
    return float(spec[(mf >= m_lo) & (mf < m_hi)].sum() / denom * 100.0)


def jitter_percent(f0):
    v = np.isfinite(f0)
    d = []
    for i in range(1, len(f0)):
        if v[i] and v[i - 1]:
            d.append(abs(f0[i] - f0[i - 1]) / f0[i - 1])
    return float(np.mean(d) * 100.0) if d else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True,
                    help="directories of .wav files; each becomes one row")
    ap.add_argument("--out", default=None, help="write per-file + summary JSON")
    ap.add_argument("--no_utmos", action="store_true")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    import soundfile as sf
    scorer = None
    if not args.no_utmos:
        from eval.naturalness import NaturalnessScorer
        scorer = NaturalnessScorer(device=args.device)

    all_rows = {}
    print(f"{'dir':28s} {'n':>3s} {'utmos':>6s} {'hf%':>6s} {'flut%':>6s} "
          f"{'jit%':>6s} {'am%':>6s} {'amPk':>5s} {'lowAM':>6s} {'flatM':>6s}")
    for d in args.dirs:
        wavs = sorted(Path(d).glob("*.wav"))
        rows = []
        for w in wavs:
            audio, sr = sf.read(str(w), dtype="float32")
            f0, fs = f0_track(audio, sr)
            am_pct, am_peak = am_probe(audio, sr)
            row = {"file": w.name,
                   "hf_pct": round(hf_percent(audio, sr), 3),
                   "flutter_pct": round(flutter_percent(f0, fs), 2),
                   "jitter_pct": round(jitter_percent(f0), 3),
                   "am_pct": round(am_pct, 2),
                   "am_peak_hz": round(am_peak, 2),
                   "lowam_pct": round(low_am_percent(audio, sr), 2),
                   "flatmod_pct": round(flatmod_percent(audio, sr), 2),
                   "dur_s": round(len(audio) / sr, 2)}
            if scorer is not None:
                row["utmos"] = round(scorer.score(audio, sr)["utmos"], 3)
            rows.append(row)
        all_rows[d] = rows

        def m(k):
            vals = [r[k] for r in rows if r.get(k) == r.get(k)]
            return sum(vals) / len(vals) if vals else float("nan")
        print(f"{Path(d).name:28s} {len(rows):3d} "
              f"{m('utmos') if scorer else float('nan'):6.2f} "
              f"{m('hf_pct'):6.2f} {m('flutter_pct'):6.1f} {m('jitter_pct'):6.2f} "
              f"{m('am_pct'):6.1f} {m('am_peak_hz'):5.1f} {m('lowam_pct'):6.1f} "
              f"{m('flatmod_pct'):6.1f}")

    if args.out:
        Path(args.out).write_text(json.dumps(all_rows, indent=2))
        print(f"-> {args.out}")

if __name__ == "__main__":
    main()
