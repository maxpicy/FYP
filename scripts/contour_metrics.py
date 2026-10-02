# contour_metrics.py: F0 contour diagnostics used by the suite.

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))


def _f0_semitones(a, sr):
    from score_naturalness_panel import f0_track
    f0, fs = f0_track(a, sr)
    f0 = np.asarray(f0, dtype=float)
    ok = np.isfinite(f0) & (f0 > 0)
    if ok.sum() < 12:
        return None, None, None
    idx = np.arange(len(f0))
    filled = np.interp(idx, idx[ok], f0[ok])
    st = 12.0 * np.log2(filled / np.median(f0[ok]))
    return st, float(fs), ok


def contour_metrics(a, sr) -> dict:
    nan = float("nan")
    st, fs, ok = _f0_semitones(a, sr)
    if st is None or len(st) < 16:
        return {"phrase": nan, "stress": nan, "fast": nan, "prosodic": nan,
                "f0e_corr": nan, "decl": nan}

    x = st - st.mean()
    win = np.hanning(len(x))
    P = np.abs(np.fft.rfft(x * win)) ** 2
    f = np.fft.rfftfreq(len(x), 1.0 / fs)
    band = lambda lo, hi: float(P[(f >= lo) & (f < hi)].sum())
    tot = band(0.2, 20.0) + 1e-12
    phrase, stress, fast = band(0.2, 1.5), band(1.5, 4.0), band(4.0, 20.0)

    hop = max(1, int(round(sr / fs)))
    n = min(len(st), len(a) // hop)
    rms = np.sqrt(np.mean(a[:n * hop].reshape(n, hop) ** 2, axis=1)) + 1e-9
    db = 20.0 * np.log10(rms)
    m = ok[:n] if ok is not None else np.ones(n, bool)
    corr = nan
    if m.sum() >= 12:
        u, v = st[:n][m], db[m]
        if u.std() > 1e-6 and v.std() > 1e-6:
            corr = float(np.corrcoef(u, v)[0, 1])

    t = np.arange(len(st)) / fs
    decl = float(np.polyfit(t, st, 1)[0]) if len(st) > 4 else nan

    return {"phrase": 100 * phrase / tot, "stress": 100 * stress / tot,
            "fast": 100 * fast / tot, "prosodic": 100 * (phrase + stress) / tot,
            "f0e_corr": corr, "decl": decl}


def contour_similarity(a_gen, sr_gen, a_ref, sr_ref) -> float:
    g, _, _ = _f0_semitones(a_gen, sr_gen)
    r, _, _ = _f0_semitones(a_ref, sr_ref)
    if g is None or r is None:
        return float("nan")
    n = min(len(g), len(r), 400)
    if n < 16:
        return float("nan")
    gi = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(g)), g)
    ri = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(r)), r)
    if gi.std() < 1e-6 or ri.std() < 1e-6:
        return float("nan")
    return float(np.corrcoef(gi, ri)[0, 1])

if __name__ == "__main__":
    import argparse

    import soundfile as sf

    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", help="wav dirs to compare")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    print(f"{'source':<32} {'n':>3} {'phrase%':>8} {'stress%':>8} {'fast%':>7} "
          f"{'PROSODIC%':>10} {'f0xLoud':>8} {'decl':>7}")
    print("-" * 90)
    for d in args.dirs:
        rows = []
        for w in sorted(Path(d).glob("*.wav"))[:args.limit]:
            a, sr = sf.read(str(w))
            if a.ndim > 1:
                a = a.mean(axis=1)
            rows.append(contour_metrics(a, sr))
        if not rows:
            print(f"{Path(d).name:<32} (no wavs)")
            continue
        mean = lambda k: float(np.nanmean([r[k] for r in rows]))
        print(f"{Path(d).name:<32} {len(rows):>3} {mean('phrase'):>8.1f} "
              f"{mean('stress'):>8.1f} {mean('fast'):>7.1f} {mean('prosodic'):>10.1f} "
              f"{mean('f0e_corr'):>8.2f} {mean('decl'):>7.2f}")
    print("\nprosodic% = pitch motion that is phrase/stress structured (higher = "
          "more expressive intent).\nf0xLoud = stress coupling; decl = "
          "semitones/s drift.")
