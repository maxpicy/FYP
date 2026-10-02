# artifact_metrics.py: Completion and prefix WER (truncation kept apart from articulation).

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))


def _ops(ref, hyp):
    n, m = len(ref), len(hyp)
    if n == 0:
        return ["I"] * m
    if m == 0:
        return ["D"] * n
    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    op = np.zeros((n + 1, m + 1), dtype=np.int8)
    dp[:, 0] = np.arange(n + 1)
    dp[0, :] = np.arange(m + 1)
    op[:, 0] = 2
    op[0, :] = 3
    op[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i, j], op[i, j] = dp[i - 1, j - 1], 0
            else:
                c = (dp[i - 1, j - 1] + 1, dp[i - 1, j] + 1, dp[i, j - 1] + 1)
                best = min(c)
                dp[i, j] = best
                op[i, j] = 1 if best == c[0] else 2 if best == c[1] else 3
    out, i, j = [], n, m
    while i > 0 or j > 0:
        o = op[i, j]
        if o == 0:
            out.append("="); i -= 1; j -= 1
        elif o == 1:
            out.append("S"); i -= 1; j -= 1
        elif o == 2:
            out.append("D"); i -= 1
        else:
            out.append("I"); j -= 1
    return out[::-1]


def truncation_metrics(reference: str, hypothesis: str) -> dict:
    from eval.intelligibility import _normalize, word_error_rate
    ref, hyp = _normalize(reference), _normalize(hypothesis)
    n = len(ref)
    if n == 0:
        return {"completion": float("nan"), "prefix_wer": float("nan"),
                "trailing_del": 0, "wer": float("nan")}
    ops = _ops(ref, hyp)
    trailing = 0
    for o in reversed(ops):
        if o == "D":
            trailing += 1
        elif o == "I":
            continue
        else:
            break
    covered = max(0, n - trailing)
    prefix_ref = " ".join(ref[:covered])
    pw = (word_error_rate(prefix_ref, " ".join(hyp))["wer"] if covered
          else float("nan"))
    return {"completion": covered / n, "prefix_wer": pw,
            "trailing_del": trailing,
            "wer": word_error_rate(" ".join(ref), " ".join(hyp))["wer"]}


def autotune_metrics(wav, sr, flat_cents=5.0, jump_cents=50.0) -> dict:
    from score_naturalness_panel import f0_track
    f0, _fs = f0_track(wav, sr)
    f0 = np.asarray(f0, dtype=float)
    v = f0[np.isfinite(f0) & (f0 > 0)]
    if v.size < 8:
        return {"flat_f0": float("nan"), "jump_f0": float("nan"),
                "stair": float("nan")}
    cents = 1200.0 * np.log2(v / v[0])
    d = np.abs(np.diff(cents))
    if d.size == 0:
        return {"flat_f0": float("nan"), "jump_f0": float("nan"),
                "stair": float("nan")}
    flat = float((d < flat_cents).mean() * 100.0)
    jump = float((d > jump_cents).mean() * 100.0)
    return {"flat_f0": flat, "jump_f0": jump, "stair": flat + jump}


def prosody_metrics(wav, sr) -> dict:
    from score_naturalness_panel import f0_track
    f0, _fs = f0_track(wav, sr)
    f0 = np.asarray(f0, dtype=float)
    ok = np.isfinite(f0) & (f0 > 0)
    v = f0[ok]
    if v.size < 8:
        return {"f0_sd_st": float("nan"), "f0_rng_st": float("nan"),
                "voiced_pct": float("nan")}
    med = float(np.median(v))
    st = 12.0 * np.log2(v / med)
    return {"f0_sd_st": float(np.std(st)),
            "f0_rng_st": float(np.percentile(st, 95) - np.percentile(st, 5)),
            "voiced_pct": float(ok.mean() * 100.0)}

if __name__ == "__main__":
    full = "good morning i hope you slept well and feel rested today"
    print("exact      ", truncation_metrics(full, full))
    print("truncated  ", truncation_metrics(full, "good morning"))
    print("garbled    ", truncation_metrics(full, "good morning i hope you slipped "
                                                  "wall and fill rusted today"))
