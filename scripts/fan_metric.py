# fan_metric.py: Code-domain repetition diagnostic (A-B-A-B limit cycles per codebook).

from __future__ import annotations

import argparse
import json
from pathlib import Path


def flip2(seq):
    if len(seq) < 3:
        return None
    hits = tot = 0
    for t in range(2, len(seq)):
        tot += 1
        if seq[t] == seq[t - 2] and seq[t] != seq[t - 1]:
            hits += 1
    return hits / tot if tot else None


def rep(seq):
    if len(seq) < 2:
        return None
    return sum(1 for t in range(1, len(seq)) if seq[t] == seq[t - 1]) / (len(seq) - 1)


def levels_of(row):
    cb0 = row.get("speech_tokens")
    res = row.get("residual_codes") or []
    if cb0 is None:
        return None
    return [cb0] + list(res)


def analyse(path, limit=None):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    if limit:
        rows = rows[:limit]
    per_level_f, per_level_r, nl = {}, {}, 0
    for r in rows:
        lv = levels_of(r)
        if not lv:
            continue
        nl = max(nl, len(lv))
        for k, seq in enumerate(lv):
            f, p = flip2(seq), rep(seq)
            if f is not None:
                per_level_f.setdefault(k, []).append(f)
            if p is not None:
                per_level_r.setdefault(k, []).append(p)
    return per_level_f, per_level_r, nl, len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    print(f"{'file':<34} {'n':>4} | " + " ".join(f"cb{k}" .rjust(6) for k in range(10))
          + " |  mean1-3  mean4-9")
    print("-" * 130)
    for p in args.paths:
        if not Path(p).exists():
            print(f"{Path(p).name:<34} (missing)")
            continue
        f, _r, nl, n = analyse(p, args.limit)
        cells = []
        for k in range(10):
            if k in f and f[k]:
                cells.append(f"{100*sum(f[k])/len(f[k]):6.2f}")
            else:
                cells.append("     -")
        def mean_of(ks):
            vs = [v for k in ks if k in f for v in f[k]]
            return f"{100*sum(vs)/len(vs):7.2f}" if vs else "      -"
        print(f"{Path(p).name:<34} {n:>4} | " + " ".join(cells)
              + f" | {mean_of(range(1,4))} {mean_of(range(4,10))}")
    print("\n(values are flip2 %, the A-B-A-B fan signature; human GT ~0.2-0.5%)")

if __name__ == "__main__":
    main()
