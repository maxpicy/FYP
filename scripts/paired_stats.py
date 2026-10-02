# paired_stats.py: Paired WER contrasts between scored cells: bootstrap interval, exact sign test, Holm.

import argparse
import json
import random
from fractions import Fraction

BOOT_N, SEED = 10000, 20260704


def boot_ci(v, n_boot=BOOT_N, seed=SEED):
    rng = random.Random(seed)
    n = len(v)
    ms = sorted(sum(rng.choices(v, k=n)) / n for _ in range(n_boot))
    return ms[int(0.025 * n_boot)], ms[int(0.975 * n_boot)]


def sign_p(b, w):
    n = b + w
    if n == 0:
        return 1.0
    k, c, tot = min(b, w), 1, 0
    for i in range(0, k + 1):
        if i > 0:
            c = c * (n - i + 1) // i
        tot += c
    return float(min(Fraction(1), Fraction(2 * tot, 2 ** n)))


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    out, run = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        out[i] = run
    return out


def contrast(wa, wb):
    if len(wa) != len(wb):
        raise ValueError(f"row counts differ ({len(wa)} vs {len(wb)}): not the same prompt rows")
    d = [x - y for x, y in zip(wa, wb)]
    lo, hi = boot_ci(d)
    b, w = sum(1 for x in d if x < 0), sum(1 for x in d if x > 0)
    return {"n": len(d), "delta": sum(d) / len(d), "ci": [lo, hi], "better": b, "worse": w, "tie": len(d) - b - w,
            "sign_p": sign_p(b, w)}


def per_row_wer(spec):
    path, cell = spec.rsplit(":", 1)
    return json.load(open(path))[cell]["per_row"]["wer"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", nargs=2, action="append", required=True, metavar=("A", "B"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rows = [dict(a=a, b=b, **contrast(per_row_wer(a), per_row_wer(b))) for a, b in args.pair]
    for r, h in zip(rows, holm([r["sign_p"] for r in rows])):
        r["holm_p"] = h
        print(f"{r['a']}  -  {r['b']}: {r['delta']:+.3f} [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}]  "
              f"{r['better']}/{r['worse']}/{r['tie']}  sign p {r['sign_p']:.2g}  Holm p {h:.2g} (over {len(rows)})")
    if args.out:
        json.dump(rows, open(args.out, "w"), indent=1)

if __name__ == "__main__":
    main()
