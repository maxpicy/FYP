# eval_suite.py: Scores generated cells: WER (field protocol), prefix WER, completion, UTMOS, per row and per cell.

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

GATES = {}


def suite_for_arm(codes_jsonl, wavdir, scorer, utmos, device="cuda", mos_panel=False):
    from fan_metric import flip2
    from score_naturalness_panel import (am_probe, f0_track, flutter_percent,
                                         hf_percent)

    from artifact_metrics import (autotune_metrics, prosody_metrics,
                                  truncation_metrics)
    from contour_metrics import contour_metrics

    rows = [json.loads(l) for l in open(codes_jsonl, encoding="utf-8") if l.strip()]
    coarse, wers, utm, amp, flut, hf = [], [], [], [], [], []
    comp, pwer, stair, f0sd, pros, fastp = [], [], [], [], [], []
    hyps, flut_row = [], []
    for i, r in enumerate(rows):
        for lv in (r.get("residual_codes") or [])[:3]:
            f = flip2(lv)
            if f is not None:
                coarse.append(100.0 * f)
        w = Path(wavdir) / f"sample_{i:02d}.wav"
        if not w.exists():
            continue
        audio, sr = sf.read(str(w))
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        hf.append(hf_percent(audio, sr))
        _am_pct, am_peak = am_probe(audio, sr)
        if am_peak == am_peak:
            amp.append(am_peak)
        f0, fs = f0_track(audio, sr)
        fl = flutter_percent(f0, fs)
        if fl is not None and not np.isnan(fl):
            flut.append(fl)
        flut_row.append(float(fl) if fl is not None and not np.isnan(fl) else float("nan"))
        if utmos is not None:
            import librosa
            import torch
            y = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=16000)
            with torch.no_grad():
                utm.append(float(utmos(torch.from_numpy(y).unsqueeze(0).to(device),
                                       16000).item()))
        stair.append(autotune_metrics(audio, sr)["stair"])
        f0sd.append(prosody_metrics(audio, sr)["f0_sd_st"])
        _c = contour_metrics(audio, sr)
        pros.append(_c["prosodic"]); fastp.append(_c["fast"])
        if scorer is not None and r.get("prompt"):
            hyp = scorer.transcribe(str(w))
            hyps.append(hyp)
            t = truncation_metrics(r["prompt"], hyp)
            wers.append(t["wer"])
            comp.append(t["completion"])
            pwer.append(t["prefix_wer"])

    def mean(xs):
        xs = [v for v in xs if v == v]
        return float(np.mean(xs)) if xs else float("nan")
    extra = {}
    if mos_panel:
        import mos_metrics
        acc, n_attempted = {}, 0
        errs: dict = {}

        def _try(label, fn):
            try:
                return fn()
            except Exception as e:
                errs.setdefault(label, f"{type(e).__name__}: {e}")
                return {}

        for i, r in enumerate(rows):
            w = Path(wavdir) / f"sample_{i:02d}.wav"
            if not w.exists():
                errs.setdefault("missing_wav", f"first missing: {w}")
                continue
            n_attempted += 1
            a, sr = sf.read(str(w))
            a = a.mean(axis=1) if a.ndim > 1 else a
            m = {}
            m.update(_try("dnsmos", lambda: mos_metrics.dnsmos_scores(a, sr)))
            m.update(_try("plcmos",
                          lambda: {"plcmos": mos_metrics.plcmos_score(a, sr)}))
            s = _try("squim", lambda: mos_metrics.squim_scores(a, sr))
            m.update({k: s[k] for k in ("squim_stoi", "squim_pesq",
                                        "squim_sisdr") if k in s})
            for k, v in m.items():
                acc.setdefault(k, []).append(v)
        extra = {k: mean(v) for k, v in acc.items()}
        extra["mos_panel_n"] = {k: len(v) for k, v in acc.items()}
        extra["mos_panel_attempted"] = n_attempted
        if errs or any(len(v) != n_attempted for v in acc.values()):
            print(f"  [mos_panel] WARNING for {codes_jsonl}: attempted "
                  f"{n_attempted}/{len(rows)} clip(s); per-metric n="
                  f"{extra['mos_panel_n']}. A metric with a smaller n was "
                  f"averaged over a DIFFERENT subset than the others and than "
                  f"the other arms. First error per scorer: {errs}", flush=True)
    extra["per_row"] = {"wer": [float(x) for x in wers],
                        "prefix_wer": [float(x) for x in pwer],
                        "utmos": [float(x) for x in utm],
                        "completion": [float(x) for x in comp],
                        "hyp": list(hyps), "flutter": flut_row}
    return {**extra,
            "n": len(rows), "wer": mean(wers), "utmos": mean(utm),
            "flip2_coarse": mean(coarse), "ampk": mean(amp),
            "flutter": mean(flut), "hf": mean(hf),
            "completion": mean(comp), "prefix_wer": mean(pwer),
            "stair": mean(stair), "f0_sd_st": mean(f0sd),
            "prosodic": mean(pros), "fast": mean(fastp)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", nargs=3, action="append", metavar=("NAME", "CODES", "WAVDIR"),
                    required=True, help="repeatable: NAME codes.jsonl wavdir")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--whisper", default="openai/whisper-base.en")
    ap.add_argument("--whisper_norm", default="simple", choices=["simple", "whisper"],
                    help="WER text normalisation: simple (legacy, all pre-S209 rows) or whisper "
                         "(Whisper EnglishTextNormalizer - the Seed-TTS/F5/CosyVoice convention; "
                         "use with --whisper openai/whisper-large-v3 for field-comparable WER)")
    ap.add_argument("--no_wer", action="store_true")
    ap.add_argument("--custom_axes", action="store_true",
                    help="also print the home-grown texture axes (flip2, amPk, flutter, "
                         "stair, f0sd, prosodic%%/fast%%). DIAGNOSTIC ONLY - scrapped as "
                         "verdicts on 2026-09-15; they are always dumped to --out.")
    ap.add_argument("--no_utmos", action="store_true")
    ap.add_argument("--out", default="out_eval_suite.json")
    ap.add_argument("--mos_panel", action="store_true",
                    help="add the VALIDATED extra predictors (DNSMOS P.835, "
                         "PLCMOS, SQUIM objective) as a second table. Each "
                         "passed the known-answer ladder in "
                         "scripts/validate_mos_metrics.py; squim_mos did not "
                         "and is excluded. Reported, not gated.")
    args = ap.parse_args()

    scorer = None
    if not args.no_wer:
        from eval.intelligibility import IntelligibilityScorer
        scorer = IntelligibilityScorer(model_id=args.whisper, device=args.device, normalizer=args.whisper_norm)
    if args.mos_panel:
        import mos_metrics
        mos_metrics._init_scorers()
    utmos = None
    if not args.no_utmos:
        import torch
        utmos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong",
                               trust_repo=True).to(args.device).eval()

    res = {}
    for name, codes, wavdir in args.arm:
        if not Path(codes).exists() or not Path(wavdir).exists():
            print(f"[skip] {name}: missing {codes} or {wavdir}")
            continue
        res[name] = suite_for_arm(codes, wavdir, scorer, utmos, args.device,
                                  mos_panel=args.mos_panel)

    hdr = f"\n{'arm':<24} {'n':>4} {'WER':>6} {'pWER':>6} {'comp':>6} {'UTMOS':>6}"
    if args.custom_axes:
        hdr += (f" {'flip2c':>7} {'amPk':>6} {'flutr':>6} {'stair':>6} {'f0sd':>5}"
                f" {'pros%':>6} {'fast%':>6}")
    print(hdr)
    print("-" * (len(hdr) - 1))
    for name, m in res.items():
        line = (f"{name:<24} {m['n']:>4} {m['wer']:>6.3f} {m['prefix_wer']:>6.3f} "
                f"{m['completion']:>6.2f} {m['utmos']:>6.2f}")
        if args.custom_axes:
            line += (f" {m['flip2_coarse']:>7.2f} {m['ampk']:>6.1f} {m['flutter']:>6.1f} "
                     f"{m['stair']:>6.1f} {m['f0_sd_st']:>5.2f} {m['prosodic']:>6.1f} "
                     f"{m['fast']:>6.1f}")
        print(line)
    proto = ("field protocol" if args.whisper_norm == "whisper"
             else "LEGACY normaliser - not field-comparable")
    print(f"\nWER: whisper={args.whisper} normaliser={args.whisper_norm} ({proto}); "
          "pWER = WER over the covered prefix; comp = fraction of the target spoken; "
          "UTMOS = UTMOS22, reported without a threshold.")
    print("No gates, no PASS/FAIL (verdict rule 2026-09-15): WER, industry-standard "
          "metrics, benchmarks and listening tests are the only verdicts."
          + (" The custom axes above are DIAGNOSTIC ONLY." if args.custom_axes else ""))

    if args.mos_panel:
        cols = [("dnsmos_ovrl", "dnsOVRL"), ("dnsmos_sig", "dnsSIG"),
                ("dnsmos_bak", "dnsBAK"), ("dnsmos_p808", "dnsP808"),
                ("plcmos", "PLCMOS"), ("squim_stoi", "sqSTOI"),
                ("squim_pesq", "sqPESQ"), ("squim_sisdr", "sqSISDR")]
        print(f"\n{'arm':<24}" + "".join(f"{lab:>9}" for _, lab in cols)
              + f"{'n(min/att)':>12}")
        for name, m in res.items():
            ns = m.get("mos_panel_n") or {}
            lo = min(ns.values()) if ns else 0
            print(f"{name:<24}"
                  + "".join(f"{m.get(k, float('nan')):>9.2f}" for k, _ in cols)
                  + f"{lo:>6}/{m.get('mos_panel_attempted', 0):<5}")
        print("\nEXTRA PREDICTORS — REPORTED, NOT GATED. Each passed the 20/10/5 dB")
        print("noise ladder in scripts/validate_mos_metrics.py; PLCMOS is the")
        print("strongest DISCONTINUITY detector tested (-1.47 on 5% frame dropout")
        print("vs UTMOS -0.85). squim_mos is EXCLUDED: it scored NOISIER speech")
        print("higher (3.48 -> 3.97 across the ladder) and failed outright.")
        print("NONE of them detects autotune: against a WORLD analysis-resynthesis")
        print("control, semitone-quantised F0 moves UTMOS -0.02, DNSMOS -0.04 and")
        print("SQUIM-STOI 0.00, on audio that is autotuned by construction. Do not")
        print("read a good score here as evidence that autotune is absent.")

    Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"-> {args.out}\nEVAL_SUITE_DONE")

if __name__ == "__main__":
    main()
