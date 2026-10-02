# decode_fish_codes.py: Fish S2 codes (10 x T per row) -> 44.1 kHz wav files.

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes", default="data/_fish_fan_codes.jsonl")
    ap.add_argument("--outdir", default="out_fish_fan")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import soundfile as sf
    from fish_codec_util import load_fish_codec
    codec = load_fish_codec(args.device)
    sr = int(codec.sample_rate)

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in open(args.codes) if l.strip()]
    manifest = []
    for i, r in enumerate(rows):
        levels = [r["speech_tokens"]] + r["residual_codes"]
        idx = torch.tensor(levels, dtype=torch.long,
                           device=args.device).unsqueeze(0)
        with torch.no_grad():
            y = codec.from_indices(idx)
        y = y[0] if isinstance(y, (tuple, list)) else y
        w = np.clip(y.squeeze().float().cpu().numpy(), -1.0, 1.0)
        p = out / f"sample_{i:02d}.wav"
        sf.write(str(p), w, sr)
        dur = len(w) / sr
        manifest.append({"file": p.name, "prompt": r["prompt"],
                         "dur_s": round(dur, 2), "n_frames": r.get("n_frames")})
        print(f"[{i}] {p.name}  {dur:.2f}s @ {sr}Hz  "
              f"rms={float(np.sqrt((w**2).mean())):.4f}  {r['prompt'][:45]!r}",
              flush=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"DECODE_FISH_CODES_DONE  {len(manifest)} wavs -> {out}")

if __name__ == "__main__":
    main()
