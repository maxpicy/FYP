# export_release_weights.py: Training checkpoint -> weights-only model.safetensors (checked bit-exact) + config.json.

import argparse
import hashlib
import json
import os
import sys

import torch

ARCH_FLAGS = {
    "pure": [],
    "hyb": ["--hybrid_attention_top_k", "4"],
    "tfm": ["--backbone", "transformer", "--model_name", "EleutherAI/pythia-1.4b"],
}
DEPTH_FLAGS = ["--depth_arch", "mamba2", "--depth_layers", "4", "--depth_dim", "1024",
               "--depth_cond", "prefix", "--depth_feedback", "all"]
ENV = {"MVC_NUM_SPEECH_TOKENS": "2048", "MVC_NUM_CODEC_LEVELS": "10", "MVC_ENABLE_PLAN_TOKENS": "1"}
SPEAKER_SUFFIX = "speaker_encoder.embedding.weight"


def jsonable(v):
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): jsonable(x) for k, x in v.items()}
    return str(v)


def sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--arch", required=True, choices=sorted(ARCH_FLAGS))
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--description", default="")
    args = ap.parse_args()
    from safetensors.torch import load_file, save_file

    os.makedirs(args.out, exist_ok=True)
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    print(f"[export] {args.checkpoint}: top-level keys {sorted(ck)}", flush=True)
    sd = ck["model_state_dict"]
    targs = ck.get("args") or {}

    tk = int(targs.get("hybrid_attention_top_k") or 0)
    tb = str(targs.get("backbone") or "mamba")
    want = {"pure": (0, "mamba"), "hyb": (4, "mamba"), "tfm": (0, "transformer")}[args.arch]
    if (tk, "transformer" if tb == "transformer" else "mamba") != want:
        sys.exit(f"[export] FATAL: --arch {args.arch} but the checkpoint trained with "
                 f"hybrid_attention_top_k={tk}, backbone={tb}")
    for k, v in zip(DEPTH_FLAGS[0::2], DEPTH_FLAGS[1::2]):
        a = k[2:]
        if a in targs and str(targs[a]) != v:
            sys.exit(f"[export] FATAL: {k} {v} in the release flags but the checkpoint trained with {a}={targs[a]}")

    out, seen = {}, {}
    for k, v in sd.items():
        v = v.detach()
        key = (v.untyped_storage().data_ptr(), v.storage_offset(), tuple(v.shape), tuple(v.stride()))
        out[k] = v.clone().contiguous() if key in seen else v.contiguous()
        seen.setdefault(key, k)
    aliases = {k: seen[(v.untyped_storage().data_ptr(), v.storage_offset(), tuple(v.shape), tuple(v.stride()))]
               for k, v in sd.items()
               if seen[(v.untyped_storage().data_ptr(), v.storage_offset(), tuple(v.shape), tuple(v.stride()))] != k}

    wpath = os.path.join(args.out, "model.safetensors")
    save_file(out, wpath, metadata={"format": "pt", "source": os.path.basename(os.path.dirname(args.checkpoint)),
                                    "step": str(ck.get("step"))})
    del out

    back = load_file(wpath)
    if set(back) != set(sd):
        sys.exit(f"[export] FATAL: key sets differ ({len(set(sd) ^ set(back))} keys)")
    bad = [k for k in sd if back[k].dtype != sd[k].dtype or back[k].shape != sd[k].shape
           or not torch.equal(back[k], sd[k])]
    if bad:
        sys.exit(f"[export] FATAL: {len(bad)} tensors differ after the round trip, e.g. {bad[:3]}")

    spk = next((int(v.shape[0]) for k, v in sd.items() if k.endswith(SPEAKER_SUFFIX) and v.ndim == 2), None)
    dtypes = {}
    for v in sd.values():
        dtypes[str(v.dtype)] = dtypes.get(str(v.dtype), 0) + v.numel()
    flags = ARCH_FLAGS[args.arch] + DEPTH_FLAGS + (["--num_speakers", str(spk)] if spk else [])
    cfg = {
        "name": args.name,
        "description": args.description,
        "architecture": {"pure": "pure Mamba-2", "hyb": "hybrid (top-4 attention)", "tfm": "transformer"}[args.arch],
        "base_model": "EleutherAI/pythia-1.4b" if args.arch == "tfm" else "state-spaces/mamba2-1.3b",
        "source_checkpoint": args.checkpoint,
        "step": ck.get("step"),
        "stage": ck.get("stage"),
        "generator_flags": flags,
        "env": ENV,
        "speaker_table_rows": spk,
        "has_plan_injector": any(k.startswith("plan_injector.") for k in sd),
        "n_tensors": len(sd),
        "n_params": sum(v.numel() for v in sd.values()),
        "dtype_numel": dtypes,
        "tied_aliases": aliases,
        "weights_file": "model.safetensors",
        "weights_sha256": sha256(wpath),
        "weights_bytes": os.path.getsize(wpath),
        "bit_exact_vs_checkpoint": True,
    }
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    with open(os.path.join(args.out, "training_args.json"), "w") as f:
        json.dump(jsonable(targs), f, indent=2, sort_keys=True)
    print(f"[export] OK {args.name}: {cfg['n_tensors']} tensors, {cfg['n_params'] / 1e9:.3f} B params, "
          f"{cfg['weights_bytes'] / 1e9:.2f} GB, dtypes {dtypes}, speaker rows {spk}, "
          f"plan injector {cfg['has_plan_injector']}, aliases {len(aliases)}, bit-exact", flush=True)

if __name__ == "__main__":
    main()
