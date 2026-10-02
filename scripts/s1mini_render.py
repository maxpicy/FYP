# s1mini_render.py: Path B: forces our codebook 0 into the fine-tuned s1-mini and lets it generate codebooks 1-9.

import argparse
import contextlib
import io
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from fish_codec_util import fish_repo_path
from gen_s1mini_synth import (
    EMOTION_TAG,
    S1MiniTiktokenTokenizer,
    load_ref_specs,
    resolve_checkpoint,
)

SEM_MAX = 4095
RES_MAX = 1023
FULL_VOCAB_MIN = 100_000


class ForcedSemantic:
    def __init__(self, real_sample, sem_begin, im_end_id, fast_temps=None):
        self.real = real_sample
        self.sem_begin = sem_begin
        self.im_end_id = im_end_id
        self.queue = None
        self.calls = 0
        self.fast_temps = fast_temps
        self.res = None
        self.fast_calls = 0

    def arm(self, q0, residuals=None):
        self.queue = list(q0)
        self.calls = 0
        self.res = residuals
        self.fast_calls = 0

    @property
    def frames_forced(self):
        return (self.calls + 1) // 2

    def __call__(self, logits, temperature, top_p, top_k):
        import torch
        if logits.shape[-1] >= FULL_VOCAB_MIN:
            idx = self.calls // 2
            self.calls += 1
            if self.queue is not None and idx < len(self.queue):
                tid = self.sem_begin + int(self.queue[idx])
            else:
                tid = self.im_end_id
            return (torch.tensor([tid], device=logits.device,
                                 dtype=torch.long), None)
        frame, level = divmod(self.fast_calls, 9)
        self.fast_calls += 1
        if self.res is not None and frame < len(self.res[0]):
            return (torch.tensor([int(self.res[level][frame])],
                                 device=logits.device,
                                 dtype=torch.long), None)
        if self.fast_temps is not None:
            temperature = torch.as_tensor(self.fast_temps[level],
                                          device=logits.device,
                                          dtype=logits.dtype)
        if logits.shape[-1] > RES_MAX + 1:
            logits = logits.clone()
            logits[..., RES_MAX + 1:] = float("-inf")
        return self.real(logits, temperature, top_p, top_k)


def load_rows(path):
    rows = []
    for line in open(path, encoding="utf-8"):
        if line.strip():
            rows.append(json.loads(line))
    return rows


def resume_state(path):
    done, kept = set(), []
    if not Path(path).exists():
        return done, kept
    for line in open(path, encoding="utf-8"):
        try:
            j = json.loads(line)["renderer"]["row"]
        except (ValueError, KeyError, TypeError):
            continue
        if j not in done:
            done.add(j)
            kept.append(line.rstrip("\n"))
    return done, kept


def parse_voice_map(arg, voices):
    vm = {}
    if not arg:
        return vm
    for part in arg.split(","):
        k, v = part.split("=", 1)
        if v not in voices:
            raise SystemExit(f"--voice_map: unknown voice {v!r}; "
                             f"available: {sorted(voices)[:12]}...")
        vm[int(k)] = v
    return vm


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--codes", required=True,
                    help="cell jsonl with speech_tokens (q0) + prompt per row")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref_dir", default="data/s1mini_refs")
    ap.add_argument("--ref_manifest", default=None)
    ap.add_argument("--voice_map", default=None,
                    help="'4200=<ref>,4201=<ref>' — speaker_id to reference "
                         "voice. Unmapped speakers cycle the ref list "
                         "DETERMINISTICALLY by speaker id, so a speaker keeps "
                         "one voice across cells (the flip contrast needs "
                         "identical voices per row across cells).")
    ap.add_argument("--emotion_tag", action="store_true", default=True)
    ap.add_argument("--no_emotion_tag", dest="emotion_tag",
                    action="store_false",
                    help="drop the '(emotion)' inline marker from the text "
                         "conditioning. Same tag across a row's cells either "
                         "way, so the flip contrast is never confounded.")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--device", default="cuda", choices=["cuda", "mps", "cpu"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force_residuals", action="store_true",
                    help="PLUMBING SELF-TEST: force q1..q9 from the input "
                         "rows too. Output must reproduce the GT roundtrip; "
                         "anything else means the harness is broken.")
    ap.add_argument("--temp", type=float, default=1.0,
                    help="fast-stage sampling (repo default; the "
                         "GRPO-polished setting). Semantic is forced.")
    ap.add_argument("--fast_temps", type=float, nargs=9, default=None,
                    help="PER-LEVEL fast-stage temps for q1..q9 (9 values), "
                         "overriding the flat --temp. q0 is forced, so ALL "
                         "articulation loss lives in the residuals; cooling "
                         "the coarse ones (q1-q2) sharpens phonetic detail "
                         "while leaving the fine ones hot preserves texture.")
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=30)
    ap.add_argument("--seed", type=int, default=20260824,
                    help="row j is sampled with seed + j, so a row renders the same whatever else ran before it")
    ap.add_argument("--resume", action="store_true",
                    help="keep the rows already in --out (those carrying renderer.row) and render only the rest; "
                         "a torn last line from an interrupted write is dropped")
    ap.add_argument("--max_seq_len", type=int, default=4096)
    args = ap.parse_args()

    repo = fish_repo_path()
    if not Path(repo).exists():
        raise SystemExit(f"fish-speech repo not found at {repo}")
    if repo not in sys.path:
        sys.path.insert(0, repo)

    import soundfile as sf
    import torch
    from loguru import logger
    logger.remove()
    logger.add(sys.stderr, level="WARNING")

    import fish_speech.models.text2semantic.inference as inf_mod
    from fish_speech.models.text2semantic.inference import (
        generate_long,
        init_model,
    )
    try:
        from fish_speech.models.text2semantic.inference import load_codec_model
    except ImportError:
        from fish_codec_util import load_fish_codec

        def load_codec_model(checkpoint_path, device, precision=None):
            os.environ.setdefault("FISH_CODEC_PTH", str(checkpoint_path))
            return load_fish_codec(device)
    from fish_codec_util import encode_np

    import inspect
    gl_accepts_top_k = "top_k" in inspect.signature(generate_long).parameters

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    precision = torch.bfloat16 if args.device == "cuda" else torch.float32
    ckpt = resolve_checkpoint(args.checkpoint)
    print(f"[render] checkpoint {ckpt} device={args.device}")
    model, decode_one_token = init_model(ckpt, args.device, precision,
                                         compile=False)
    if (getattr(model, "tokenizer", None) is None
            or getattr(model.config, "semantic_begin_id", 0) == 0):
        tok = S1MiniTiktokenTokenizer(ckpt)
        model.tokenizer = tok
        model.config.semantic_begin_id = tok.semantic_begin_id
        model.config.semantic_end_id = tok.semantic_end_id
        print("[render] tiktoken adapter attached")
    if args.max_seq_len < model.config.max_seq_len:
        model.config.max_seq_len = args.max_seq_len
    codec = load_codec_model(ckpt / "codec.pth", args.device, torch.float32)

    im_end_id = model.tokenizer.get_token_id("<|im_end|>")
    forced = ForcedSemantic(inf_mod.sample, model.config.semantic_begin_id,
                            im_end_id, fast_temps=args.fast_temps)
    if args.fast_temps:
        print(f"[render] per-level fast temps q1..q9: {args.fast_temps}")
    inf_mod.sample = forced
    print(f"[render] sample() patched (semantic_begin "
          f"{model.config.semantic_begin_id}, im_end {im_end_id})")

    ref_specs = load_ref_specs(args.ref_dir, args.ref_manifest)
    voices = {r["name"]: r for r in ref_specs}
    vmap = parse_voice_map(args.voice_map, voices)
    names = sorted(voices)

    ref_cache = {}

    def get_ref(name):
        if name not in ref_cache:
            spec = voices[name]
            wav, sr = sf.read(spec["wav"], dtype="float32")
            codes = encode_np(codec, wav, sr, args.device)
            if codes is None:
                raise RuntimeError(f"reference encode failed: {spec['wav']}")
            ref_cache[name] = (spec["text"],
                               torch.tensor(codes, dtype=torch.long))
        return ref_cache[name]

    def voice_for(speaker_id):
        if speaker_id in vmap:
            return vmap[speaker_id]
        return names[int(speaker_id) % len(names)]

    rows = load_rows(args.codes)
    if args.limit:
        rows = rows[: args.limit]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done, kept = resume_state(out_path) if args.resume else (set(), [])
    if args.resume:
        print(f"[render] resume: {len(done)} rows already rendered, {len(rows) - len(done)} to go")
    n_ok = n_mismatch = n_short = 0
    t0 = time.time()
    with open(out_path, "w", encoding="utf-8") as fout:
        for line in kept:
            fout.write(line + "\n")
        fout.flush()
        for j, r in enumerate(rows):
            if j in done:
                continue
            torch.manual_seed(args.seed + j)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(args.seed + j)
            q0 = r["speech_tokens"]
            if not q0 or max(q0) > SEM_MAX or min(q0) < 0:
                raise SystemExit(f"row {j}: q0 out of the fish semantic range "
                                 f"— wrong cell file?")
            spk = r.get("speaker_id", 0)
            vname = voice_for(spk)
            ref_text, ref_tokens = get_ref(vname)
            emo = r.get("emotion") or r.get("emotion_label")
            text = r["prompt"]
            if args.emotion_tag and emo and emo != "neutral":
                text = f"({EMOTION_TAG.get(emo, emo)}) {text}"

            forced.arm(q0, residuals=(r["residual_codes"]
                                      if args.force_residuals else None))
            gen_kwargs = dict(
                model=model, device=args.device,
                decode_one_token=decode_one_token,
                text=text, num_samples=1,
                max_new_tokens=len(q0) + 8,
                top_p=args.top_p, repetition_penalty=1.1,
                temperature=args.temp, compile=False,
                prompt_text=[ref_text], prompt_tokens=[ref_tokens],
            )
            if gl_accepts_top_k:
                gen_kwargs["top_k"] = args.top_k
            parts = []
            t_row = time.time()
            with contextlib.redirect_stdout(io.StringIO()):
                for resp in generate_long(**gen_kwargs):
                    if resp.action == "sample" and resp.codes is not None:
                        parts.append(resp.codes.cpu())
            if args.device == "cuda":
                torch.cuda.synchronize()
            render_s = time.time() - t_row
            if not parts:
                print(f"[render] row {j}: EMPTY generation — skipped")
                n_short += 1
                continue
            codes = torch.cat(parts, dim=1)[:, : len(q0)]
            sem = codes[0].tolist()
            if sem != q0[: len(sem)]:
                bad = next(i for i, (a, b) in enumerate(zip(sem, q0))
                           if a != b)
                print(f"[render] row {j}: FORCING MISMATCH at frame {bad} "
                      f"(got {sem[bad]}, forced {q0[bad]}) — row dropped")
                n_mismatch += 1
                continue
            if len(sem) < len(q0):
                print(f"[render] row {j}: short render {len(sem)}/{len(q0)} "
                      f"frames — kept, truncated")
                n_short += 1
            res = codes[1:]
            if res.min() < 0 or res.max() > RES_MAX:
                raise SystemExit(f"row {j}: residual out of range — "
                                 f"codec/vocab drift, aborting")
            out = dict(r)
            out["speech_tokens"] = sem
            out["residual_codes"] = [res[k].tolist()
                                     for k in range(res.shape[0])]
            out["n_frames"] = len(sem)
            out["renderer"] = {"model": "fishaudio/s1-mini", "row": j, "seed": args.seed + j,
                               "voice": vname, "text_tagged": text != r["prompt"],
                               "temp": args.temp, "top_p": args.top_p,
                               "seconds": round(render_s, 3),
                               "rtf": round(render_s / max(len(sem) / 21.533, 1e-6), 4),
                               "device": (torch.cuda.get_device_name(0) if args.device == "cuda" else args.device)}
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            fout.flush()
            n_ok += 1
            if (j + 1) % 5 == 0:
                print(f"  {j + 1}/{len(rows)} rows "
                      f"({(time.time() - t0) / 60:.1f} min)", flush=True)

    print(f"[render] DONE: {n_ok} ok, {n_mismatch} forcing-mismatch, "
          f"{n_short} short/empty -> {out_path}")
    if n_mismatch:
        print("[render] WARNING: forcing mismatches mean the shim missed a "
              "sampling path — do NOT use this output until explained.")
    print("S1MINI_RENDER_DONE")

if __name__ == "__main__":
    main()
