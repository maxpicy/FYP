# gen_s1mini_synth.py: s1-mini helpers used by the Path B renderer (model loading, reference voices).

import argparse
import contextlib
import io
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from fish_codec_util import fish_repo_path

SEED_DEFAULT = 20260720
FRAME_HZ = 21.53
SEM_MAX = 4095
RES_MAX = 1023

EMOTION_TAG = {
    "whisper": "whispering",
}

_TIKTOKEN_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|"
    r" ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)


class S1MiniTiktokenTokenizer:
    def __init__(self, ckpt_dir):
        import base64
        import tiktoken

        ckpt_dir = Path(ckpt_dir)
        ranks = {}
        with open(ckpt_dir / "tokenizer.tiktoken", "rb") as f:
            for line in f:
                if line.strip():
                    tok_b64, rank = line.split()
                    ranks[base64.b64decode(tok_b64)] = int(rank)
        with open(ckpt_dir / "special_tokens.json", encoding="utf-8") as f:
            self.special_tokens = json.load(f)
        self._enc = tiktoken.Encoding(
            name="s1mini",
            pat_str=_TIKTOKEN_PATTERN,
            mergeable_ranks=ranks,
            special_tokens=self.special_tokens,
        )
        self.semantic_begin_id = self.special_tokens["<|semantic:0|>"]
        self.semantic_end_id = self.special_tokens["<|semantic:4095|>"]
        assert self.semantic_end_id - self.semantic_begin_id == 4095
        self.semantic_id_to_token_id = {
            i: self.semantic_begin_id + i for i in range(4096)}
        import torch
        self.semantic_map_tensor = torch.arange(
            self.semantic_begin_id, self.semantic_end_id + 1, dtype=torch.long)

    def encode(self, text, add_special_tokens=False, **kwargs):
        return self._enc.encode(text, allowed_special="all")

    def decode(self, tokens, **kwargs):
        if isinstance(tokens, int):
            tokens = [tokens]
        return self._enc.decode(list(tokens))

    def get_token_id(self, token):
        if token in self.special_tokens:
            return self.special_tokens[token]
        ids = self._enc.encode(token, allowed_special="all")
        assert len(ids) == 1, f"{token!r} is not a single token"
        return ids[0]


def load_texts(paths, min_words, max_words, rng):
    texts = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    t = json.loads(line).get("prompt", "")
                except json.JSONDecodeError:
                    continue
                n = len(t.split())
                if min_words <= n <= max_words:
                    texts.append(t)
    rng.shuffle(texts)
    if not texts:
        raise SystemExit("no usable texts found")
    return texts


def load_ref_specs(ref_dir, ref_manifest):
    specs = {}
    if ref_dir:
        d = Path(ref_dir)
        for wav in sorted(d.glob("*.wav")):
            txt = wav.with_suffix(".txt")
            if not txt.exists():
                print(f"[refs] skipping {wav.name}: no sibling transcript")
                continue
            text = txt.read_text(encoding="utf-8").strip()
            if text:
                specs[wav.stem] = {"name": wav.stem, "wav": str(wav), "text": text}
    if ref_manifest:
        with open(ref_manifest, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                specs[r["name"]] = {"name": r["name"], "wav": r["wav"],
                                    "text": r["text"].strip()}
    if not specs:
        raise SystemExit("no reference voices (need --ref_dir and/or --ref_manifest)")
    return [specs[k] for k in sorted(specs)]


def resolve_checkpoint(arg):
    if arg:
        return Path(arg)
    env = os.environ.get("S1MINI_CKPT")
    if env:
        return Path(env)
    cache = Path.home() / (".cache/huggingface/hub/models--fishaudio--s1-mini/"
                           "snapshots/f4b445029346701e082b60bb63fcc2d1bb17a0e2")
    if cache.exists():
        return cache
    from huggingface_hub import snapshot_download
    return Path(snapshot_download("fishaudio/s1-mini"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--output", required=True)
    ap.add_argument("--target_rows", type=int, default=1000)
    ap.add_argument("--text_jsonl", nargs="+", required=True,
                    help="JSONL file(s) with a 'prompt' field per row")
    ap.add_argument("--ref_dir", default=None,
                    help="dir of <voice>.wav + <voice>.txt reference pairs")
    ap.add_argument("--ref_manifest", default=None,
                    help="JSONL of {'name','wav','text'} reference rows")
    ap.add_argument("--emotions", default="angry,sad,happy,whisper,excited,neutral",
                    help="comma list cycled across rows; 'neutral' = no tag")
    ap.add_argument("--min_words", type=int, default=4)
    ap.add_argument("--max_words", type=int, default=28)
    ap.add_argument("--max_frames", type=int, default=645,
                    help="skip rows longer than this (~30 s at 21.53 Hz)")
    ap.add_argument("--checkpoint", default=None,
                    help="s1-mini snapshot dir (default: S1MINI_CKPT env / HF cache)")
    ap.add_argument("--device", default="cuda", choices=["cuda", "mps", "cpu"])
    ap.add_argument("--precision", default="auto",
                    choices=["auto", "bf16", "fp16", "fp32"],
                    help="auto = bf16 on cuda, fp32 elsewhere")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the decode step (CUDA only)")
    ap.add_argument("--seed", type=int, default=SEED_DEFAULT)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=30)
    ap.add_argument("--repetition_penalty", type=float, default=1.1,
                    help="accepted but UNUSED by the PyTorch path (RAS instead)")
    ap.add_argument("--max_seq_len", type=int, default=4096,
                    help="shrink the KV cache below the config's 8192 "
                         "(must exceed prompt_len + 2048)")
    ap.add_argument("--log_every", type=int, default=10)
    args = ap.parse_args()

    emotions = [e.strip() for e in args.emotions.split(",") if e.strip()]
    if not emotions:
        raise SystemExit("--emotions parsed to an empty list")

    repo = fish_repo_path()
    if not Path(repo).exists():
        raise SystemExit(f"fish-speech repo not found at {repo} "
                         f"(set FISH_SPEECH_REPO)")
    if repo not in sys.path:
        sys.path.insert(0, repo)

    import numpy as np
    import soundfile as sf
    import torch
    from loguru import logger

    logger.remove()
    logger.add(sys.stderr, level="WARNING")

    from fish_speech.models.text2semantic.inference import (
        generate_long,
        init_model,
    )
    try:
        from fish_speech.models.text2semantic.inference import (
            load_codec_model,
        )
    except ImportError:
        from fish_codec_util import load_fish_codec

        def load_codec_model(checkpoint_path, device, precision=None):
            os.environ.setdefault("FISH_CODEC_PTH", str(checkpoint_path))
            return load_fish_codec(device)
    from fish_codec_util import encode_np

    import inspect
    _GL_ACCEPTS_TOP_K = "top_k" in inspect.signature(generate_long).parameters

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    if args.precision == "auto":
        precision = torch.bfloat16 if args.device == "cuda" else torch.float32
    else:
        precision = {"bf16": torch.bfloat16, "fp16": torch.half,
                     "fp32": torch.float32}[args.precision]

    ckpt = resolve_checkpoint(args.checkpoint)
    print(f"[s1mini] checkpoint {ckpt}  device={args.device} precision={precision}")

    t_load0 = time.time()
    model, decode_one_token = init_model(
        ckpt, args.device, precision,
        compile=args.compile and args.device == "cuda")
    if (getattr(model, "tokenizer", None) is None
            or getattr(model.config, "semantic_begin_id", 0) == 0):
        tok = S1MiniTiktokenTokenizer(ckpt)
        model.tokenizer = tok
        model.config.semantic_begin_id = tok.semantic_begin_id
        model.config.semantic_end_id = tok.semantic_end_id
        print(f"[s1mini] tiktoken adapter attached (semantic ids "
              f"{tok.semantic_begin_id}..{tok.semantic_end_id})")
    if args.max_seq_len < model.config.max_seq_len:
        model.config.max_seq_len = args.max_seq_len
    codec = load_codec_model(ckpt / "codec.pth", args.device, torch.float32)
    print(f"[s1mini] model + codec loaded in {time.time() - t_load0:.1f}s "
          f"(max_seq_len {model.config.max_seq_len})")

    ref_specs = load_ref_specs(args.ref_dir, args.ref_manifest)
    voices = [r["name"] for r in ref_specs]
    spec_by_name = {r["name"]: r for r in ref_specs}
    print(f"[s1mini] {len(voices)} reference voices: {voices}")
    print(f"[s1mini] emotions: {emotions}")

    ref_cache = {}

    def get_ref(name):
        if name not in ref_cache:
            spec = spec_by_name[name]
            wav, sr = sf.read(spec["wav"], dtype="float32")
            codes = encode_np(codec, wav, sr, args.device)
            if codes is None:
                raise RuntimeError(f"reference encode failed: {spec['wav']}")
            ref_cache[name] = (spec["text"], torch.tensor(codes, dtype=torch.long))
            print(f"[refs] encoded {name}: {len(codes[0])} frames "
                  f"({len(codes[0]) / FRAME_HZ:.1f}s)")
        return ref_cache[name]

    rng = random.Random(args.seed)
    texts = load_texts(args.text_jsonl, args.min_words, args.max_words, rng)
    print(f"[s1mini] {len(texts)} candidate texts")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    n_existing = 0
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                done.add((r["prompt"], r["speaker"], r["emotion_label"]))
                n_existing += 1
    print(f"[s1mini] existing rows: {n_existing} / target {args.target_rows}")
    if n_existing >= args.target_rows:
        print("target already met; nothing to do")
        print("GEN_S1MINI_SYNTH_DONE")
        return

    written = n_existing
    n_skip_len, n_skip_range, frames_hist = 0, 0, []
    fout = open(out_path, "a", encoding="utf-8")
    t_gen0 = time.time()
    rows_at_t0 = written
    try:
        for i, text in enumerate(texts):
            if written >= args.target_rows:
                break
            voice = voices[i % len(voices)]
            emotion = emotions[i % len(emotions)]
            key = (text, f"s1mini:{voice}", emotion)
            if key in done:
                continue
            ref_text, ref_tokens = get_ref(voice)
            tag = EMOTION_TAG.get(emotion, emotion)
            tts_text = text if emotion == "neutral" else f"({tag}) {text}"

            gen_kwargs = dict(
                model=model,
                device=args.device,
                decode_one_token=decode_one_token,
                text=tts_text,
                num_samples=1,
                max_new_tokens=args.max_frames + 32,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                temperature=args.temp,
                compile=args.compile and args.device == "cuda",
                prompt_text=[ref_text],
                prompt_tokens=[ref_tokens],
            )
            if _GL_ACCEPTS_TOP_K:
                gen_kwargs["top_k"] = args.top_k
            gen = generate_long(**gen_kwargs)
            parts = []
            with contextlib.redirect_stdout(io.StringIO()):
                for resp in gen:
                    if resp.action == "sample" and resp.codes is not None:
                        parts.append(resp.codes.cpu())
            if not parts:
                n_skip_len += 1
                continue
            codes = torch.cat(parts, dim=1)
            T = codes.shape[1]
            if T < 10 or T > args.max_frames:
                n_skip_len += 1
                continue
            sem, res = codes[0], codes[1:]
            if (sem.min() < 0 or sem.max() > SEM_MAX
                    or res.min() < 0 or res.max() > RES_MAX):
                n_skip_range += 1
                print(f"[qc] range violation skipped (sem max {int(sem.max())}, "
                      f"res max {int(res.max())})")
                continue

            row = {
                "prompt": text,
                "speech_tokens": sem.tolist(),
                "residual_codes": [res[k].tolist() for k in range(res.shape[0])],
                "speaker": f"s1mini:{voice}",
                "emotion_label": emotion,
                "source": "s1mini_synth",
            }
            fout.write(json.dumps(row) + "\n")
            fout.flush()
            written += 1
            frames_hist.append(T)
            if (written - rows_at_t0) % args.log_every == 0:
                mins = (time.time() - t_gen0) / 60
                rate = (written - rows_at_t0) / mins if mins > 0 else 0
                print(f"  {written}/{args.target_rows} rows | {rate:.2f} rows/min | "
                      f"mean {sum(frames_hist)/len(frames_hist):.0f} frames | "
                      f"skips len={n_skip_len} range={n_skip_range}", flush=True)
    finally:
        fout.close()

    mins = (time.time() - t_gen0) / 60
    n_new = written - rows_at_t0
    print(f"[s1mini] wrote {n_new} new rows -> {out_path} (total {written}); "
          f"{n_new / mins:.2f} rows/min excl. load; "
          f"skips len={n_skip_len} range={n_skip_range}")
    if frames_hist:
        print(f"[s1mini] frames/row: min {min(frames_hist)} "
              f"mean {sum(frames_hist)/len(frames_hist):.0f} max {max(frames_hist)}")
    if args.device == "cuda" and torch.cuda.is_available():
        print(f"[s1mini] peak VRAM reserved: "
              f"{torch.cuda.max_memory_reserved() / 1e9:.2f} GB")
    print("GEN_S1MINI_SYNTH_DONE")

if __name__ == "__main__":
    main()
