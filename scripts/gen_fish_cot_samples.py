# gen_fish_cot_samples.py: The generator: text -> think block (self-written, ground truth, or none) -> Fish S2
# codes, with the plan window, plan injection, the silence guard and the loop breaker.

import argparse
import json
import os as _os_rtf
import time as _time_rtf
CODEC_FPS = float(_os_rtf.environ.get("MVC_CODEC_FPS", "21.53"))
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, NamedTuple, Optional

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

FISH_CB = [4096] + [1024] * 9

PLAN_STATUS_OK = "ok"
PLAN_STATUS_ABSENT = "absent"
PLAN_STATUS_EMPTY = "empty"
PLAN_STATUS_UNTERMINATED = "unterminated"
PLAN_STATUS_MALFORMED = "malformed"

PLAN_BIN_ORDER = ("pitch", "duration", "energy")


class PlanIdMaps(NamedTuple):
    start: int
    end: int
    word: int
    pitch: dict
    duration: dict
    energy: dict


def plan_id_maps(reg) -> PlanIdMaps:
    if not getattr(reg, "plan_enabled", False):
        raise SystemExit(
            "the tokenizer in this process has no P²-CoT plan block — export "
            "MVC_ENABLE_PLAN_TOKENS=1 before running (it must also match the "
            "checkpoint: the gate shifts every speech-token id).")
    return PlanIdMaps(
        start=reg.plan_start_id, end=reg.plan_end_id, word=reg.plan_word_sep_id,
        pitch={t: b for b, t in enumerate(reg.pitch_ids)},
        duration={t: b for b, t in enumerate(reg.duration_ids)},
        energy={t: b for b, t in enumerate(reg.energy_ids)},
    )


@dataclass
class ParsedPlan:
    words: List[dict] = field(default_factory=list)
    status: str = PLAN_STATUS_ABSENT
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.status == PLAN_STATUS_OK

    @property
    def n_words(self) -> int:
        return len(self.words)


def plan_region(ids, reg) -> List[int]:
    ids = [int(t) for t in ids]
    if reg.think_start_id not in ids:
        return []
    lo = len(ids) - 1 - ids[::-1].index(reg.think_start_id)
    body = ids[lo + 1:]
    if reg.think_end_id in body:
        body = body[:body.index(reg.think_end_id)]
    return body


def parse_plan_tokens(ids, maps: PlanIdMaps) -> ParsedPlan:
    ids = [int(t) for t in ids]
    if maps.start not in ids:
        return ParsedPlan([], PLAN_STATUS_ABSENT, "no <PLAN> token in the block")
    lo = ids.index(maps.start)
    body = ids[lo + 1:]
    if maps.start in body:
        return ParsedPlan([], PLAN_STATUS_MALFORMED,
                          "more than one <PLAN> block in the think region")
    terminated = maps.end in body
    if terminated:
        body = body[:body.index(maps.end)]

    def _name(tid: int) -> str:
        for fam, table in (("pitch", maps.pitch), ("duration", maps.duration),
                           ("energy", maps.energy)):
            if tid in table:
                return f"{fam} bin {table[tid]}"
        if tid == maps.word:
            return "[PW]"
        if tid == maps.end:
            return "</PLAN>"
        return "non-plan token"

    words: List[dict] = []
    i, n = 0, len(body)
    while i < n:
        if body[i] != maps.word:
            return ParsedPlan(
                words, PLAN_STATUS_MALFORMED,
                f"word {len(words)}: expected [PW], got id {body[i]} "
                f"({_name(body[i])}) at plan token {i}")
        i += 1
        pitch = None
        if i < n and body[i] in maps.pitch:
            pitch = maps.pitch[body[i]]
            i += 1
        if i >= n or body[i] not in maps.duration:
            got = "end of block" if i >= n else f"id {body[i]} ({_name(body[i])})"
            return ParsedPlan(words, PLAN_STATUS_MALFORMED,
                              f"word {len(words)}: expected [D:*], got {got}")
        dur = maps.duration[body[i]]
        i += 1
        if i >= n or body[i] not in maps.energy:
            got = "end of block" if i >= n else f"id {body[i]} ({_name(body[i])})"
            return ParsedPlan(words, PLAN_STATUS_MALFORMED,
                              f"word {len(words)}: expected [E:*], got {got}")
        words.append({"pitch_bin": pitch, "dur_bin": dur,
                      "energy_bin": maps.energy[body[i]]})
        i += 1

    if not terminated:
        return ParsedPlan(
            words, PLAN_STATUS_UNTERMINATED,
            f"<PLAN> never closed ({len(words)} word(s) parsed before the block "
            f"ended) — ΣD is truncated, so the window it implies is too short")
    if not words:
        return ParsedPlan([], PLAN_STATUS_EMPTY, "<PLAN></PLAN> carries no words")
    return ParsedPlan(words, PLAN_STATUS_OK, "")


def plan_dur_frames(words) -> List[float]:
    import plan_extract as pe
    q = [{"pitch_bin": (pe.PITCH_UNVOICED_BIN if w["pitch_bin"] is None
                        else int(w["pitch_bin"])),
          "dur_bin": int(w["dur_bin"]), "energy_bin": int(w["energy_bin"])}
         for w in words]
    return [float(x["dur_hat_frames"]) for x in pe.dequantise_plan(q)]


def plan_sum_frames(words) -> float:
    return float(sum(plan_dur_frames(words)))


def plan_token_string(words) -> str:
    import plan_extract as pe
    q = [{"pitch_bin": (pe.PITCH_UNVOICED_BIN if w["pitch_bin"] is None
                        else int(w["pitch_bin"])),
          "dur_bin": int(w["dur_bin"]), "energy_bin": int(w["energy_bin"])}
         for w in words]
    return pe.plan_to_tokens(q, wrap=True)


class DecodeWindow(NamedTuple):
    min_fr: int
    max_fr: int
    source: str
    sum_frames: Optional[float]
    fallback: bool
    t_hat: Optional[float] = None


def resolve_decode_window(parsed: Optional[ParsedPlan], base_min: int,
                          base_max: int, base_source: str, *,
                          dur_lo: float, dur_hi: float, use_plan: bool,
                          floor: int = 20,
                          speech_ratio: float = 1.0) -> DecodeWindow:
    if not use_plan:
        return DecodeWindow(base_min, base_max, base_source, None, False, None)
    if parsed is None or not parsed.ok:
        return DecodeWindow(base_min, base_max, base_source, None, True, None)
    if not (speech_ratio > 0.0):
        raise ValueError(f"speech_ratio must be > 0, got {speech_ratio!r}")
    sigma_d = plan_sum_frames(parsed.words)
    t_hat = sigma_d / speech_ratio
    min_fr = max(floor, int(dur_lo * t_hat))
    max_fr = max(min_fr + 5, int(dur_hi * t_hat))
    return DecodeWindow(min_fr, max_fr, "plan", sigma_d, False, t_hat)


def resolve_plan_injection(on: bool, off: bool) -> bool:
    if off and on:
        print("[plan] --plan_injection_off: the injection channel is OFF for "
              "this cell (prefix-only consumption, master plan R3).", flush=True)
    return bool(on) and not bool(off)


def _row_window(row: dict, field: str, val_path: str, j: int) -> tuple:
    w = row.get(field)
    if w is None:
        raise SystemExit(
            f"{val_path} row {j} has no {field!r} field, but --plan_window_field "
            f"{field} was requested. Every row of a flip/D-PLAN cell must carry "
            f"the SAME window its control was decoded under (scripts/"
            f"plan_flip_probe.py build writes `plan_window`); falling back "
            f"row-by-row would mix two decode regimes in one cell.")
    if not isinstance(w, (list, tuple)) or len(w) != 2:
        raise SystemExit(f"{val_path} row {j}: {field!r} must be "
                         f"[min_frames, max_frames], got {w!r}")
    try:
        lo, hi = int(w[0]), int(w[1])
    except (TypeError, ValueError):
        raise SystemExit(f"{val_path} row {j}: {field!r}={w!r} is not a pair of "
                         f"integers") from None
    if lo < 1 or hi <= lo:
        raise SystemExit(f"{val_path} row {j}: {field!r}=[{lo}, {hi}] is not a "
                         f"usable window (need 1 <= min < max)")
    return lo, hi


def _norm_plan_word(w, i: int, where: str) -> dict:
    from config import DURATION_BINS, ENERGY_BINS, PITCH_BINS
    if isinstance(w, (list, tuple)):
        if len(w) != 3:
            raise SystemExit(f"{where} word {i}: a compact word must be "
                             f"[{', '.join(PLAN_BIN_ORDER)}], got {list(w)}")
        w = dict(zip(("pitch_bin", "dur_bin", "energy_bin"), w))
    if not isinstance(w, dict):
        raise SystemExit(f"{where} word {i}: expected an object, got "
                         f"{type(w).__name__}")

    def _req(key, hi):
        if w.get(key) is None:
            raise SystemExit(f"{where} word {i}: missing {key!r} "
                             f"(keys={sorted(w)[:8]})")
        v = int(w[key])
        if not 0 <= v < hi:
            raise SystemExit(f"{where} word {i}: {key}={v} out of range [0,{hi})")
        return v

    pb = w.get("pitch_bin")
    if pb is None or int(pb) == PITCH_BINS:
        pitch = None
    else:
        pitch = int(pb)
        if not 0 <= pitch < PITCH_BINS:
            raise SystemExit(f"{where} word {i}: pitch_bin={pitch} out of range "
                             f"[0,{PITCH_BINS}] ({PITCH_BINS} = unvoiced)")
    return {"pitch_bin": pitch, "dur_bin": _req("dur_bin", DURATION_BINS),
            "energy_bin": _req("energy_bin", ENERGY_BINS)}


def plan_words_from_obj(obj, *, where: str) -> List[dict]:
    if isinstance(obj, str):
        import plan_extract as pe
        try:
            raw = pe.tokens_to_plan(obj)
        except Exception as e:
            raise SystemExit(f"{where}: unparseable plan string ({e})")
    elif isinstance(obj, dict):
        if obj.get("ok") is False:
            raise SystemExit(
                f"{where}: this plan was REJECTED by the extractor "
                f"(flags={obj.get('flags')}, reason={obj.get('reason')!r}). "
                f"Fix or drop the row; do not condition on a plan the audio "
                f"does not follow.")
        raw = obj.get("words") if obj.get("words") is not None else obj.get("plan")
        if raw is None:
            raise SystemExit(f"{where}: object has neither 'words' nor 'plan' "
                             f"(keys={sorted(obj)[:8]})")
        return plan_words_from_obj(raw, where=where)
    else:
        raw = obj
    if not isinstance(raw, (list, tuple)) or not raw:
        raise SystemExit(f"{where}: empty or non-list plan "
                         f"({type(raw).__name__})")
    return [_norm_plan_word(w, i, where) for i, w in enumerate(raw)]


def load_forced_plans(path: str) -> dict:
    rows, ordered = {}, []
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError as e:
                raise SystemExit(f"{path}:{ln} is not valid JSON: {e}")
            if not isinstance(d, dict):
                raise SystemExit(f"{path}:{ln} is not a JSON object")
            if "row" in d and d["row"] is not None:
                key = int(d["row"])
                if key in rows:
                    raise SystemExit(f"{path}:{ln} repeats row {key} — the "
                                     f"forced plan for a row must be unique")
                rows[key] = d
            ordered.append(d)
    if not ordered:
        raise SystemExit(f"{path} has no plan rows")
    if rows and len(rows) != len(ordered):
        raise SystemExit(f"{path} mixes rows with and without a 'row' key "
                         f"({len(rows)} of {len(ordered)}) — pick one")
    return {"by_row": rows, "ordered": ordered}


def forced_plan_for(forced: dict, row_index: int, order_index: int, path: str):
    if forced["by_row"]:
        if row_index not in forced["by_row"]:
            raise SystemExit(
                f"--force_plan {path} has no entry for val row {row_index}. "
                f"Every generated row must carry the plan it was supposed to "
                f"be given, or the arm is a mixture of two conditions.")
        return forced["by_row"][row_index]
    if order_index >= len(forced["ordered"]):
        raise SystemExit(
            f"--force_plan {path} holds {len(forced['ordered'])} plans but the "
            f"selection reached index {order_index}. Add a 'row' key to each "
            f"line to bind plans to val row indices instead of file order.")
    return forced["ordered"][order_index]


def plan_prefix_ids(reg, words) -> List[int]:
    ids = [reg.plan_start_id]
    for w in words:
        ids.append(reg.plan_word_sep_id)
        if w["pitch_bin"] is not None:
            ids.append(reg.pitch_bin_to_id(w["pitch_bin"]))
        ids.append(reg.duration_bin_to_id(w["dur_bin"]))
        ids.append(reg.energy_bin_to_id(w["energy_bin"]))
    ids.append(reg.plan_end_id)
    return ids


def plan_bins_tensor(words, device=None):
    from model import PLAN_BIN_ABSENT, PLAN_BIN_CHANNELS
    if tuple(PLAN_BIN_CHANNELS) != PLAN_BIN_ORDER:
        raise RuntimeError(
            f"model.PLAN_BIN_CHANNELS is {PLAN_BIN_CHANNELS} but this script "
            f"builds columns as {PLAN_BIN_ORDER} — a transposed plan tensor "
            f"trains/decodes silently and looks like 'the plan does nothing'.")
    rows = [[(PLAN_BIN_ABSENT if w["pitch_bin"] is None else int(w["pitch_bin"])),
             int(w["dur_bin"]), int(w["energy_bin"])] for w in words]
    return torch.tensor([rows], dtype=torch.long, device=device)


def _speaker_inject(model, speaker_id, device):
    speaker_emb = inject = None
    if speaker_id is not None and getattr(model, "use_speaker_conditioning", False):
        sid = torch.tensor([speaker_id], dtype=torch.long, device=device)
        speaker_emb = model.speaker_encoder(sid)
        if getattr(model, "use_speaker_input", False):
            inject = model.speaker_input_proj(speaker_emb)
    return speaker_emb, inject


@torch.no_grad()
def think_budget(max_think, prompt):
    if max_think and max_think > 0:
        return int(max_think)
    return max(120, 6 * len((prompt or "").split()) + 64)


def gen_reasoning(model, prefix, reg, max_think, temp, top_k, speaker_id, device):
    import test_depth_checkpoint as tdc
    NL = tdc.NUM_LEVELS
    backbone = model.backbone.backbone
    emb = backbone.embedding
    speaker_emb, inject = _speaker_inject(model, speaker_id, device)
    ids = prefix.clone().to(device)
    codes = torch.full((1, ids.shape[1], NL), -1, dtype=torch.long, device=device)
    think_end = reg.think_end_id
    gen_toks = []
    for _ in range(max_think):
        if inject is not None:
            emb.set_speaker(inject)
        emb.set_codes(codes)
        h = backbone(ids)
        if speaker_emb is not None:
            h = model.adaln(h, speaker_emb)
        logits = model.backbone.lm_head(h[:, -1, :])[0].float()
        for _bad in (reg.audio_start_id, reg.audio_end_id, reg.eos_id,
                     reg.bos_id, reg.pad_id, reg.user_prompt_id,
                     reg.think_start_id):
            logits[_bad] = float("-inf")
        logits[reg.speech_token_id_min:reg.speech_token_id_max + 1] = float("-inf")
        if temp and temp > 0:
            probs = torch.softmax(logits / temp, dim=-1)
            if top_k and top_k > 0:
                v, ix = torch.topk(probs, min(top_k, probs.numel()))
                probs = torch.zeros_like(probs).scatter_(0, ix, v)
                probs = probs / probs.sum()
            tok = int(torch.multinomial(probs, 1).item())
        else:
            tok = int(logits.argmax().item())
        gen_toks.append(tok)
        col = torch.full((1, 1, NL), -1, dtype=torch.long, device=device)
        ids = torch.cat([ids, torch.tensor([[tok]], device=device)], dim=1)
        codes = torch.cat([codes, col], dim=1)
        if tok == think_end:
            break
    if not gen_toks or gen_toks[-1] != think_end:
        print(f"    [WARN] reasoning hit max_think={max_think} without </THINK> "
              f"-> degenerate prefix", flush=True)
        ids = torch.cat([ids, torch.tensor([[think_end]], device=device)], dim=1)
    reasoning_str = model.tokenizer.decode([t for t in gen_toks if t != think_end])
    return ids, reasoning_str


def predict_frames(dm, row):
    t = row.get("prompt") or ""
    p = (row.get("pace") or "normal").strip().lower()
    x = [1.0, float(len(t.split())), float(len(t)),
         1.0 if p == "slow" else 0.0, 1.0 if p == "fast" else 0.0]
    c = dm["coef"]
    if len(c) != len(x):
        raise ValueError(f"duration model has {len(c)} coefs, features give "
                         f"{len(x)} — regenerate it with fit_duration_model.py")
    return max(20.0, sum(a * b for a, b in zip(c, x)))


def cot_text(row, cot_mode):
    if cot_mode in ("none", "plan"):
        return ""
    tags = (f"EMO={row.get('emotion_label') or 'neutral'} "
            f"PACE={row.get('pace') or 'normal'} "
            f"PITCH={row.get('pitch') or 'normal'}")
    if cot_mode in ("tags", "tags+plan"):
        return tags
    prose = (row.get("reasoning") or "").strip()
    if cot_mode == "full":
        return f"{prose} {tags}" if prose else tags
    return prose


def f0_row(row, args):
    if not (args.prompt_tags or args.force_emotion or args.force_pace
            or args.force_pitch):
        return row, None
    r = dict(row)
    if args.force_emotion:
        r["emotion_label"] = args.force_emotion
    if args.force_pace:
        r["pace"] = args.force_pace
    if args.force_pitch:
        r["pitch"] = args.force_pitch
    tag = cot_text(r, "tags")
    if args.prompt_tags:
        r["prompt"] = f"{row['prompt']} {tag}"
    return r, tag


def build_prefix(mode, model, reg, row, max_think, think_temp, think_top_k,
                 speaker_id, device, cot_mode="prose", plan_ids=None):
    tok = model.tokenizer
    prompt_ids = tok.encode(row["prompt"], add_special_tokens=False)
    base = [reg.bos_id, reg.user_prompt_id] + prompt_ids
    if plan_ids and mode != "oracle":
        raise ValueError(f"--force_plan / an oracle plan cannot be given to "
                         f"--mode {mode}: the plan would have nowhere to go "
                         f"(nocot has no <THINK> block) or would compete with "
                         f"the model's own program (self).")
    if mode == "nocot":
        ids = torch.tensor([base + [reg.audio_start_id]], dtype=torch.long)
        return ids, None
    if mode == "oracle":
        rtext = cot_text(row, cot_mode)
        r_ids = tok.encode(rtext, add_special_tokens=False)
        ids = torch.tensor([base + [reg.think_start_id] + r_ids
                            + list(plan_ids or [])
                            + [reg.think_end_id, reg.audio_start_id]],
                           dtype=torch.long)
        return ids, rtext
    if mode == "self":
        think_prefix = torch.tensor([base + [reg.think_start_id]], dtype=torch.long)
        ids, rstr = gen_reasoning(model, think_prefix, reg, max_think,
                                  think_temp, think_top_k, speaker_id, device)
        ids = torch.cat(
            [ids, torch.tensor([[reg.audio_start_id]], device=ids.device)], dim=1)
        return ids, rstr
    raise ValueError(mode)


def _teacher_forced_cache_check(model, reg, args):
    import test_depth_checkpoint as tdc
    backbone = model.backbone.backbone
    emb = backbone.embedding
    NL = tdc.NUM_LEVELS
    dev = args.device
    rows = [json.loads(l) for l in open(args.tf_check, encoding="utf-8") if l.strip()]
    summary = []
    for r in rows:
        prefix, _ = build_prefix("nocot", model, reg, r, think_budget(args.max_think, r.get("prompt")), args.think_temp,
                                 args.think_top_k, r.get("speaker_id"), dev)
        prefix = prefix.to(dev)
        levels = [r["speech_tokens"]] + r["residual_codes"]
        T = len(levels[0])
        frames = [[int(levels[k][t]) for k in range(NL)] for t in range(T)]
        spk = r.get("speaker_id"); speaker_emb = None; inject = None
        if spk is not None and getattr(model, "use_speaker_conditioning", False):
            sid = torch.tensor([int(spk)], dtype=torch.long, device=dev)
            speaker_emb = model.speaker_encoder(sid)
            if getattr(model, "use_speaker_input", False):
                inject = model.speaker_input_proj(speaker_emb)
        placeholder = reg.audio_start_id
        P = prefix.shape[1]

        def ids_codes(t):
            ids = torch.cat([prefix, torch.full((1, t), placeholder, dtype=torch.long, device=dev)], dim=1)
            codes = torch.full((1, P + t, NL), -1, dtype=torch.long, device=dev)
            for i in range(t):
                codes[0, P + i] = torch.tensor(frames[i], device=dev)
            return ids, codes

        def head(h):
            if speaker_emb is not None:
                h = model.adaln(h, speaker_emb)
            last = h[:, -1, :]
            return last.float().clone(), model.backbone.lm_head(last).float()[0].clone()

        last_re, logit_re, last_ca, logit_ca = [], [], [], []
        with torch.no_grad():
            for t in range(T):
                ids, codes = ids_codes(t)
                if inject is not None:
                    emb.set_speaker(inject)
                emb.set_codes(codes)
                a, b = head(backbone(ids)); last_re.append(a); logit_re.append(b)
            ip = tdc._fresh_inference_cache(P + T + 8); n_fed = 0
            for t in range(T):
                ids, codes = ids_codes(t)
                lo = n_fed
                if inject is not None:
                    emb.set_speaker(inject)
                emb.set_codes(codes[:, lo:])
                ip.seqlen_offset = n_fed
                h = backbone(ids[:, lo:], inference_params=ip)
                n_fed = ids.shape[1]
                a, b = head(h); last_ca.append(a); logit_ca.append(b)
            fr_re = [model.depth_module.generate_frame(a) for a in last_re]
            fr_ca = [model.depth_module.generate_frame(a) for a in last_ca]
        rel = [float((a - b).norm() / (a.norm() + 1e-9)) for a, b in zip(last_re, last_ca)]
        lm_mism = []
        for t, (la, lb) in enumerate(zip(logit_re, logit_ca)):
            if int(la.argmax()) != int(lb.argmax()):
                top2 = torch.topk(la, 2).values
                lm_mism.append({"t": t, "top2_gap": float(top2[0] - top2[1]),
                                "max_abs_logit_diff": float((la - lb).abs().max())})
        fr_mism = [t for t in range(T) if list(fr_re[t]) != list(fr_ca[t])]
        rec = {"row": r.get("row"), "frames": T, "max_rel_hidden": max(rel),
               "mean_rel_hidden": sum(rel) / len(rel), "lmhead_argmax_mismatch": len(lm_mism),
               "lmhead_mismatches": lm_mism[:10], "depth_frame_mismatch": len(fr_mism),
               "depth_frame_mismatch_first": fr_mism[:10]}
        summary.append(rec)
        print(f"[tf_check] row {rec['row']}: frames={T} max_rel_hidden={rec['max_rel_hidden']:.2e} "
              f"mean_rel={rec['mean_rel_hidden']:.2e} lmhead_argmax_mismatch={len(lm_mism)}/{T} "
              f"depth_frame_mismatch={len(fr_mism)}/{T}"
              + (f" first_depth_mismatch_t={fr_mism[0]}" if fr_mism else ""), flush=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    print(f"[tf_check] wrote {args.out}", flush=True)

ARM_E_TEMPS = (0.2, [0.2, 0.2, 0.3, 0.9, 1.3, 1.6, 1.9, 2.2, 2.5])
H_TEMPS = (0.7, [0.80, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7])


def resolve_decode_temps(args):
    explicit = args.temp_semantic is not None and args.acoustic_temps is not None
    h_family = (getattr(args, "rep_penalty_levels", None) is not None or (getattr(args, "ras_tau", 0.0) or 0.0) > 0
                or getattr(args, "ras_tau_levels", None) is not None)
    if h_family and not explicit and not getattr(args, "allow_default_temps", False):
        raise SystemExit(
            "decode temperatures not given: --rep_penalty_levels / --ras_tau belong to decode H, whose temperatures are "
            "--temp_semantic 0.7 --acoustic_temps 0.80 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7. Pass them explicitly (or "
            "--allow_default_temps to reproduce a pre-2026-09-28 cold cell).")
    if args.temp_semantic is None:
        args.temp_semantic = ARM_E_TEMPS[0]
    if args.acoustic_temps is None:
        args.acoustic_temps = list(ARM_E_TEMPS[1])
    args.temps_explicit = explicit
    return args


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["gt", "nocot", "oracle", "self"])
    ap.add_argument("--cot_mode", default="prose",
                    choices=["prose", "tags", "none", "plan", "tags+plan", "full"],
                    help="MUST match the arm. Only affects --mode oracle, which "
                         "otherwise injects the prose into every arm. The last "
                         "three are the P²-CoT tiers (C1/C2/C3): they put a "
                         "<PLAN> program in the block, which needs a plan for "
                         "the row (sidecar `--plan_field`, or --force_plan).")
    ap.add_argument("--checkpoint", help="required for nocot/oracle/self")
    ap.add_argument("--val", default="data/s2_qwen_cot_fish_val.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", type=int, nargs="*", default=None,
                    help="explicit 0-based val row indices; default = --limit head")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--max_words", type=int, default=0,
                    help="skip prompts longer than this. Decode cost is ~T^2 "
                         "(full-sequence forward per frame) and min_frames is "
                         "6*words, so long prompts are quadratically expensive; "
                         "0 = no filter.")
    ap.add_argument("--model_name", default="state-spaces/mamba2-1.3b")
    ap.add_argument("--rep_penalty", type=float, default=1.0,
                    help="windowed repetition penalty per codebook (1.0 = off). "
                         "Every reference AR-TTS uses one: fish-speech 1.1-1.2 "
                         "over 16 frames, CosyVoice-2 RAS over 10, IndexTTS 10.0. "
                         "The fan IS token repetition; this is the mechanism "
                         "that suppresses it WITHOUT cooling the temperature.")
    ap.add_argument("--rep_penalty_levels", type=float, nargs=10, default=None,
                    help="PER-LEVEL penalties (10 values, cb0..cb9), overriding "
                         "--rep_penalty. The residual fan concentrates in cb1 "
                         "(flip2 1.58 vs GT 0.85) while cb4-9 already sit near "
                         "GT, so a global rise would tax levels that are fine.")
    ap.add_argument("--ras_tau", type=float, default=0.0,
                    help="Repetition Aware Sampling threshold (VALL-E 2 Alg.1; "
                         "0 = off). After nucleus sampling, if the drawn token "
                         "already occupies >= tau of this level's last "
                         "--ras_win frames, REDRAW from the full untruncated "
                         "distribution. CosyVoice-2 ships tau=0.1 over win 10.")
    ap.add_argument("--ras_tau_levels", type=float, nargs=10, default=None,
                    help="per-level RAS thresholds (cb0..cb9), overriding "
                         "--ras_tau — our fan is localized to cb1.")
    ap.add_argument("--ras_win", type=int, default=10)
    ap.add_argument("--rep_win", type=int, default=16,
                    help="repetition-penalty window in frames (fish uses 16).")
    ap.add_argument("--loop_break_flipk", type=float, default=0.0,
                    help="cb0 LOOP-BREAKER trigger: trailing-window flipK%% "
                         "(period-2..8 cycle rate) at/above which the kick "
                         "engages. 0 = off (default; every frozen battery DEC "
                         "is unchanged). 5.0 is the ear-validated threshold "
                         "(2026-08-23 map: warble clips 5-29, clean/GT <2). "
                         "The loop is the tremolo+early-cutoff mechanism: "
                         "rep_penalty 1.2 and RAS cannot break it at "
                         "temp_semantic 0.2 (peaked distribution).")
    ap.add_argument("--loop_break_win", type=int, default=32,
                    help="trailing cb0 window (frames) for the loop metric.")
    ap.add_argument("--loop_break_temp", type=float, default=0.6,
                    help="cb0 temp while inside a detected loop (kick).")
    ap.add_argument("--loop_break_rep", type=float, default=2.0,
                    help="cb0 rep_penalty while inside a detected loop.")
    ap.add_argument("--loop_break_stop_after", type=int, default=0,
                    help="force a clean stop after this many CONSECUTIVE "
                         "kicked frames past min_frames (0 = never): a loop "
                         "that survives the kick ends at a frame boundary "
                         "instead of vamping to the window cap.")
    ap.add_argument("--silence_guard_codes", default="",
                    help="SILENCE GUARD (ledger S200): path to a JSON list of "
                         "cb0 PAUSE codes (data/silence_codes_fish_s2.json). "
                         "Empty = off (default; every frozen DEC unchanged). The "
                         "'cuts off / fails to speak' clips are the semantic "
                         "stream collapsing into these codes for 30-240 frames "
                         "where real speech never exceeds 9.")
    ap.add_argument("--silence_guard_budget", type=int, default=10,
                    help="mid-utterance: after this many CONSECUTIVE pause-set "
                         "frames, mask the set on the next frame so the head "
                         "must re-engage with the injected word. 10 > GT max 9.")
    ap.add_argument("--silence_guard_stop_after", type=int, default=10,
                    help="at the plan's LAST word: a pause run this long is the "
                         "end of the utterance -> clean stop (0 = never).")
    ap.add_argument("--silence_guard_stop_noplan", action="store_true",
                    help="also apply the stop rule when there is NO plan "
                         "schedule (base / noplan cells), VAD-style.")
    ap.add_argument("--silence_guard_retries", type=int, default=0,
                    help="v2 REWIND (S201): on a spent mid-utterance budget, "
                         "discard the whole pause run and re-sample it from the "
                         "last speech frame with the pause set masked for "
                         "`budget` frames and cb0 at --silence_guard_retry_temp; "
                         "up to this many times per row. 0 = v1 (single-frame "
                         "mask), bit-identical to before.")
    ap.add_argument("--silence_guard_retry_temp", type=float, default=0.6,
                    help="cb0 temperature while re-sampling a rewound segment.")
    ap.add_argument("--onset_frames", type=int, default=0,
                    help="damp the ACOUSTIC temperatures for this many frames at "
                         "audio start. flip2 peaks in the first 20%% of every "
                         "generated utterance (5.4-8.2%% vs GT 1.26%%) because "
                         "there is no acoustic history yet. 0 = off.")
    ap.add_argument("--onset_temp_scale", type=float, default=1.0,
                    help="multiplier applied to acoustic temps during the onset "
                         "window (0.0 = argmax, which also tests whether the "
                         "onset fan is a SAMPLING artifact or the model itself).")
    ap.add_argument("--hybrid_attention_top_k", type=int, default=0,
                    help="rebuild the top-K attention graft before loading. MUST "
                         "match the checkpoint: a hybrid ckpt loaded into a pure "
                         "model has shape-mismatched attention projections, and "
                         "those layers would fall back to PRETRAINED Mamba weights "
                         "while the rest are trained (now a hard error).")
    ap.add_argument("--backbone", choices=["mamba", "transformer"], default="mamba",
                    help="must match the checkpoint. 'transformer' also requires "
                         "--model_name EleutherAI/pythia-1.4b. Mismatching these "
                         "now raises in load_checkpoint rather than degrading to "
                         "base text weights, but set them correctly regardless.")
    ap.add_argument("--num_speakers", type=int, default=8192)
    ap.add_argument("--speaker_dim", type=int, default=256)
    ap.add_argument("--depth_dim", type=int, default=1024)
    ap.add_argument("--depth_layers", type=int, default=2)
    ap.add_argument("--depth_arch", choices=["gru", "mamba2", "transformer"], default="gru",
                    help="MUST match the checkpoint's depth head.")
    ap.add_argument("--depth_heads", type=int, default=8)
    ap.add_argument("--depth_feedback", choices=["all", "semantic"],
                    default="semantic",
                    help="MUST match training. fish-recipe arms use 'all' (MCF).")
    ap.add_argument("--decode_feedback", choices=["all", "semantic", "argmax"],
                    default=None,
                    help="what the TEMPORAL STREAM is fed at decode; defaults "
                         "to --depth_feedback (i.e. match training). 'argmax' "
                         "is decode-time STATE DECOUPLING: emit the sampled "
                         "frame but feed the MODE back, so the recurrence sees "
                         "near-manifold inputs instead of codes drawn at temps "
                         "up to 2.5 while training was teacher-forced on "
                         "ground truth (the S11 mechanism, inference side). "
                         "Separate from --depth_feedback because that one also "
                         "configures the model's teacher-forced masking and "
                         "must keep matching the checkpoint.")
    ap.add_argument("--depth_cond", choices=["add", "prefix"], default="add",
                    help="MUST match training. fish-recipe arms use 'prefix'.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    ap.add_argument("--candidates", type=int, default=1,
                    help="best-of-N: emit N sampled candidates per row (rank "
                         "them by WER downstream). Decode is stochastic, and "
                         "each candidate reseeds so draws are independent.")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--force_speaker_id", type=int, default=None,
                    help="override the row's speaker_id. Needed to test the "
                         "STAGE-1 base fairly: ids 4200-4204 are Stage-2-only "
                         "(Stage-1 trained 0-3955), so scoring the base on them "
                         "would confound 'untrained embedding' with 'collapse'.")
    ap.add_argument("--max_frames", type=int, default=400)
    ap.add_argument("--tf_check", default="",
                    help="DIAGNOSTIC: a generated JSONL whose rows are teacher-forced through both the "
                         "recompute and the cached backbone paths; per-frame hidden-state and logit "
                         "agreement is written to --out and the run exits (pbs/cached_identity.pbs phase 2).")
    ap.add_argument("--cached", action="store_true",
                    help="Cached-state decode (2026-09-11): carry Mamba2 conv/ssm states, the hybrid "
                         "KV cache and the transformer past_key_values across frames and feed one "
                         "position per step, instead of re-running the whole prefix every frame. "
                         "Numerics-equivalent to the default path, NOT bit-identical (S212: hidden <=3e-3 rel, "
                         "LM-head argmax 360/360; token-identical on mamba2-130m in tests); a battery adopts it "
                         "for ALL compared arms or none (prereg Amendment 7, pbs/p2cot_eval.pbs CACHED=1). "
                         "Off by default: every banked result used the recompute path.")
    ap.add_argument("--force_frames", type=int, default=0,
                    help="EFFICIENCY SWEEP ONLY (external_benchmarks.md §3.4): pin the decode to exactly N "
                         "frames (min_frames = max_frames = N), overriding every derived window, so "
                         "rtf_lm / peak_mem_gb are measured at a chosen generated length. Use with "
                         "--mode nocot; a plan schedule would be desynchronised from a forced length.")
    ap.add_argument("--min_frames_per_word", type=float, default=6.0)
    ap.add_argument("--duration_model", default=None,
                    help="JSON from scripts/fit_duration_model.py. When set, "
                         "min/max frames come from the predicted duration "
                         "instead of the flat 6-frames-per-word floor and the "
                         "fixed --max_frames cap.")
    ap.add_argument("--dur_lo", type=float, default=0.6,
                    help="min_frames = dur_lo * predicted. Below 1.0 so the "
                         "stop head can still end a row the predictor "
                         "overestimated.")
    ap.add_argument("--dur_hi", type=float, default=1.5,
                    help="hard cap = dur_hi * predicted. This is what removes "
                         "the runaway; 1.5 leaves room for the ~30%% of rows "
                         "the predictor misses by more than 20%%.")
    ap.add_argument("--temp_semantic", type=float, default=None)
    ap.add_argument("--acoustic_temps", type=float, nargs="*", default=None)
    ap.add_argument("--allow_default_temps", action="store_true",
                    help="decode on the arm-E ladder even though H-family controls (--rep_penalty_levels / --ras_tau) "
                         "are set, i.e. reproduce a cell decoded before 2026-09-28. Never for new cells.")
    ap.add_argument("--top_k", type=int, default=64)
    ap.add_argument("--top_p", type=float, default=0.0,
                    help="nucleus sampling (0 = off). Reference AR-TTS systems "
                         "all use it: fish-speech 0.7-0.8, CosyVoice-2 0.8, "
                         "IndexTTS 0.8. generate_frame already implements it; "
                         "only the CLI flag was missing.")
    ap.add_argument("--stop_threshold", type=float, default=0.95)
    ap.add_argument("--dump_stop_probs", action="store_true",
                    help="record p(stop) for EVERY frame into the output row as "
                         "`stop_probs` (+ `stop_argmax`). Combined with a "
                         "threshold that never fires (--stop_threshold 1.1) "
                         "this makes one run yield the exact clip length for "
                         "every (threshold, dur_lo) pair, because the stop "
                         "check is a pure break that consumes no randomness. "
                         "Consumed by scripts/stop_threshold_sweep.py.")
    ap.add_argument("--antifan_bias", type=float, default=0.0)
    ap.add_argument("--antifan_levels", type=int, nargs="*", default=None)
    ap.add_argument("--scale_cap", type=float, default=0.0,
                    help="UNBOUNDED decode with the frame cap scaled to the "
                         "predicted duration (e.g. 3.0 = 3x). A flat 400 cap "
                         "means ~5x on a 14-word line and ~1.5x on a 35-word "
                         "one, so 'at-cap' is not comparable across prompt "
                         "lengths (S141). Needs --duration_model_path.")
    ap.add_argument("--duration_model_path", default=None,
                    help="predictor for --scale_cap ONLY; does not bound")
    ap.add_argument("--prompt_tags", action="store_true",
                    help="append the EMO/PACE/PITCH tag TEXT to the prompt "
                         "region. For probing a base model that has never seen "
                         "a <THINK> block; the record keeps the original prompt "
                         "so WER still catches the model speaking the tag.")
    ap.add_argument("--force_emotion", default=None,
                    help="override emotion_label on every row (tag-flip probe)")
    ap.add_argument("--force_pace", default=None,
                    help="override pace on every row (tag-flip probe)")
    ap.add_argument("--force_pitch", default=None,
                    help="override pitch on every row (tag-flip probe)")
    ap.add_argument("--think_temp", type=float, default=0.7)
    ap.add_argument("--think_top_k", type=int, default=50)
    ap.add_argument("--max_think", type=int, default=120,
                    help="<THINK> token budget per row; 0 = AUTO (6 x words + 64, floor 120). The fixed "
                         "default caps a self-authored plan at ~28 words - use 0 on any set with longer texts.")
    ap.add_argument("--plan_window", action="store_true",
                    help="derive the decode window from the PLAN the model "
                         "emitted (or was given): min/max frames = "
                         "--plan_dur_lo/--plan_dur_hi times ΣD̂ of its "
                         "dequantised word durations, replacing the "
                         "text-regression --duration_model for that row. S155: "
                         "the stop threshold is inert and the duration FLOOR is "
                         "the lever, so this is the mechanism that ties clip "
                         "length to the model's stated intent. A row whose plan "
                         "is missing or malformed keeps the old window and says "
                         "so (plan_status / plan_window_fallback).")
    ap.add_argument("--plan_injection", action="store_true",
                    help="feed the plan into the AUDIO block through the "
                         "duration-scheduled input-injection channel (§3.2), "
                         "scheduled from the model's OWN emitted durations. "
                         "Requires a checkpoint trained with --plan_injection: "
                         "a zero-init injector is refused rather than run as a "
                         "silent no-op.")
    ap.add_argument("--plan_injection_off", action="store_true",
                    help="cancel --plan_injection for THIS cell, keeping every "
                         "other plan flag identical. The prefix-only condition "
                         "(master plan R3): if injection is the only path to "
                         "the program, 'the plan was consumed' is trivially "
                         "true, so the honest headline number is measured with "
                         "the channel off. Exists as a negation rather than by "
                         "dropping the flag so a batch script can share one "
                         "argument block across the ON and OFF cells "
                         "(pbs/p2cot_eval.pbs) — the two cells then differ by "
                         "exactly this one token.")
    ap.add_argument("--plan_dur_lo", type=float, default=0.8,
                    help="min_frames = plan_dur_lo * T-hat (D-PLAN profile). "
                         "0.8 per Amendment 4: [0.9,1.5] contained the true "
                         "length on only 82.6%% of eval rows even rho-corrected "
                         "(bar 95%%); [0.8,1.5] measures 98.2%%.")
    ap.add_argument("--plan_dur_hi", type=float, default=1.5,
                    help="hard cap = plan_dur_hi * T-hat, where T-hat = ΣD̂ / ρ "
                         "(D-PLAN profile). NB the cap scales T-hat, NOT ΣD̂ "
                         "directly — see plan_decode_window(), which computes "
                         "max_fr from t_hat. The help text used to say ΣD̂, which "
                         "understates the cap by 1/ρ (~1.5x) and cost a truncation "
                         "diagnosis a wrong turn on 2026-09-01.")
    ap.add_argument("--force_plan", default=None,
                    help="JSONL of plans to inject INSTEAD of the model's own — "
                         "the oracle-plan condition and the plan-flip probe. "
                         "One object per line, `row` binding it to a val row "
                         "index (else file order over the selection), carrying "
                         "`plan` (a token string, a word list, or an extractor "
                         "sidecar). Requires --mode oracle; get the model's own "
                         "plans from a --mode self run's plan_words first.")
    ap.add_argument("--plan_field", default="plan",
                    help="row field holding the plan sidecar for a --mode "
                         "oracle plan arm (matches the dataset's plan_field).")
    ap.add_argument("--plan_window_field", default=None,
                    help="row field holding an EXPLICIT [min_frames, max_frames] "
                         "window, which then bounds decode instead of "
                         "--duration_model or --plan_window. This is how the "
                         "G-P4 flip cells keep one window across conditions: "
                         "duration bins bound the decode window (§3.4), so "
                         "letting the window follow a FLIPPED plan would "
                         "manufacture the length effect the probe measures. "
                         "scripts/plan_flip_probe.py build writes the control "
                         "plan's window into every cell row as `plan_window`. A "
                         "row missing the field is a hard error, not a silent "
                         "fall back to a different window — that would put two "
                         "decode regimes inside one cell.")
    ap.add_argument("--plan_schedule", choices=["stretch", "pack"],
                    default="stretch",
                    help="how plan words map onto frames during injection. "
                         "'stretch' is training's default and needs a target "
                         "length (ΣD̂ / --plan_speech_ratio); 'pack' needs only "
                         "the plan but drifts early at every pause.")
    ap.add_argument("--plan_speech_ratio", type=float, default=1.0,
                    help="ρ for the 'stretch' target: T̂ = ΣD̂ / ρ. ΣD counts "
                         "SPEECH frames only (word spans exclude pauses — 0.64 "
                         "of total on the one clip measured so far), so the "
                         "default 1.0 schedules the program over its own ΣD̂ "
                         "frames and holds the last word after that. Set it "
                         "from the corpus's measured time_coverage to spread "
                         "the program the way training's 'stretch' did; the "
                         "default invents no constant.")
    ap.add_argument("--plan_d_plan", type=int, default=256,
                    help="MUST match the checkpoint's plan-injector width.")
    ap.add_argument("--plan_on_malformed", choices=["fallback", "skip", "error"],
                    default="fallback",
                    help="what to do with a row whose plan is absent/truncated/"
                         "malformed: keep it on the OLD window (recorded, "
                         "warned, counted), drop the candidate, or abort.")
    args = ap.parse_args()
    resolve_decode_temps(args)

    args.plan_injection = resolve_plan_injection(args.plan_injection,
                                                 args.plan_injection_off)

    plan_mode = bool(args.plan_window or args.plan_injection or args.force_plan
                     or args.plan_window_field
                     or args.cot_mode in ("plan", "tags+plan", "full"))

    os.environ.setdefault("MVC_NUM_SPEECH_TOKENS", "2048")
    os.environ.setdefault("MVC_NUM_CODEC_LEVELS", "10")
    if plan_mode:
        os.environ.setdefault("MVC_ENABLE_PLAN_TOKENS", "1")
        if args.mode in ("gt", "nocot"):
            raise SystemExit(
                f"--mode {args.mode} has no <THINK> block, so there is no plan "
                f"to read, force or schedule from. Plan decode applies to "
                f"--mode self (the model's own program) and --mode oracle (a "
                f"given one).")
        if args.force_plan and args.mode != "oracle":
            raise SystemExit(
                "--force_plan requires --mode oracle. The flip/oracle probe is "
                "two passes: --mode self to get the model's own plans "
                "(plan_words in the output), then --mode oracle --force_plan "
                "with the modified file.")
        if args.mode == "oracle" and args.cot_mode not in ("plan", "tags+plan",
                                                           "full"):
            raise SystemExit(
                f"--mode oracle with plan decode needs --cot_mode plan/"
                f"tags+plan/full (got {args.cot_mode!r}): the window and the "
                f"injection must come from a plan the model was actually "
                f"conditioned on, not one it never saw.")
        if args.plan_dur_lo > args.plan_dur_hi:
            raise SystemExit(f"--plan_dur_lo {args.plan_dur_lo} > --plan_dur_hi "
                             f"{args.plan_dur_hi}: the window would be empty")
        if args.plan_window and args.plan_window_field:
            raise SystemExit(
                "--plan_window and --plan_window_field are mutually exclusive: "
                "the first derives the window from the plan IN CONTEXT (which "
                "moves when the plan is flipped), the second pins it to the "
                "value carried on the row. Pick one per cell.")
        if args.plan_speech_ratio <= 0.0:
            raise SystemExit("--plan_speech_ratio must be > 0")
        if (args.plan_injection and args.plan_schedule == "stretch"
                and args.plan_speech_ratio == 1.0):
            print("[plan] WARNING: --plan_schedule stretch with "
                  "--plan_speech_ratio 1.0 schedules the program over its own "
                  "SumD_hat frames and holds the last word afterwards. Training "
                  "stretched it over the FULL utterance (SumD/T ~ 0.64 on the "
                  "one clip measured so far), so this is a train/inference "
                  "schedule mismatch unless your corpus really has no pauses. "
                  "Use the `rho = SumD/T median` line from the training log.",
                  flush=True)
        if args.mode == "self":
            print(f"[plan] --max_think {args.max_think} caps the program at "
                  f"~{max(0, (args.max_think - 8)) // 4} words (3-4 tokens "
                  f"each, before any prose/tags). Rows that outgrow it are "
                  f"reported as `unterminated`, not silently shortened.",
                  flush=True)
        if args.plan_window and not args.dump_stop_probs:
            print("[plan] NOTE: --dump_stop_probs is off, so this run cannot "
                  "feed the S154 stop-calibration readout (the pre-registered "
                  "prediction is that plan windows remove the certain-early "
                  "population). The window itself is recorded either way.",
                  flush=True)

    rows_all = [json.loads(l) for l in open(args.val, encoding="utf-8") if l.strip()]
    if args.rows is not None:
        idxs = args.rows
    else:
        cand = range(len(rows_all))
        if args.max_words:
            cand = [i for i in cand
                    if len(rows_all[i]["prompt"].split()) <= args.max_words]
        idxs = list(cand)[:args.limit]
        print(f"[gen] {len(idxs)} prompts selected"
              + (f" (<= {args.max_words} words)" if args.max_words else ""),
              flush=True)
    rows = [rows_all[i] for i in idxs]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "gt":
        n = 0
        with out_path.open("w", encoding="utf-8") as f:
            for j, r in zip(idxs, rows):
                f.write(json.dumps({
                    "prompt": r["prompt"], "row": j, "mode": "gt",
                    "emotion": r.get("emotion_label"), "speaker_id": r.get("speaker_id"),
                    "speech_tokens": r["speech_tokens"],
                    "residual_codes": r["residual_codes"],
                    "n_frames": len(r["speech_tokens"]),
                }) + "\n")
                n += 1
        print(f"GEN_FISH_COT_SAMPLES_DONE mode=gt wrote {n} -> {out_path}")
        return

    assert args.checkpoint, "nocot/oracle/self require --checkpoint"
    import config as _cfg
    from model import MambaCoTModel
    from train import load_checkpoint
    import test_depth_checkpoint as tdc
    assert tdc.NUM_LEVELS == 10, (
        f"MVC_NUM_CODEC_LEVELS not seen as 10 (tdc.NUM_LEVELS={tdc.NUM_LEVELS}, "
        f"env={os.environ.get('MVC_NUM_CODEC_LEVELS')!r})")
    assert _cfg.NUM_SPEECH_TOKENS == 2048, (
        f"MVC_NUM_SPEECH_TOKENS not seen as 2048 "
        f"(config.NUM_SPEECH_TOKENS={_cfg.NUM_SPEECH_TOKENS}, "
        f"env={os.environ.get('MVC_NUM_SPEECH_TOKENS')!r})")

    from train import checkpoint_speaker_rows
    _ckpt_spk = checkpoint_speaker_rows(args.checkpoint)
    if _ckpt_spk is not None and _ckpt_spk != args.num_speakers:
        print(f"[speaker] checkpoint carries a {_ckpt_spk}-row speaker table; "
              f"--num_speakers {args.num_speakers} overridden to match it "
              f"(a mismatch would re-initialise every voice).", flush=True)
        args.num_speakers = _ckpt_spk

    _dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    model = MambaCoTModel(model_name=args.model_name, device=args.device,
                          dtype=_dtype, mtp_num_heads=0,
                          backbone_kind=args.backbone)
    if args.hybrid_attention_top_k:
        model.enable_hybrid_attention(args.hybrid_attention_top_k)
    model.enable_speaker_conditioning(speaker_dim=args.speaker_dim,
                                      num_speakers=args.num_speakers,
                                      input_injection=True)
    model.enable_depth_module(num_levels=10, codebook_size=2048,
                              d_depth=args.depth_dim, depth_layers=args.depth_layers,
                              depth_feedback=args.depth_feedback, cross_frame=0,
                              codebook_sizes=FISH_CB,
                              depth_arch=args.depth_arch,
                              depth_cond=args.depth_cond,
                              depth_heads=args.depth_heads)
    _ckpt_has_injector = False
    try:
        from train import checkpoint_tensor_shapes
        _ckpt_has_injector = any(k.startswith("plan_injector.")
                                 for k in checkpoint_tensor_shapes(args.checkpoint))
    except Exception as e:
        print(f"[plan] could not peek at {args.checkpoint} ({type(e).__name__}); "
              f"falling back to the flag", flush=True)
    if args.plan_injection or _ckpt_has_injector:
        model.enable_plan_injection(d_plan=args.plan_d_plan,
                                    scaffold_dropout=0.0)
        if _ckpt_has_injector and not args.plan_injection:
            print("[plan] checkpoint carries a plan injector; building it so the "
                  "weights load EXACTLY, but this cell decodes with the plan "
                  "channel unused (Amendment 4's matched floor).", flush=True)
    load_checkpoint(args.checkpoint, model, device=args.device)
    model.eval()
    reg = model.token_registry
    if args.tf_check:
        _teacher_forced_cache_check(model, reg, args)
        return
    lts = [args.temp_semantic] + list(args.acoustic_temps)

    maps = None
    if plan_mode:
        maps = plan_id_maps(reg)
    if args.plan_injection:
        w = model.plan_injector.proj.weight
        if float(w.detach().abs().sum().item()) == 0.0:
            raise SystemExit(
                f"--plan_injection: {args.checkpoint} carried no trained "
                f"plan_injector (proj.weight is still exactly zero-init), so "
                f"injection would be a numeric no-op. Either the checkpoint "
                f"predates the plan channel, or --plan_d_plan "
                f"{args.plan_d_plan} does not match its width and the weights "
                f"were dropped on load.")
        print(f"[plan] injection ON (d_plan={args.plan_d_plan}, bins="
              f"{model.plan_bin_sizes}, schedule={args.plan_schedule})",
              flush=True)

    if args.force_pace and args.duration_model:
        print(f"[f0] bounded + --force_pace {args.force_pace}: window is "
              f"computed from the ORIGINAL row; readout_f0_probe.py verifies "
              f"it is identical across cells from the recorded dur_window",
              flush=True)

    dur_model_for_cap = None
    if args.scale_cap and args.duration_model:
        raise SystemExit(
            "--scale_cap and --duration_model are mutually exclusive: one "
            "imposes a WINDOW on the length, the other only a ceiling, and "
            "the bounded path silently wins. Pick one.")
    if args.scale_cap:
        if not args.duration_model_path:
            raise SystemExit(
                "--scale_cap needs --duration_model_path (the predictor is "
                "used for the ceiling only; pass --duration_model instead if "
                "you want a bounded window)")
        dur_model_for_cap = json.loads(
            Path(args.duration_model_path).read_text(encoding="utf-8"))
        print(f"CAP SCALED to {args.scale_cap:g}x the predicted duration "
              f"(floor {args.max_frames}); decode is otherwise UNBOUNDED",
              flush=True)

    dur_model = None
    if args.duration_model:
        dur_model = json.loads(Path(args.duration_model).read_text(
            encoding="utf-8"))
        print(f"DURATION-BOUNDED decode: {args.duration_model} "
              f"feats={dur_model.get('features')} "
              f"window [{args.dur_lo:g}x, {args.dur_hi:g}x] of prediction",
              flush=True)

    forced = load_forced_plans(args.force_plan) if args.force_plan else None
    if args.plan_window:
        print(f"PLAN-BOUNDED decode (§3.4): window [{args.plan_dur_lo:g}x, "
              f"{args.plan_dur_hi:g}x] of the plan's own ΣD̂"
              + (f", overriding {args.duration_model}" if dur_model else "")
              + f"; malformed -> {args.plan_on_malformed}", flush=True)
    plan_counts = {}

    n = 0
    with out_path.open("w", encoding="utf-8") as f, torch.no_grad():
        for order_index, (j, r_orig) in enumerate(zip(idxs, rows)):
            r, f0_tag = f0_row(r_orig, args)
            spk = (args.force_speaker_id if args.force_speaker_id is not None
                   else r.get("speaker_id", 1))
            min_fr = max(20, int(len(r_orig["prompt"].split())
                                 * args.min_frames_per_word))
            max_fr = args.max_frames
            if dur_model is not None:
                t_hat = predict_frames(dur_model, r_orig)
                min_fr = max(20, int(args.dur_lo * t_hat))
                max_fr = max(min_fr + 5, int(args.dur_hi * t_hat))
            elif args.scale_cap:
                t_hat = predict_frames(dur_model_for_cap, r_orig)
                max_fr = max(args.max_frames, int(args.scale_cap * t_hat))
            window_field_value = None
            if args.plan_window_field:
                window_field_value = _row_window(r_orig, args.plan_window_field,
                                                 args.val, j)
                min_fr, max_fr = window_field_value
            base_min_fr, base_max_fr = min_fr, max_fr
            base_source = (
                f"row_field:{args.plan_window_field}" if window_field_value
                else "duration_model" if dur_model is not None
                else "scale_cap" if args.scale_cap else "words")

            given_words, plan_ids, plan_source = None, None, "self"
            if plan_mode and args.mode == "oracle":
                if forced is not None:
                    obj = forced_plan_for(forced, j, order_index, args.force_plan)
                    given_words = plan_words_from_obj(
                        obj, where=f"--force_plan entry for row {j}")
                    plan_source = "forced"
                else:
                    if r_orig.get(args.plan_field) is None:
                        raise SystemExit(
                            f"{args.val} row {j} has no {args.plan_field!r} "
                            f"field, so there is no oracle plan to condition "
                            f"on. Point --plan_field at the sidecar, or pass "
                            f"--force_plan.")
                    given_words = plan_words_from_obj(
                        r_orig[args.plan_field],
                        where=f"{args.val} row {j} field {args.plan_field!r}")
                    plan_source = "row"
                plan_ids = plan_prefix_ids(reg, given_words)
            for c in range(args.candidates):
                torch.manual_seed(args.seed + 1000 * j + c)
                prefix, rstr = build_prefix(args.mode, model, reg, r, think_budget(args.max_think, r.get("prompt")),
                                            args.think_temp, args.think_top_k, spk,
                                            args.device, cot_mode=args.cot_mode,
                                            plan_ids=plan_ids)
                min_fr, max_fr = base_min_fr, base_max_fr
                plan_extra, plan_bins, plan_sched = {}, None, None
                if plan_mode:
                    parsed = parse_plan_tokens(
                        plan_region(prefix[0].tolist(), reg), maps)
                    if given_words is not None:
                        got = [(w["pitch_bin"], w["dur_bin"], w["energy_bin"])
                               for w in parsed.words]
                        want = [(w["pitch_bin"], w["dur_bin"], w["energy_bin"])
                                for w in given_words]
                        if not parsed.ok or got != want:
                            raise SystemExit(
                                f"row {j}: the plan written into the prefix does "
                                f"not read back ({parsed.status}: "
                                f"{parsed.reason or 'bins differ'}). The "
                                f"registry's bin<->id mapping and the parser "
                                f"disagree — every conditioned arm would be "
                                f"conditioned on something other than the plan "
                                f"it was handed.")
                    win = resolve_decode_window(
                        parsed, base_min_fr, base_max_fr, base_source,
                        dur_lo=args.plan_dur_lo, dur_hi=args.plan_dur_hi,
                        use_plan=args.plan_window,
                        speech_ratio=args.plan_speech_ratio)
                    min_fr, max_fr = win.min_fr, win.max_fr
                    plan_counts[parsed.status] = plan_counts.get(
                        parsed.status, 0) + 1
                    if not parsed.ok:
                        print(f"[row {j} cand {c}] PLAN {parsed.status}: "
                              f"{parsed.reason}", flush=True)
                        if args.plan_on_malformed == "error":
                            raise SystemExit(
                                f"row {j} cand {c}: plan {parsed.status} and "
                                f"--plan_on_malformed error")
                        if args.plan_on_malformed == "skip":
                            continue
                    target_frames = None
                    if args.plan_injection and parsed.ok:
                        durs = plan_dur_frames(parsed.words)
                        if args.plan_schedule == "stretch":
                            target_frames = max(1, int(round(
                                sum(durs) / args.plan_speech_ratio)))
                        plan_sched = tdc.build_decode_plan_schedule(
                            durs, prefix.shape[1], max_fr,
                            mode=args.plan_schedule, target_frames=target_frames,
                            device=args.device)
                        plan_bins = plan_bins_tensor(parsed.words,
                                                     device=args.device)
                    plan_extra = {
                        "plan_source": plan_source,
                        "plan_status": parsed.status,
                        "plan_ok": parsed.ok,
                        "plan_reason": parsed.reason,
                        "plan_n_words": parsed.n_words,
                        "plan_used": (plan_token_string(parsed.words)
                                      if parsed.ok else None),
                        "plan_words": [[w["pitch_bin"], w["dur_bin"],
                                        w["energy_bin"]] for w in parsed.words],
                        "plan_t_hat": (round(win.t_hat, 2)
                                       if win.t_hat is not None else None),
                        "plan_sum_frames": (round(plan_sum_frames(parsed.words), 2)
                                            if parsed.ok else None),
                        "plan_window_fallback": win.fallback,
                        "plan_injected": plan_bins is not None,
                        "plan_schedule_mode": (args.plan_schedule
                                               if plan_bins is not None else None),
                        "plan_speech_ratio": (args.plan_speech_ratio
                                              if plan_bins is not None else None),
                        "plan_target_frames": target_frames,
                        "dur_window_source": win.source,
                    }
                sp_out = [] if args.dump_stop_probs else None
                lb = (tdc.LoopBreaker(threshold=args.loop_break_flipk,
                                      win=args.loop_break_win,
                                      temp=args.loop_break_temp,
                                      rep=args.loop_break_rep,
                                      stop_after=args.loop_break_stop_after)
                      if args.loop_break_flipk > 0 else None)
                if args.force_frames:
                    min_fr = max_fr = int(args.force_frames)
                sg = None
                if args.silence_guard_codes:
                    with open(args.silence_guard_codes, encoding="utf-8") as _sf:
                        _sc = json.load(_sf)
                    sg = tdc.SilenceGuard(
                        codes=(_sc["codes"] if isinstance(_sc, dict) else _sc),
                        budget=args.silence_guard_budget,
                        stop_after=args.silence_guard_stop_after,
                        stop_without_plan=args.silence_guard_stop_noplan,
                        retries=args.silence_guard_retries,
                        retry_temp=args.silence_guard_retry_temp)
                _cuda = str(args.device).startswith("cuda")
                if _cuda:
                    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                _t0 = _time_rtf.perf_counter()
                codes = tdc.depth_generate(
                    model, prefix, reg, max_frames=max_fr, level_temps=lts,
                    stop_prob_out=sp_out,
                    top_k=args.top_k, top_p=args.top_p, allow_stop=True, min_frames=min_fr,
                    stop_threshold=args.stop_threshold, speaker_id=spk,
                    feedback=(args.decode_feedback or args.depth_feedback),
                    cross_frame_decode=False,
                    device=args.device, antifan_bias=args.antifan_bias,
                    antifan_levels=args.antifan_levels,
                    onset_frames=args.onset_frames,
                    onset_temp_scale=args.onset_temp_scale,
                    rep_penalty=(args.rep_penalty_levels or args.rep_penalty),
                    rep_win=args.rep_win,
                    ras_tau=(args.ras_tau_levels or args.ras_tau),
                    ras_win=args.ras_win,
                    plan_bins=plan_bins, plan_schedule=plan_sched,
                    loop_breaker=lb, silence_guard=sg, cached=args.cached)
                if _cuda:
                    torch.cuda.synchronize()
                _gen_s = _time_rtf.perf_counter() - _t0
                _peak_gb = (torch.cuda.max_memory_allocated() / 2 ** 30) if _cuda else None
                if codes is None or len(codes[0]) < 3:
                    print(f"[row {j} cand {c}] SKIP empty gen: {r['prompt'][:40]!r}",
                          flush=True)
                    continue
                stop_extra = {}
                if sp_out is not None:
                    stop_extra = {
                        "stop_probs": [round(p, 6) for p, _ in sp_out],
                        "stop_argmax": [int(a) for _, a in sp_out],
                    }
                f.write(json.dumps({
                    "prompt": r_orig["prompt"], "row": j, "cand": c,
                    "mode": args.mode,
                    "prompt_used": r["prompt"], "tag": f0_tag,
                    "dur_window": [min_fr, max_fr],
                    "emotion": r.get("emotion_label"), "speaker_id": spk,
                    "reasoning_used": rstr,
                    "speech_tokens": codes[0], "residual_codes": codes[1:],
                    "n_frames": len(codes[0]),
                    "audio_seconds": round(len(codes[0]) / CODEC_FPS, 3),
                    "gen_seconds": round(_gen_s, 3),
                    "rtf_lm": round(_gen_s / (len(codes[0]) / CODEC_FPS), 4),
                    "cached": bool(args.cached),
                    "seed": int(args.seed),
                    "level_temps": [round(float(t), 3) for t in lts],
                    "temps_explicit": bool(getattr(args, "temps_explicit", False)),
                    **({"peak_mem_gb": round(_peak_gb, 3)} if _peak_gb is not None else {}),
                    "cb0_flipk": round(tdc.flipk_rate(codes[0]), 2),
                    **({"loop_break": lb.stats()} if lb is not None else {}),
                    **({"silence_guard": sg.stats()} if sg is not None else {}),
                    **plan_extra,
                    **stop_extra,
                }) + "\n")
                n += 1
                f.flush()
                extra = f" think={rstr[:40]!r}" if rstr else ""
                if plan_extra:
                    extra += (f" plan={plan_extra['plan_status']}/"
                              f"{plan_extra['plan_n_words']}w "
                              f"win={min_fr}-{max_fr}")
                print(f"[row {j} c{c}] {len(codes[0])} fr  "
                      f"emo={r.get('emotion_label')}  {r['prompt'][:34]!r}{extra}",
                      flush=True)
    if plan_mode:
        print("PLAN_STATUS_SUMMARY " + (" ".join(
            f"{k}={v}" for k, v in sorted(plan_counts.items())) or "none"),
            flush=True)
    print(f"GEN_FISH_COT_SAMPLES_DONE mode={args.mode} wrote {n} -> {out_path}")

if __name__ == "__main__":
    main()
