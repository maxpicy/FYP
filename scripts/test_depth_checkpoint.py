# test_depth_checkpoint.py: Frame-by-frame decoding with the depth module (per-level sampling, cached state,
# plan schedule) and the SilenceGuard.

import argparse
import json
import os
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

NUM_LEVELS = int(os.environ.get("MVC_NUM_CODEC_LEVELS", "8"))
CODEBOOK_SIZE = 2048

DEFAULT_PROMPTS = [
    "The quick brown fox jumps over the lazy dog.",
    "It was the best of times, it was the worst of times.",
    "Good morning. I hope you slept well last night.",
]


@torch.no_grad()
def trim_trailing_silence(wav, sr, thresh_db=-40.0, keep_ms=120):
    import numpy as np
    w = np.asarray(wav, dtype=np.float32)
    mono = w.mean(axis=1) if w.ndim > 1 else w
    peak = float(np.max(np.abs(mono))) + 1e-9
    thr = peak * (10.0 ** (thresh_db / 20.0))
    above = np.abs(mono) > thr
    if not above.any():
        return wav
    last = int(np.nonzero(above)[0][-1])
    keep = min(len(mono), last + int(sr * keep_ms / 1000))
    return w[:keep]


@torch.no_grad()
def think_generate(model, prefix, registry, tokenizer, max_think_tokens=200,
                   temp=0.7, speaker_id=None, device="cuda"):
    backbone = model.backbone.backbone
    emb = backbone.embedding
    inject = None
    speaker_emb = None
    if speaker_id is not None and getattr(model, "use_speaker_conditioning", False):
        sid = torch.tensor([speaker_id], dtype=torch.long, device=device)
        speaker_emb = model.speaker_encoder(sid)
        if getattr(model, "use_speaker_input", False):
            inject = model.speaker_input_proj(speaker_emb)

    ids = prefix.clone().to(device)
    think_ids = []
    ban_lo, ban_hi = registry.speech_token_id_min, registry.speech_token_id_max
    banned_ctl = [registry.audio_start_id, registry.audio_end_id,
                  registry.eos_id, registry.think_start_id]
    for _ in range(max_think_tokens):
        if inject is not None:
            emb.set_speaker(inject)
        emb.set_codes(torch.full((1, ids.shape[1], NUM_LEVELS), -1,
                                 dtype=torch.long, device=device))
        h = backbone(ids)
        if speaker_emb is not None:
            h = model.adaln(h, speaker_emb)
        logits = model.backbone.lm_head(h[:, -1, :])[0].float()
        logits[ban_lo:ban_hi + 1] = float("-inf")
        for t in banned_ctl:
            logits[t] = float("-inf")
        if temp > 0.0:
            nxt = int(torch.multinomial(
                torch.softmax(logits / temp, dim=-1), 1).item())
        else:
            nxt = int(logits.argmax().item())
        if nxt == registry.think_end_id:
            break
        think_ids.append(nxt)
        ids = torch.cat([ids, torch.tensor([[nxt]], device=device)], dim=1)
    tail = torch.tensor([[registry.think_end_id, registry.audio_start_id]],
                        device=device)
    ids = torch.cat([ids, tail], dim=1)
    reasoning = tokenizer.decode(think_ids) if think_ids else ""
    return ids, reasoning


def _half_up(v) -> int:
    return int(float(v) + 0.5)


def build_decode_plan_schedule(durations, prefix_len, max_frames, *,
                               mode="stretch", target_frames=None, device=None):
    from delay_dataset import NO_WORD, build_plan_schedule as _frame_map
    from model import build_plan_schedule as _position_schedule

    if prefix_len < 1:
        raise ValueError(f"prefix_len must be >= 1 (the <AUDIO> token sits at "
                         f"prefix_len - 1), got {prefix_len}")
    if max_frames < 1:
        raise ValueError(f"max_frames must be >= 1, got {max_frames}")
    durs = [_half_up(d) for d in durations]
    if not durs:
        raise ValueError("plan has no words — nothing to schedule")
    if any(d < 0 for d in durs):
        raise ValueError(f"negative word duration in plan: {durs}")
    if sum(durs) <= 0:
        raise ValueError("plan durations sum to 0 frames — the keystone token "
                         "(§3.1) is degenerate; refuse rather than schedule "
                         "every frame onto word 0")

    if mode == "stretch":
        if target_frames is None:
            raise ValueError(
                "mode='stretch' needs target_frames (the expected TOTAL frame "
                "count). ΣD counts speech frames only, so the caller must "
                "supply ΣD̂ / ρ with ρ the corpus speech/total frame ratio — or "
                "use mode='pack', which needs no estimate.")
        target_frames = int(target_frames)
        if target_frames < 1:
            raise ValueError(f"target_frames must be >= 1, got {target_frames}")
        frame_to_word = _frame_map(durs, target_frames, "stretch")
        word_frames = [0] * len(durs)
        for w in frame_to_word:
            if w != NO_WORD:
                word_frames[w] += 1
    elif mode == "pack":
        word_frames = durs
    else:
        raise ValueError(f"plan schedule mode {mode!r} not in ('stretch', 'pack')")

    return _position_schedule([word_frames], audio_start=prefix_len - 1,
                              n_frames=max_frames,
                              seq_len=prefix_len + max_frames, device=device)


def _fresh_inference_cache(max_seqlen, max_batch_size=1):
    from mamba_ssm.utils.generation import InferenceParams
    return InferenceParams(max_seqlen=int(max_seqlen), max_batch_size=int(max_batch_size))


def _check_plan_decode(model, plan_bins, plan_schedule, prefix_len, max_frames):
    if not getattr(model, "use_plan_injection", False):
        raise RuntimeError(
            "plan_bins/plan_schedule were passed but this model has no plan "
            "injection channel — call model.enable_plan_injection() BEFORE "
            "load_checkpoint so the checkpoint's plan_injector.* weights land "
            "in it. Without the call the tensors would be dropped and the run "
            "would silently measure the prefix-only condition.")
    if model.training:
        raise RuntimeError(
            "plan-injected decode requires model.eval(): PlanInjector's "
            "scaffold dropout keys off self.training and would drop the "
            "injection on a random ~scaffold_dropout share of utterances.")
    if plan_bins.dim() != 3 or plan_bins.shape[0] != 1 or plan_bins.shape[-1] != 3:
        raise ValueError(f"plan_bins must be [1, W, 3] (pitch, duration, "
                         f"energy), got {tuple(plan_bins.shape)}")
    if plan_schedule.dim() != 2 or plan_schedule.shape[0] != 1:
        raise ValueError(f"plan_schedule must be [1, S], got "
                         f"{tuple(plan_schedule.shape)}")
    need = prefix_len + max_frames - 1
    if plan_schedule.shape[1] < need:
        raise ValueError(
            f"plan_schedule covers {plan_schedule.shape[1]} positions but the "
            f"decode can reach position {need - 1} (prefix {prefix_len} + "
            f"{max_frames} frames) — build it with build_decode_plan_schedule("
            f"..., prefix_len={prefix_len}, max_frames={max_frames}).")
    if bool((plan_schedule[0, :prefix_len - 1] >= 0).any()):
        raise ValueError(
            "plan_schedule injects at a position before the <AUDIO> token "
            f"(prefix_len - 1 = {prefix_len - 1}). The program conditions the "
            "AUDIO block only; a schedule reaching the prompt/<THINK> region "
            "was built against a different prefix length.")
    if int(plan_schedule[0, prefix_len - 1]) < 0:
        raise ValueError(
            "plan_schedule has no word on the <AUDIO> token at position "
            f"{prefix_len - 1}, which is where frame 0's word belongs "
            "(delay_dataset plan_lookahead=1). The schedule is off by at least "
            "one position relative to training.")


def flipk_rate(cb0, k_lo=2, k_hi=8):
    n = len(cb0)
    hits = tot = 0
    for k in range(k_lo, k_hi + 1):
        if n <= k:
            continue
        for t in range(k, n):
            if cb0[t] == cb0[t - k] and cb0[t] != cb0[t - 1]:
                hits += 1
        tot += n - k
    return 100.0 * hits / tot if tot else 0.0


class LoopBreaker:
    def __init__(self, threshold=5.0, win=32, temp=0.6, rep=2.0,
                 stop_after=0):
        self.threshold = float(threshold)
        self.win = int(win)
        self.temp = float(temp)
        self.rep = float(rep)
        self.stop_after = int(stop_after)
        self.hist = []
        self.active = False
        self.metric = 0.0
        self.max_metric = 0.0
        self.events = 0
        self.frames_intervened = 0
        self.run = 0
        self.stopped = False

    def update(self, cb0_code):
        self.hist.append(int(cb0_code))
        if len(self.hist) > self.win:
            del self.hist[0]
        if len(self.hist) < self.win:
            return
        self.metric = flipk_rate(self.hist)
        self.max_metric = max(self.max_metric, self.metric)
        was = self.active
        self.active = self.metric >= self.threshold
        if self.active:
            if not was:
                self.events += 1
            self.run += 1
            self.frames_intervened += 1
        else:
            self.run = 0

    @property
    def force_stop(self):
        return bool(self.stop_after) and self.run >= self.stop_after

    def rewind(self, n):
        if n > 0:
            del self.hist[-n:]
        self.run = 0
        self.active = False

    def stats(self):
        return {"events": self.events,
                "frames": self.frames_intervened,
                "max_flipk": round(self.max_metric, 2),
                "stopped": self.stopped}


class SilenceGuard:
    def __init__(self, codes, budget=10, stop_after=10, stop_without_plan=False,
                 retries=0, retry_temp=0.6):
        self.codes = frozenset(int(c) for c in codes)
        if not self.codes:
            raise ValueError("SilenceGuard needs a non-empty pause-code set")
        self.budget = int(budget)
        self.stop_after = int(stop_after)
        self.stop_without_plan = bool(stop_without_plan)
        self.retries = int(retries)
        self.retry_temp = float(retry_temp)
        self.run = 0
        self.max_run = 0
        self.mask_events = 0
        self.frames_masked = 0
        self.stopped = False
        self.retries_used = 0
        self.rewinds = 0
        self.frames_rewound = 0
        self.mask_left = 0
        self.spoke_at_last = False
        self._masking = False
        self._idx = {}

    def update(self, cb0_code, at_last=None):
        if int(cb0_code) in self.codes:
            self.run += 1
            self.max_run = max(self.max_run, self.run)
        else:
            self.run = 0
            if at_last is True:
                self.spoke_at_last = True

    def plan_rewind(self, at_last):
        if (self.budget <= 0 or self.retries_used >= self.retries
                or at_last is True or self.run < self.budget):
            return 0
        n = self.run
        self.retries_used += 1
        self.rewinds += 1
        self.frames_rewound += n
        self.run = 0
        self.mask_left = self.budget
        return n

    @property
    def kicking(self):
        return self.mask_left > 0

    def mask_now(self, at_last):
        if self.mask_left > 0:
            self.mask_left -= 1
            if not self._masking:
                self.mask_events += 1
            self.frames_masked += 1
            self._masking = True
            return True
        want = (self.budget > 0 and self.run >= self.budget
                and at_last is not True)
        if want:
            if not self._masking:
                self.mask_events += 1
            self.frames_masked += 1
        self._masking = want
        return want

    def force_stop(self, at_last):
        if not self.stop_after or self.run < self.stop_after:
            return False
        if at_last is True:
            exhausted = self.retries > 0 and self.retries_used >= self.retries
            return self.spoke_at_last or exhausted
        return at_last is None and self.stop_without_plan

    def mask_index(self, device):
        key = str(device)
        if key not in self._idx:
            self._idx[key] = torch.tensor(sorted(self.codes), dtype=torch.long,
                                          device=device)
        return self._idx[key]

    def stats(self):
        return {"mask_events": self.mask_events,
                "frames_masked": self.frames_masked,
                "max_run": self.max_run,
                "stopped": self.stopped,
                "rewinds": self.rewinds,
                "frames_rewound": self.frames_rewound,
                "retries_used": self.retries_used,
                "spoke_at_last": self.spoke_at_last}


def depth_generate(model, prefix, registry, max_frames=200, level_temps=None,
                   top_k=0, top_p=0.0, allow_stop=True, min_frames=20,
                   stop_threshold=0.0, sticky_bias=0.0,
                   speaker_id=None, feedback="all", cross_frame_decode=True,
                   device="cuda", antifan_bias=0.0, antifan_levels=None,
                   onset_frames=0, onset_temp_scale=1.0,
                   rep_penalty=1.0, rep_win=16, ras_tau=0.0, ras_win=10,
                   stop_prob_out=None, plan_bins=None, plan_schedule=None,
                   loop_breaker=None, silence_guard=None, cached=False):
    placeholder = registry.audio_start_id
    audio_end = registry.audio_end_id
    eos = registry.eos_id
    backbone = model.backbone.backbone
    emb = backbone.embedding
    speaker_emb = None
    inject = None
    if speaker_id is not None and getattr(model, "use_speaker_conditioning", False):
        sid = torch.tensor([speaker_id], dtype=torch.long, device=device)
        speaker_emb = model.speaker_encoder(sid)
        if getattr(model, "use_speaker_input", False):
            inject = model.speaker_input_proj(speaker_emb)

    if (plan_bins is None) != (plan_schedule is None):
        raise ValueError(
            "plan_bins and plan_schedule must be supplied together (got "
            f"plan_bins={'set' if plan_bins is not None else 'None'}, "
            f"plan_schedule={'set' if plan_schedule is not None else 'None'}) "
            "— half a plan is a wiring bug whose failure mode is a silent null.")
    if plan_bins is not None:
        _check_plan_decode(model, plan_bins, plan_schedule, prefix.shape[1],
                           max_frames)

    ids = prefix.clone().to(device)
    codes = torch.full((1, ids.shape[1], NUM_LEVELS), -1,
                       dtype=torch.long, device=device)
    frames = []
    iters = max_frames
    if silence_guard is not None and silence_guard.retries > 0:
        iters += silence_guard.retries * (silence_guard.budget + 1)
    ip = None
    n_fed = 0
    if cached:
        ip = _fresh_inference_cache(ids.shape[1] + iters + 8)
    for _ in range(iters):
        if len(frames) >= max_frames:
            break
        if inject is not None:
            emb.set_speaker(inject)
        lo = n_fed if cached else 0
        emb.set_codes(codes[:, lo:])
        at_last = None
        if plan_bins is not None:
            plan_vec = model.plan_injector(plan_bins, plan_schedule[:, :ids.shape[1]])
            emb.set_plan(plan_vec[:, lo:])
            cur_word = int(plan_schedule[0, ids.shape[1] - 1].item())
            at_last = (cur_word >= plan_bins.shape[1] - 1) if cur_word >= 0 else False
        if silence_guard is not None:
            nb = silence_guard.plan_rewind(at_last)
            if nb:
                nb = min(nb, len(frames))
                if nb:
                    frames = frames[:-nb]
                    ids = ids[:, :-nb]
                    codes = codes[:, :-nb]
                    if loop_breaker is not None:
                        loop_breaker.rewind(nb)
                    if cached:
                        ip = _fresh_inference_cache(ids.shape[1] + iters + 8)
                        n_fed = 0
                continue
        if cached:
            ip.seqlen_offset = n_fed
            h = backbone(ids[:, lo:], inference_params=ip)
            n_fed = ids.shape[1]
        else:
            h = backbone(ids)
        if speaker_emb is not None:
            h = model.adaln(h, speaker_emb)
        last = h[:, -1, :]
        may_stop = allow_stop and len(frames) >= min_frames
        if may_stop or stop_prob_out is not None:
            logits = model.backbone.lm_head(last)[0]
            probs = torch.softmax(logits.float(), dim=-1)
            p_stop = float(probs[audio_end].item() + probs[eos].item())
            need_argmax = (stop_prob_out is not None) or stop_threshold <= 0.0
            argmax_stop = (int(logits.argmax().item()) in (audio_end, eos)
                           if need_argmax else False)
            if stop_prob_out is not None:
                stop_prob_out.append((p_stop, argmax_stop))
            if may_stop:
                if stop_threshold > 0.0:
                    if p_stop > stop_threshold:
                        break
                elif argmax_stop:
                    break
        lt = level_temps
        if onset_frames and lt is not None and len(frames) < onset_frames:
            lt = [lt[0]] + [t * onset_temp_scale for t in lt[1:]]
        rp_eff = rep_penalty
        if loop_breaker is not None and loop_breaker.active:
            if lt is not None:
                lt = [max(loop_breaker.temp, lt[0])] + list(lt[1:])
            else:
                lt = [loop_breaker.temp] + [0.0] * (NUM_LEVELS - 1)
            if rp_eff is None or isinstance(rp_eff, (int, float)):
                base = float(rp_eff) if rp_eff else 1.0
                rp_eff = ([max(loop_breaker.rep, base)]
                          + [base] * (NUM_LEVELS - 1))
            else:
                rp_eff = ([max(loop_breaker.rep, float(rp_eff[0]))]
                          + list(rp_eff[1:]))
        if silence_guard is not None and silence_guard.kicking:
            if lt is not None:
                lt = [max(silence_guard.retry_temp, lt[0])] + list(lt[1:])
            else:
                lt = [silence_guard.retry_temp] + [0.0] * (NUM_LEVELS - 1)
        rw = None
        _rp_on = (any(float(x) != 1.0 for x in rep_penalty)
                  if rep_penalty is not None
                  and not isinstance(rep_penalty, (int, float))
                  else bool(rep_penalty) and rep_penalty != 1.0)
        _lb_on = loop_breaker is not None and loop_breaker.active
        if (_rp_on or ras_tau or _lb_on) and frames:
            recent = frames[-rep_win:]
            rw = [[f[k] for f in recent] for k in range(NUM_LEVELS)]
        frame = model.depth_module.generate_frame(
            last, level_temps=lt, top_k=top_k, top_p=top_p,
            return_argmax=(feedback == "argmax"),
            rep_window=rw, rep_penalty=rp_eff,
            ras_tau=ras_tau, ras_win=ras_win,
            sticky_prev=(frames[-1] if ((sticky_bias or antifan_bias) and frames)
                         else None),
            sticky_bias=sticky_bias,
            prev_codes=(frames[-1] if (frames and cross_frame_decode)
                        else None),
            antifan_prev2=(frames[-2] if (antifan_bias and len(frames) >= 2)
                           else None),
            antifan_bias=antifan_bias, antifan_levels=antifan_levels,
            cb0_mask=(silence_guard.mask_index(device)
                      if (silence_guard is not None
                          and silence_guard.mask_now(at_last)) else None))
        frame_amax = None
        if feedback == "argmax":
            frame, frame_amax = frame
        frames.append(frame)
        if loop_breaker is not None:
            loop_breaker.update(frame[0])
            if loop_breaker.force_stop and len(frames) >= min_frames:
                loop_breaker.stopped = True
                break
        if silence_guard is not None:
            silence_guard.update(frame[0], at_last=at_last)
            if silence_guard.force_stop(at_last) and len(frames) >= min_frames:
                silence_guard.stopped = True
                break
        fb = list(frame)
        if feedback == "semantic":
            fb = [frame[0]] + [-1] * (NUM_LEVELS - 1)
        elif feedback == "argmax":
            fb = list(frame_amax)
        col = torch.tensor(fb, dtype=torch.long, device=device).view(1, 1, NUM_LEVELS)
        ids = torch.cat([ids, torch.tensor([[placeholder]], device=device)], dim=1)
        codes = torch.cat([codes, col], dim=1)
    if not frames:
        return None
    return [[f[k] for f in frames] for k in range(NUM_LEVELS)]


@torch.no_grad()
def flow_generate(model, prefix, registry, max_frames=200, allow_stop=True,
                  min_frames=20, stop_threshold=0.0, speaker_id=None,
                  feedback="all", flow_steps=16, flow_noise=1.0, cb0_temp=0.0,
                  device="cuda"):
    placeholder = registry.audio_start_id
    audio_end = registry.audio_end_id
    eos = registry.eos_id
    backbone = model.backbone.backbone
    emb = backbone.embedding
    speaker_emb = None
    inject = None
    if speaker_id is not None and getattr(model, "use_speaker_conditioning", False):
        sid = torch.tensor([speaker_id], dtype=torch.long, device=device)
        speaker_emb = model.speaker_encoder(sid)
        if getattr(model, "use_speaker_input", False):
            inject = model.speaker_input_proj(speaker_emb)

    ids = prefix.clone().to(device)
    codes = torch.full((1, ids.shape[1], NUM_LEVELS), -1,
                       dtype=torch.long, device=device)
    latents = []
    for _ in range(max_frames):
        if inject is not None:
            emb.set_speaker(inject)
        emb.set_codes(codes)
        h = backbone(ids)
        if speaker_emb is not None:
            h = model.adaln(h, speaker_emb)
        last = h[:, -1, :]
        if allow_stop and len(latents) >= min_frames:
            logits = model.backbone.lm_head(last)[0]
            if stop_threshold > 0.0:
                probs = torch.softmax(logits.float(), dim=-1)
                if float(probs[audio_end].item() + probs[eos].item()) > stop_threshold:
                    break
            elif int(logits.argmax().item()) in (audio_end, eos):
                break
        frame = model.depth_module.generate_frame(
            last, level_temps=[cb0_temp] + [0.0] * (NUM_LEVELS - 1))
        cb0 = torch.tensor([frame[0]], device=device)
        lat = model.flow_head.sample(last, cb0, steps=flow_steps,
                                     noise_scale=flow_noise)
        latents.append(lat[0])
        fb = list(frame)
        if feedback == "semantic":
            fb = [frame[0]] + [-1] * (NUM_LEVELS - 1)
        col = torch.tensor(fb, dtype=torch.long, device=device).view(1, 1, NUM_LEVELS)
        ids = torch.cat([ids, torch.tensor([[placeholder]], device=device)], dim=1)
        codes = torch.cat([codes, col], dim=1)
    if not latents:
        return None
    return torch.stack(latents, dim=0).transpose(0, 1).unsqueeze(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model_name", default="state-spaces/mamba2-1.3b")
    ap.add_argument("--lora_rank", type=int, default=0)
    ap.add_argument("--num_speakers", type=int, default=20000)
    ap.add_argument("--speaker_dim", type=int, default=256)
    ap.add_argument("--speaker_input_injection", action="store_true",
                    help="MUST match training (all depth recipes pass it)")
    ap.add_argument("--speaker_id", type=int, default=None)
    ap.add_argument("--depth_dim", type=int, default=1024)
    ap.add_argument("--depth_layers", type=int, default=2)
    ap.add_argument("--no_cross_frame_decode", action="store_true",
                    help="S32 residual-loop diag: disable the cf3 cross-frame "
                         "channel at DECODE time (every frame uses the learned "
                         "no-predecessor cross_start vector) — fully opens the "
                         "acoustic loop on a cross_frame-trained checkpoint")
    ap.add_argument("--depth_feedback", choices=["all", "semantic"], default="all",
                    help="MUST match the checkpoint's training mode")
    ap.add_argument("--depth_cross_frame", type=int, default=0,
                    help="MUST match the checkpoint's --depth_cross_frame "
                         "(S30 cross-frame acoustic channel; 3 = fan fix)")
    ap.add_argument("--temp_semantic", type=float, default=0.0)
    ap.add_argument("--temp_acoustic", type=float, default=None)
    ap.add_argument("--acoustic_temps", type=float, nargs=7, default=None)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--top_p", type=float, default=0.0)
    ap.add_argument("--best_of", type=int, default=1)
    ap.add_argument("--bestof_semantic_temps", type=float, nargs="+",
                    default=[0.1, 0.3, 0.5, 0.7])
    ap.add_argument("--max_frames", type=int, default=200)
    ap.add_argument("--min_frames", type=int, default=20)
    ap.add_argument("--min_frames_per_word", type=float, default=0.0,
                    help="per-prompt stop floor = max(min_frames, "
                         "words * this). ~4.5 (0.36 s/word) prevents the "
                         "first-sentence EOS truncation on multi-sentence "
                         "prompts (S2 arms learned utterance-final stop)")
    ap.add_argument("--allow_stop", action="store_true")
    ap.add_argument("--stop_threshold", type=float, default=0.0,
                    help=">0: honor AUDIO_END/EOS only when P(stop) exceeds this "
                         "(softmax over lm_head); 0 = bare-argmax stop (legacy)")
    ap.add_argument("--sticky_bias", type=float, default=0.0,
                    help="S28 fan mitigation: logit bonus for the previous "
                         "frame's code at each acoustic level (cb1-7) — "
                         "decode-side temporal texture persistence. Try 1-3")
    ap.add_argument("--trim_silence", action="store_true",
                    help="trim trailing near-silence from decoded audio")
    ap.add_argument("--whisper", default="openai/whisper-base.en")
    ap.add_argument("--prompts", nargs="*", default=None)
    ap.add_argument("--prompts_from", default=None,
                    help="read prompts from a JSONL (6-20 word rows) instead of "
                         "the 3 default sentences — for a larger naturalness n")
    ap.add_argument("--num_prompts", type=int, default=15)
    ap.add_argument("--outdir", default="out_depth_eval")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--flow", action="store_true",
                    help="PB-14 option A: render via the continuous flow head "
                         "(RVQ-recon latent -> decode_latents) instead of the "
                         "discrete codebooks. Requires a --flow_head checkpoint.")
    ap.add_argument("--flow_hidden", type=int, default=1024)
    ap.add_argument("--flow_layers", type=int, default=4)
    ap.add_argument("--flow_steps", type=int, default=16,
                    help="Euler ODE steps for flow sampling")
    ap.add_argument("--hybrid_attention_top_k", type=int, default=0,
                    help="Phase 8: load a hybrid checkpoint (top-K mixers are "
                         "attention). Must match the trained value (4).")
    ap.add_argument("--cot", action="store_true",
                    help="Stage-2 CoT decode: free-generate the <THINK> block "
                         "(speech/audio tokens banned), then constrained depth "
                         "audio. Requires a stage2_cot_* checkpoint.")
    ap.add_argument("--max_think_tokens", type=int, default=200)
    ap.add_argument("--think_temp", type=float, default=0.7)
    ap.add_argument("--flow_noise", type=float, default=1.0,
                    help="flow start-noise scale (1=full N(0,I); 0=deterministic "
                         "from origin -> endpoint nearer the RVQ lattice)")
    args = ap.parse_args()
    prompts = args.prompts or DEFAULT_PROMPTS
    if args.prompts_from:
        import json as _json
        pf = []
        with open(args.prompts_from, encoding="utf-8") as _fh:
            for _line in _fh:
                _t = _json.loads(_line).get("prompt", "").strip()
                if 6 <= len(_t.split()) <= 20:
                    pf.append(_t)
                if len(pf) >= args.num_prompts:
                    break
        prompts = pf or prompts

    import config as _cfg
    import soundfile as sf
    from model import MambaCoTModel
    from train import load_checkpoint, apply_lora
    from inference import build_inference_prefix
    from codec import load_codec
    from eval.intelligibility import IntelligibilityScorer
    assert _cfg.NUM_SPEECH_TOKENS == CODEBOOK_SIZE, "run with MVC_NUM_SPEECH_TOKENS=2048"

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    model = MambaCoTModel(model_name=args.model_name, device=args.device,
                          dtype=torch.bfloat16, mtp_num_heads=0)
    model.enable_speaker_conditioning(speaker_dim=args.speaker_dim,
                                      num_speakers=args.num_speakers,
                                      input_injection=args.speaker_input_injection)
    if args.lora_rank > 0:
        apply_lora(model, rank=args.lora_rank, alpha=args.lora_rank,
                   target_modules=["in_proj", "out_proj"], dropout=0.0)
    model.enable_depth_module(num_levels=NUM_LEVELS, codebook_size=CODEBOOK_SIZE,
                              d_depth=args.depth_dim,
                              depth_layers=args.depth_layers,
                              depth_feedback=args.depth_feedback,
                              cross_frame=args.depth_cross_frame)
    if args.hybrid_attention_top_k > 0:
        model.enable_hybrid_attention(top_k=args.hybrid_attention_top_k)
    if args.flow:
        model.enable_flow_head(latent_dim=512, cb0_size=CODEBOOK_SIZE,
                               d_hidden=args.flow_hidden, n_layers=args.flow_layers)
    load_checkpoint(args.checkpoint, model, device=args.device)
    model.eval()
    registry = model.token_registry
    codec = load_codec("mimi", device=args.device)
    scorer = IntelligibilityScorer(model_id=args.whisper, device=args.device)

    results = []
    for i, prompt in enumerate(prompts):
        print(f"\n[{i}] prompt: {prompt!r}", flush=True)
        if args.cot:
            p2 = build_inference_prefix(model.tokenizer, registry, prompt, 2)
            prefix, reasoning = think_generate(
                model, p2, registry, model.tokenizer,
                max_think_tokens=args.max_think_tokens, temp=args.think_temp,
                speaker_id=args.speaker_id, device=args.device)
            print(f"    think ({prefix.shape[1] - p2.shape[1] - 2} tok): "
                  f"{reasoning!r}", flush=True)
        else:
            prefix = build_inference_prefix(model.tokenizer, registry, prompt, 1)
        min_fr = max(args.min_frames,
                     int(len(prompt.split()) * args.min_frames_per_word))
        best = None
        for c in range(max(1, args.best_of)):
            sem_t = (args.temp_semantic if args.best_of <= 1
                     else args.bestof_semantic_temps[c % len(args.bestof_semantic_temps)])
            if args.acoustic_temps is not None:
                lts = [sem_t] + list(args.acoustic_temps)
            else:
                ac = args.temp_acoustic if args.temp_acoustic is not None else 0.0
                lts = [sem_t] + [ac] * (NUM_LEVELS - 1)
            if args.flow:
                latents = flow_generate(
                    model, prefix, registry, max_frames=args.max_frames,
                    allow_stop=args.allow_stop, min_frames=min_fr,
                    stop_threshold=args.stop_threshold, speaker_id=args.speaker_id,
                    feedback=args.depth_feedback, flow_steps=args.flow_steps,
                    flow_noise=args.flow_noise,
                    cb0_temp=sem_t, device=args.device)
                if latents is None:
                    continue
                wav, sr = codec.decode_latents(latents)
                nframes = latents.shape[-1]
            else:
                codes = depth_generate(
                    model, prefix, registry, max_frames=args.max_frames,
                    level_temps=lts, top_k=args.top_k, top_p=args.top_p,
                    allow_stop=args.allow_stop, min_frames=min_fr,
                    stop_threshold=args.stop_threshold,
                    sticky_bias=args.sticky_bias,
                    speaker_id=args.speaker_id, feedback=args.depth_feedback,
                    cross_frame_decode=not args.no_cross_frame_decode,
                    device=args.device)
                if codes is None:
                    continue
                wav, sr = codec.decode_multi(codes)
                nframes = len(codes[0])
            if args.trim_silence:
                wav = trim_trailing_silence(wav, sr)
            tmp = outdir / f"_cand_{i:02d}_{c}.wav"
            sf.write(str(tmp), wav, sr)
            score = scorer.score(str(tmp), prompt)
            tmp.unlink(missing_ok=True)
            wer = score["wer"]
            cand = {"prompt": prompt, "frames": nframes, "sem_temp": sem_t,
                    "asr": score["hypothesis"],
                    "wer": round(wer, 4) if wer == wer else None,
                    "_wav": wav, "_sr": sr}
            print(f"    cand {c} semT={sem_t} frames={cand['frames']} "
                  f"wer={cand['wer']} asr={cand['asr']!r}", flush=True)
            if best is None or (cand["wer"] is not None and
                                (best["wer"] is None or cand["wer"] < best["wer"])):
                best = cand
        if best is None:
            print("    (no frames)"); continue
        if args.cot:
            best["reasoning"] = reasoning
        sf.write(str(outdir / f"sample_{i:02d}.wav"), best.pop("_wav"), best.pop("_sr"))
        results.append(best)
        print(f"    BEST wer={best['wer']}")

    (outdir / "intelligibility_depth.json").write_text(json.dumps(results, indent=2))
    wers = [r["wer"] for r in results if r["wer"] is not None]
    if wers:
        print(f"\ndepth-module mean WER: {sum(wers)/len(wers):.3f}")
    print("TEST_DEPTH_DONE")

if __name__ == "__main__":
    main()
