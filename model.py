# model.py: MambaCoTModel: the backbone (pure Mamba-2, the hybrid attention graft, or the transformer), speaker
# input injection, the plan injector, and the Mamba-2 depth module that generates codebooks 1-9 per frame.

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from mamba_ssm import MambaLMHeadModel

from tokenizer import expand_tokenizer, TokenRegistry, load_base_tokenizer, NUM_SPEECH_TOKENS

try:
    from torch.nn import RMSNorm
except ImportError:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm


class AdaLN(nn.Module):
    def __init__(self, d_model: int, speaker_dim: int = 256):
        super().__init__()
        self.proj = nn.Linear(speaker_dim, 2 * d_model)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, speaker_emb: torch.Tensor) -> torch.Tensor:
        scale, shift = self.proj(speaker_emb).chunk(2, dim=-1)
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class SpeakerEncoder(nn.Module):
    def __init__(self, num_speakers: int = 1000, speaker_dim: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(num_speakers, speaker_dim)

    def forward(self, speaker_ids: torch.LongTensor) -> torch.Tensor:
        return self.embedding(speaker_ids)


class DelayEmbedding(nn.Module):
    def __init__(self, base: nn.Embedding, num_levels: int = 8,
                 codebook_size: int = 2048, codebook_sizes=None):
        super().__init__()
        self.base = base
        self.num_levels = num_levels
        self.codebook_size = codebook_size
        self.level_sizes = (list(codebook_sizes) if codebook_sizes
                            else [codebook_size] * num_levels)
        assert len(self.level_sizes) == num_levels
        ref = base.weight
        self.level_embeds = nn.ModuleList([
            nn.Embedding(self.level_sizes[k] + 1, ref.shape[1],
                         device=ref.device, dtype=ref.dtype)
            for k in range(num_levels)
        ])
        for e in self.level_embeds:
            nn.init.zeros_(e.weight)
        self._codes = None
        self._speaker = None
        self._plan = None

    @property
    def weight(self):
        return self.base.weight

    def set_codes(self, codes):
        self._codes = codes

    def set_speaker(self, vec):
        self._speaker = vec

    def set_plan(self, vec):
        self._plan = vec

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        emb = self.base(input_ids)
        codes = self._codes
        self._codes = None
        if codes is not None:
            valid = codes[..., 0] >= 0
            for k in range(self.num_levels):
                idx = (codes[..., k] + 1).clamp(min=0)
                emb = emb + self.level_embeds[k](idx) * valid.unsqueeze(-1)
        plan = self._plan
        self._plan = None
        if plan is not None:
            emb = emb + plan
        spk = self._speaker
        self._speaker = None
        if spk is not None:
            emb = emb + spk.unsqueeze(1)
        return emb

PLAN_BIN_CHANNELS = ("pitch", "duration", "energy")
PLAN_BIN_ABSENT = -1

_PLAN_WITHOUT_INJECTION_WARNED = False


def _warn_plan_without_injection() -> None:
    global _PLAN_WITHOUT_INJECTION_WARNED
    if not _PLAN_WITHOUT_INJECTION_WARNED:
        _PLAN_WITHOUT_INJECTION_WARNED = True
        print("[plan-injection] WARNING: plan tensors were supplied but plan "
              "injection is DISABLED on this model (enable_plan_injection() "
              "was never called) — the plan reaches the model only through the "
              "<THINK> prefix. Intended for the injection-OFF arm; if this run "
              "was meant to inject, the flag is missing.", flush=True)


def build_plan_schedule(plan_durations, audio_start, n_frames, seq_len=None,
                        device=None) -> torch.LongTensor:
    rows = _plan_duration_rows(plan_durations)
    B = len(rows)
    starts = _as_int_vector(audio_start, B, "audio_start")
    counts = _as_int_vector(n_frames, B, "n_frames")
    for b in range(B):
        if starts[b] < 0:
            raise ValueError(f"audio_start[{b}]={starts[b]} is negative — "
                             f"audio_start is the first AUDIO FRAME position")
        if counts[b] < 0:
            raise ValueError(f"n_frames[{b}]={counts[b]} is negative")

    need = max((starts[b] + counts[b] for b in range(B)), default=0)
    if seq_len is None:
        seq_len = need
    else:
        seq_len = int(seq_len)
        if need > seq_len:
            raise ValueError(
                f"audio block runs past seq_len: max(audio_start + n_frames)="
                f"{need} > seq_len={seq_len}. Pass the COLLATED sequence length "
                f"and the per-row audio_start/n_frames of that same batch.")

    sched = torch.full((B, seq_len), -1, dtype=torch.long, device=device)
    for b in range(B):
        durs, a0, nf = rows[b], starts[b], counts[b]
        if not durs or nf == 0:
            continue
        ends = torch.tensor(durs, dtype=torch.long, device=device).cumsum(0)
        last_real = max((w for w, d in enumerate(durs) if d > 0), default=-1)
        if last_real < 0:
            continue
        f = torch.arange(nf, dtype=torch.long, device=device)
        idx = torch.searchsorted(ends, f, right=True).clamp(max=last_real)
        sched[b, a0:a0 + nf] = idx
    return sched


def _plan_duration_rows(plan_durations):
    if hasattr(plan_durations, "ndim") and hasattr(plan_durations, "tolist"):
        if int(plan_durations.ndim) != 2:
            raise ValueError(f"plan_durations array must be [B, W], got "
                             f"{int(plan_durations.ndim)} dimension(s)")
        raw = plan_durations.tolist()
    else:
        raw = list(plan_durations)
        if raw and isinstance(raw[0], (int, float)):
            raise ValueError("plan_durations must be a list of PER-ROW duration "
                             "lists (or a [B, W] array) — got a flat sequence")
        raw = [list(r) for r in raw]

    rows = []
    for b, row in enumerate(raw):
        out, padding = [], False
        for w, v in enumerate(row):
            v = float(v)
            if v < 0:
                if v != PLAN_BIN_ABSENT:
                    raise ValueError(
                        f"plan_durations[{b}][{w}]={v}: the only accepted "
                        f"negative is the pad sentinel {PLAN_BIN_ABSENT}")
                padding = True
                continue
            if padding:
                raise ValueError(
                    f"plan_durations[{b}]: pad sentinel at an interior position "
                    f"(word {w} is a real duration after padding started) — "
                    f"padding must be a trailing run")
            out.append(int(v + 0.5))
        rows.append(out)
    return rows


def _as_int_vector(x, B: int, name: str):
    if hasattr(x, "tolist"):
        x = x.tolist()
    try:
        iter(x)
    except TypeError:
        return [int(x)] * B
    vals = [int(v) for v in x]
    if len(vals) != B:
        raise ValueError(f"{name} has length {len(vals)} but the batch has "
                         f"{B} row(s)")
    return vals


class PlanInjector(nn.Module):
    def __init__(self, d_model: int, n_pitch_bins: int, n_duration_bins: int,
                 n_energy_bins: int, d_plan: int = 256,
                 scaffold_dropout: float = 0.3, device=None, dtype=None):
        super().__init__()
        kw = {"device": device, "dtype": dtype}
        if not 0.0 <= scaffold_dropout <= 1.0:
            raise ValueError(f"scaffold_dropout must be a probability in "
                             f"[0, 1], got {scaffold_dropout}")
        self.bin_sizes = (int(n_pitch_bins), int(n_duration_bins),
                          int(n_energy_bins))
        self.d_plan = int(d_plan)
        self.scaffold_dropout = float(scaffold_dropout)
        self.embeds = nn.ModuleList([
            nn.Embedding(n + 1, d_plan, **kw) for n in self.bin_sizes
        ])
        for e in self.embeds:
            nn.init.normal_(e.weight, std=0.02)
            with torch.no_grad():
                e.weight[0].zero_()
        self.proj = nn.Linear(d_plan, d_model, **kw)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def word_vectors(self, plan_bins: torch.Tensor) -> torch.Tensor:
        if plan_bins.dim() != 3 or plan_bins.shape[-1] != len(PLAN_BIN_CHANNELS):
            raise ValueError(
                f"plan_bins must be [B, W, {len(PLAN_BIN_CHANNELS)}] with column "
                f"order {PLAN_BIN_CHANNELS}, got shape {tuple(plan_bins.shape)}")
        vec = None
        for c, name in enumerate(PLAN_BIN_CHANNELS):
            n_bins = self.bin_sizes[c]
            col = plan_bins[..., c]
            hi = n_bins if name == "pitch" else n_bins - 1
            bad = (col < PLAN_BIN_ABSENT) | (col > hi)
            if bool(bad.any()):
                v = int(col[bad][0])
                raise ValueError(
                    f"plan_bins {name} bin {v} out of range: expected "
                    f"0..{n_bins - 1} or {PLAN_BIN_ABSENT} (pad/absent)"
                    + (f" or {n_bins} (unvoiced sentinel)" if name == "pitch"
                       else ""))
            idx = torch.where((col >= 0) & (col < n_bins), col + 1,
                              torch.zeros_like(col))
            e = self.embeds[c](idx)
            vec = e if vec is None else vec + e
        return vec

    def forward(self, plan_bins: torch.Tensor,
                schedule: torch.Tensor) -> torch.Tensor:
        if schedule.dim() != 2:
            raise ValueError(f"schedule must be [B, S], got "
                             f"{tuple(schedule.shape)}")
        if plan_bins.shape[0] != schedule.shape[0]:
            raise ValueError(f"batch mismatch: plan_bins B={plan_bins.shape[0]} "
                             f"vs schedule B={schedule.shape[0]}")
        W = plan_bins.shape[1]
        if W == 0:
            raise ValueError(
                "plan_bins has zero words. A batch in which no row carries a "
                "plan must OMIT plan_bins entirely (as the collator omits "
                "prosody_vecs), so the injection channel is off rather than "
                "silently injecting nothing.")
        top = int(schedule.max()) if schedule.numel() else -1
        if top >= W:
            raise ValueError(
                f"schedule references word {top} but plan_bins holds {W} word "
                f"slot(s) — schedule and plan came from different rows.")

        vec = self.word_vectors(plan_bins)
        valid = schedule >= 0
        idx = schedule.clamp(min=0).unsqueeze(-1).expand(-1, -1, self.d_plan)
        gathered = vec.gather(1, idx)
        out = self.proj(gathered)
        out = out * valid.unsqueeze(-1).to(out.dtype)
        if self.training and self.scaffold_dropout > 0.0:
            keep = (torch.rand(out.shape[0], 1, 1, device=out.device)
                    >= self.scaffold_dropout).to(out.dtype)
            out = out * keep
        return out


class DepthModule(nn.Module):
    def __init__(self, d_model: int, num_levels: int = 8,
                 codebook_size: int = 2048, d_depth: int = 1024,
                 num_layers: int = 2, cross_frame: int = 0,
                 codebook_sizes=None, device=None, dtype=None,
                 depth_arch: str = "gru", depth_heads: int = 8,
                 level_decay: float = 1.0, depth_cond: str = "add"):
        super().__init__()
        kw = {"device": device, "dtype": dtype}
        self.depth_arch = depth_arch
        self.level_decay = float(level_decay)
        self.num_levels = num_levels
        self.codebook_size = codebook_size
        self.level_sizes = (list(codebook_sizes) if codebook_sizes
                            else [codebook_size] * num_levels)
        assert len(self.level_sizes) == num_levels
        self.d_depth = d_depth
        self.cross_frame = int(cross_frame)
        self.h_proj = nn.Linear(d_model, d_depth, bias=False, **kw)
        self.depth_cond = depth_cond
        if depth_cond == "prefix":
            self.shared_embed = nn.Embedding(max(self.level_sizes), d_depth, **kw)
            self.level_embed = nn.Parameter(torch.zeros(num_levels, d_depth, **kw))
            nn.init.normal_(self.shared_embed.weight, std=0.02)
            nn.init.normal_(self.level_embed, std=0.02)
        else:
            self.code_embeds = nn.ModuleList([
                nn.Embedding(self.level_sizes[k], d_depth, **kw)
                for k in range(num_levels - 1)
            ])
        if self.cross_frame > 0:
            self.cross_embeds = nn.ModuleList([
                nn.Embedding(self.level_sizes[1 + c], d_depth, **kw)
                for c in range(self.cross_frame)
            ])
            self.cross_start = nn.Parameter(torch.zeros(d_depth, **kw))
            for e in self.cross_embeds:
                nn.init.normal_(e.weight, std=0.02)
        if depth_cond != "prefix":
            self.start = nn.Parameter(torch.zeros(d_depth, **kw))
        if depth_arch == "mamba2":
            from mamba_ssm.modules.mamba2 import Mamba2
            self.depth_ssm = nn.ModuleList([
                Mamba2(d_model=d_depth, **kw) for _ in range(num_layers)
            ])
            self.depth_norms = nn.ModuleList([
                RMSNorm(d_depth, **kw) for _ in range(num_layers)
            ])
        elif depth_arch == "transformer":
            self.depth_pos = nn.Parameter(torch.zeros(num_levels, d_depth, **kw))
            nn.init.normal_(self.depth_pos, std=0.02)
            layer = nn.TransformerEncoderLayer(
                d_model=d_depth, nhead=depth_heads,
                dim_feedforward=4 * d_depth, dropout=0.0,
                activation="gelu", batch_first=True, norm_first=True, **kw)
            self.depth_tf = nn.TransformerEncoder(layer, num_layers=num_layers)
        else:
            self.gru = nn.GRU(d_depth, d_depth, num_layers=num_layers,
                              batch_first=True, **kw)
        self.norm = RMSNorm(d_depth, **kw)
        self.heads = nn.ModuleList([
            nn.Linear(d_depth, self.level_sizes[k], bias=False, **kw)
            for k in range(num_levels)
        ])
        for h in self.heads:
            nn.init.normal_(h.weight, std=0.02)
        if depth_cond != "prefix":
            for e in self.code_embeds:
                nn.init.normal_(e.weight, std=0.02)

    def _cross_summary(self, prev_codes):
        if self.cross_frame <= 0:
            return None
        if prev_codes is None:
            return self.cross_start.unsqueeze(0)
        lv = prev_codes[:, 1:1 + self.cross_frame]
        valid = (lv >= 0).all(dim=-1, keepdim=True)
        summ = torch.zeros(prev_codes.shape[0], self.d_depth,
                           device=prev_codes.device, dtype=self.cross_start.dtype)
        for c in range(self.cross_frame):
            summ = summ + self.cross_embeds[c](lv[:, c].clamp(min=0))
        return torch.where(valid, summ, self.cross_start.unsqueeze(0))

    def _inputs_teacher_forced(self, h, targets, prev_codes=None):
        hp = self.h_proj(h)
        cs = self._cross_summary(prev_codes)
        if cs is not None:
            hp = hp + cs
        if self.depth_cond == "prefix":
            steps = [hp]
            for k in range(self.num_levels - 1):
                steps.append(self.shared_embed(targets[:, k].clamp(min=0)))
            x = torch.stack(steps, dim=1)
            return x + self.level_embed[:self.num_levels].unsqueeze(0)
        steps = [self.start.unsqueeze(0).expand(h.shape[0], -1)]
        for k in range(self.num_levels - 1):
            steps.append(self.code_embeds[k](targets[:, k].clamp(min=0)))
        x = torch.stack(steps, dim=1)
        return x + hp.unsqueeze(1)

    def _depth_run(self, x):
        if self.depth_arch == "mamba2":
            for norm, blk in zip(self.depth_norms, self.depth_ssm):
                x = x + blk(norm(x))
            return x
        if self.depth_arch == "transformer":
            K = x.shape[1]
            x = x + self.depth_pos[:K].unsqueeze(0)
            mask = torch.triu(torch.full((K, K), float("-inf"),
                                         device=x.device, dtype=x.dtype),
                              diagonal=1)
            return self.depth_tf(x, mask=mask)
        y, _ = self.gru(x)
        return y

    def forward_loss(self, h, targets, semantic_weight: float = 1.0,
                     prev_codes=None, frame_weights=None):
        x = self._inputs_teacher_forced(h, targets, prev_codes)
        y = self._depth_run(x)
        y = self.norm(y)
        d = float(getattr(self, "level_decay", 1.0) or 1.0)
        ws = [semantic_weight if k == 0 else d ** (k - 1)
              for k in range(self.num_levels)]
        if d != 1.0:
            scale = self.num_levels / max(sum(ws), 1e-8)
            ws = [w * scale for w in ws]
        loss = torch.tensor(0.0, device=h.device, dtype=torch.float32)
        for k in range(self.num_levels):
            logits = self.heads[k](y[:, k]).float()
            if k == 0 and frame_weights is not None:
                per = F.cross_entropy(logits, targets[:, 0], reduction="none")
                w = frame_weights.to(per.dtype)
                ce = (per * w).sum() / w.sum().clamp(min=1e-8)
            else:
                ce = F.cross_entropy(logits, targets[:, k])
            loss = loss + ws[k] * ce
        return loss

    @torch.no_grad()
    def generate_frame(self, h, level_temps=None, top_k=0, top_p=0.0,
                       sticky_prev=None, sticky_bias=0.0, prev_codes=None,
                       antifan_prev2=None, antifan_bias=0.0,
                       antifan_levels=None, rep_window=None,
                       rep_penalty=1.0, ras_tau=0.0, ras_win=10,
                       return_argmax=False, cb0_mask=None):
        _af_levels = (set(antifan_levels) if antifan_levels is not None
                      else set(range(1, min(4, self.num_levels))))
        amax = [] if return_argmax else None
        hp = self.h_proj(h)
        if self.cross_frame > 0:
            pc = None
            if prev_codes is not None:
                pc = torch.as_tensor(prev_codes, device=h.device,
                                     dtype=torch.long).view(1, -1)
            hp = hp + self._cross_summary(pc)
        state = None
        if self.depth_cond == "prefix":
            steps = [(hp + self.level_embed[0]).unsqueeze(1)]
        else:
            steps = [(self.start.unsqueeze(0) + hp).unsqueeze(1)]
        out = []
        for k in range(self.num_levels):
            if self.depth_arch in ("transformer", "mamba2"):
                yk = self._depth_run(torch.cat(steps, dim=1))[:, -1]
            else:
                y, state = self.gru(steps[-1], state)
                yk = y[:, -1]
            logits = self.heads[k](self.norm(yk)).float()[0]
            if (sticky_bias and sticky_prev is not None and k >= 1
                    and 0 <= sticky_prev[k] < logits.shape[-1]):
                logits[sticky_prev[k]] += sticky_bias
            if (antifan_bias and antifan_prev2 is not None
                    and sticky_prev is not None
                    and k in _af_levels
                    and 0 <= antifan_prev2[k] < logits.shape[-1]
                    and antifan_prev2[k] != sticky_prev[k]):
                logits[antifan_prev2[k]] -= antifan_bias
            if k == 0 and cb0_mask is not None:
                logits[cb0_mask] = float("-inf")
            base_logits = logits.clone() if ras_tau else None
            rp = rep_penalty
            if rp is not None and not isinstance(rp, (int, float)):
                rp = rp[k] if k < len(rp) else 1.0
            if rp and rp != 1.0 and rep_window:
                hist = rep_window[k] if k < len(rep_window) else ()
                if hist:
                    idx = torch.as_tensor(sorted({int(c) for c in hist
                                                  if 0 <= c < logits.shape[-1]}),
                                          device=logits.device, dtype=torch.long)
                    if idx.numel():
                        s = logits[idx]
                        logits[idx] = torch.where(s < 0, s * rp, s / rp)
            t = level_temps[k] if level_temps else 0.0
            if t and t > 0:
                lk = logits / t
                if top_k and top_k > 0:
                    kth = torch.topk(lk, min(top_k, lk.shape[-1]))[0][..., -1]
                    lk = lk.masked_fill(lk < kth, float("-inf"))
                if top_p and 0.0 < top_p < 1.0:
                    sl, si = torch.sort(lk, descending=True)
                    cum = torch.cumsum(F.softmax(sl, dim=-1), dim=-1)
                    cut = cum - F.softmax(sl, dim=-1) > top_p
                    sl = sl.masked_fill(cut, float("-inf"))
                    lk = torch.full_like(lk, float("-inf")).scatter(-1, si, sl)
                ck = int(torch.multinomial(F.softmax(lk, dim=-1), 1).item())
                if ras_tau and rep_window:
                    rt = ras_tau
                    if not isinstance(rt, (int, float)):
                        rt = rt[k] if k < len(rt) else 0.0
                    if rt and rt > 0:
                        hist = rep_window[k] if k < len(rep_window) else ()
                        win = list(hist)[-ras_win:] if ras_win else list(hist)
                        if win and sum(1 for c in win if c == ck) >= max(
                                1, int(round(len(win) * rt))):
                            ck = int(torch.multinomial(
                                F.softmax(base_logits / t, dim=-1), 1).item())
            else:
                ck = int(logits.argmax(-1).item())
            out.append(ck)
            if amax is not None:
                amax.append(int(logits.argmax(-1).item()))
            if k < self.num_levels - 1:
                idx = torch.tensor([ck], device=h.device)
                if self.depth_cond == "prefix":
                    nxt = self.shared_embed(idx) + self.level_embed[k + 1]
                else:
                    nxt = self.code_embeds[k](idx) + hp
                steps.append(nxt.unsqueeze(1))
        return (out, amax) if return_argmax else out


class AttentionMixer(nn.Module):
    def __init__(self, d_model: int, head_dim: int = 64,
                 rope_base: float = 10000.0, device=None, dtype=None,
                 layer_idx=None):
        super().__init__()
        assert d_model % head_dim == 0
        kw = {"device": device, "dtype": dtype}
        self.num_heads = d_model // head_dim
        self.head_dim = head_dim
        self.layer_idx = layer_idx
        self.in_proj = nn.Linear(d_model, 3 * d_model, bias=False, **kw)
        self.out_proj = nn.Linear(d_model, d_model, bias=False, **kw)
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        inv_freq = 1.0 / (rope_base ** (
            torch.arange(0, head_dim, 2, device=device).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _rope(self, x, positions):
        freqs = torch.outer(positions.float(), self.inv_freq)
        cos = freqs.cos()[None, None, :, :].to(x.dtype)
        sin = freqs.sin()[None, None, :, :].to(x.dtype)
        x1, x2 = x[..., 0::2], x[..., 1::2]
        out = torch.empty_like(x)
        out[..., 0::2] = x1 * cos - x2 * sin
        out[..., 1::2] = x1 * sin + x2 * cos
        return out

    def forward(self, hidden_states, inference_params=None, **kwargs):
        B, S, D = hidden_states.shape
        qkv = self.in_proj(hidden_states)
        q, k, v = qkv.chunk(3, dim=-1)
        shape = (B, S, self.num_heads, self.head_dim)
        q = q.view(shape).transpose(1, 2)
        k = k.view(shape).transpose(1, 2)
        v = v.view(shape).transpose(1, 2)
        if inference_params is None:
            pos = torch.arange(S, device=hidden_states.device)
            q = self._rope(q, pos)
            k = self._rope(k, pos)
            attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            if self.layer_idx is None:
                raise RuntimeError("AttentionMixer.layer_idx is required for the KV cache "
                                   "(enable_hybrid_attention sets it)")
            off = int(inference_params.seqlen_offset)
            pos = torch.arange(off, off + S, device=hidden_states.device)
            q = self._rope(q, pos)
            k = self._rope(k, pos)
            cache = (inference_params.key_value_memory_dict.get(self.layer_idx)
                     if off > 0 else None)
            if cache is not None:
                k = torch.cat([cache[0], k], dim=2)
                v = torch.cat([cache[1], v], dim=2)
            inference_params.key_value_memory_dict[self.layer_idx] = (k, v)
            if S == 1:
                attn = F.scaled_dot_product_attention(q, k, v, is_causal=False)
            elif off == 0:
                attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            else:
                L = k.shape[2]
                mask = torch.ones(S, L, dtype=torch.bool, device=q.device).tril(diagonal=L - S)
                attn = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        attn = attn.transpose(1, 2).reshape(B, S, D)
        return self.out_proj(attn)


class ContinuousLatentHead(nn.Module):
    def __init__(self, d_model: int, latent_dim: int = 512, cb0_size: int = 2048,
                 d_hidden: int = 1024, n_layers: int = 4, device=None, dtype=None):
        super().__init__()
        kw = {"device": device, "dtype": torch.float32}
        self.latent_dim = latent_dim
        self.d_hidden = d_hidden
        self.h_proj = nn.Linear(d_model, d_hidden, **kw)
        self.cb0_embed = nn.Embedding(cb0_size, d_hidden, **kw)
        self.x_in = nn.Linear(latent_dim, d_hidden, **kw)
        self.t_embed = nn.Sequential(
            nn.Linear(1, d_hidden, **kw), nn.SiLU(),
            nn.Linear(d_hidden, d_hidden, **kw))
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Linear(d_hidden, d_hidden, **kw), nn.SiLU(),
                          nn.Linear(d_hidden, d_hidden, **kw))
            for _ in range(n_layers)])
        self.norm = RMSNorm(d_hidden, device=device, dtype=torch.float32)
        self.out = nn.Linear(d_hidden, latent_dim, **kw)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def _cond(self, h, cb0):
        return self.h_proj(h.float()) + self.cb0_embed(cb0.clamp(min=0))

    def _velocity(self, x_t, t, cond):
        z = self.x_in(x_t) + self.t_embed(t) + cond
        for blk in self.blocks:
            z = z + blk(z)
        return self.out(self.norm(z))

    def flow_loss(self, h, cb0, target):
        with torch.autocast(device_type="cuda", enabled=False):
            target = target.float()
            cond = self._cond(h, cb0)
            t = torch.rand(target.shape[0], 1, device=target.device)
            noise = torch.randn_like(target)
            x_t = (1.0 - t) * noise + t * target
            v = self._velocity(x_t, t, cond)
            return F.mse_loss(v, target - noise)

    @torch.no_grad()
    def sample(self, h, cb0, steps: int = 16, noise_scale: float = 1.0):
        with torch.autocast(device_type="cuda", enabled=False):
            cond = self._cond(h, cb0)
            x = torch.randn(h.shape[0], self.latent_dim, device=h.device) * noise_scale
            dt = 1.0 / steps
            for i in range(steps):
                t = torch.full((h.shape[0], 1), i * dt, device=h.device)
                x = x + dt * self._velocity(x, t, cond)
            return x


class MultiTokenPrediction(nn.Module):
    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        num_crh_heads: int = 0,
        crh_codebook_size: int = 2048,
        num_spec_heads: int = 0,
        shared_spec_weights: bool = True,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.num_crh_heads = num_crh_heads
        self.num_spec_heads = num_spec_heads
        self.shared_spec_weights = shared_spec_weights
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.crh_codebook_size = crh_codebook_size

        self.num_heads = num_crh_heads + num_spec_heads

        if num_crh_heads > 0:
            self.crh_norms = nn.ModuleList([
                RMSNorm(d_model, device=device, dtype=dtype)
                for _ in range(num_crh_heads)
            ])
            self.crh_heads = nn.ModuleList([
                nn.Linear(d_model, crh_codebook_size, bias=False, device=device, dtype=dtype)
                for _ in range(num_crh_heads)
            ])

        if num_spec_heads > 0:
            if shared_spec_weights:
                self.spec_shared_norm = RMSNorm(d_model, device=device, dtype=dtype)
                self.spec_shared_head = nn.Linear(
                    d_model, vocab_size, bias=False, device=device, dtype=dtype
                )
            else:
                self.spec_norms = nn.ModuleList([
                    RMSNorm(d_model, device=device, dtype=dtype)
                    for _ in range(num_spec_heads)
                ])
                self.spec_heads = nn.ModuleList([
                    nn.Linear(d_model, vocab_size, bias=False, device=device, dtype=dtype)
                    for _ in range(num_spec_heads)
                ])

        self._init_weights()

    def _init_weights(self):
        if self.num_crh_heads > 0:
            for head in self.crh_heads:
                nn.init.normal_(head.weight, std=0.02)
        if self.num_spec_heads > 0:
            if self.shared_spec_weights:
                nn.init.normal_(self.spec_shared_head.weight, std=0.02)
            else:
                for head in self.spec_heads:
                    nn.init.normal_(head.weight, std=0.02)

    def compute_loss(
        self,
        hidden_states: torch.Tensor,
        labels: torch.Tensor,
        speech_mask: torch.BoolTensor,
        residual_codes: Optional[torch.Tensor] = None,
        crh_gamma: float = 0.85,
        spec_gamma: float = 0.8,
    ) -> torch.Tensor:
        total_loss = torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype)
        total_loss = total_loss + self._compute_crh_loss(hidden_states, speech_mask, residual_codes, crh_gamma)
        total_loss = total_loss + self._compute_spec_loss(hidden_states, labels, speech_mask, spec_gamma)
        return total_loss

    def _compute_crh_loss(self, hidden_states, speech_mask, residual_codes, gamma) -> torch.Tensor:
        if self.num_crh_heads == 0 or residual_codes is None:
            return torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype)

        B, T, D = hidden_states.shape
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction="mean")
        total = torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype)

        for k in range(self.num_crh_heads):
            target_k = residual_codes[:, k, :]
            masked_target = target_k.clone()
            masked_target[~speech_mask] = -100

            if (masked_target != -100).sum() == 0:
                continue

            h_normed = self.crh_norms[k](hidden_states)
            logits_k = self.crh_heads[k](h_normed)

            loss_k = loss_fct(
                logits_k.reshape(-1, self.crh_codebook_size),
                masked_target.reshape(-1),
            )

            discount = gamma ** (k + 1)
            total = total + discount * loss_k

        return total

    def _compute_spec_loss(self, hidden_states, labels, speech_mask, gamma) -> torch.Tensor:
        if self.num_spec_heads == 0:
            return torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype)

        B, T, D = hidden_states.shape
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction="mean")
        total = torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype)

        for m_idx in range(self.num_spec_heads):
            offset = m_idx + 2

            if offset >= T:
                continue

            h = hidden_states[:, :T - offset, :]
            target = labels[:, offset:]
            target_speech_mask = speech_mask[:, offset:]
            valid_mask = target_speech_mask & (target != -100)

            if not valid_mask.any():
                continue

            if self.shared_spec_weights:
                h_normed = self.spec_shared_norm(h)
                logits_m = self.spec_shared_head(h_normed)
            else:
                h_normed = self.spec_norms[m_idx](h)
                logits_m = self.spec_heads[m_idx](h_normed)

            masked_target = target.clone()
            masked_target[~valid_mask] = -100

            loss_m = loss_fct(
                logits_m.reshape(-1, self.vocab_size),
                masked_target.reshape(-1),
            )

            discount = gamma ** (m_idx + 1)
            total = total + discount * loss_m

        return total

    def predict_offset_2(self, hidden_state: torch.Tensor) -> torch.Tensor:
        assert self.shared_spec_weights, (
            "predict_offset_2 requires shared_spec_weights=True"
        )
        assert self.num_spec_heads > 0, "No speculative heads configured"

        if hidden_state.dim() == 1:
            hidden_state = hidden_state.unsqueeze(0)
        h_normed = self.spec_shared_norm(hidden_state)
        logits = self.spec_shared_head(h_normed)
        return logits.argmax(dim=-1)

    def speculative_draft(
        self,
        hidden_state: torch.Tensor,
        num_draft_tokens: int = 1,
    ) -> torch.Tensor:
        if num_draft_tokens > 1:
            import warnings
            warnings.warn(
                "speculative_draft(num_draft_tokens > 1) is unsupported with "
                "shared_spec_weights — returning the same offset-2 draft "
                f"replicated {num_draft_tokens} times. Use predict_offset_2 "
                "for the single-draft architecture. Multi-draft requires "
                "Medusa or EAGLE-style retraining.",
                DeprecationWarning,
                stacklevel=2,
            )
        single = self.predict_offset_2(hidden_state)
        return single.unsqueeze(-1).expand(single.size(0), num_draft_tokens)


class MambaCoTModel(nn.Module):
    def __init__(
        self,
        model_name: str = "state-spaces/mamba2-1.3b",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        mtp_num_heads: int = 3,
        mtp_gamma: float = 0.8,
        shared_spec_weights: bool = True,
        num_crh_heads: int = 0,
        crh_codebook_size: int = 2048,
        crh_gamma: float = 0.85,
        backbone_kind: str = "mamba",
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.mtp_gamma = mtp_gamma
        self.crh_gamma = crh_gamma

        from backbones import load_backbone
        self.backbone_kind = backbone_kind
        print(f"Loading {backbone_kind} backbone: {model_name}...")
        self.backbone = load_backbone(backbone_kind, model_name,
                                      device=device, dtype=dtype)

        self.tokenizer = load_base_tokenizer()
        num_added = expand_tokenizer(self.tokenizer)
        print(f"Added {num_added} new tokens to tokenizer (vocab: {len(self.tokenizer)}).")

        self._resize_embeddings()

        self.token_registry = TokenRegistry(self.tokenizer)

        d_model = self.backbone.config.d_model
        vocab_size = self.backbone.config.vocab_size
        self.mtp_module = MultiTokenPrediction(
            d_model=d_model,
            vocab_size=vocab_size,
            num_crh_heads=num_crh_heads,
            crh_codebook_size=crh_codebook_size,
            num_spec_heads=mtp_num_heads,
            shared_spec_weights=shared_spec_weights,
            device=device,
            dtype=dtype,
        )

        self.use_speaker_conditioning = False
        self.use_speaker_input = False
        self.use_prosody_input = False
        self.use_plan_injection = False
        self.use_delay_pattern = False
        self.use_depth_module = False
        self.use_flow_head = False
        self.use_hybrid_attention = False
        self.hybrid_replaced_layers = []

    def enable_delay_pattern(self, num_levels: int = 8, codebook_size: int = 2048,
                             codebook_sizes=None):
        sizes = (list(codebook_sizes) if codebook_sizes
                 else [codebook_size] * num_levels)
        emb = self.backbone.backbone.embedding
        if not isinstance(emb, DelayEmbedding):
            self.backbone.backbone.embedding = DelayEmbedding(
                emb, num_levels=num_levels, codebook_size=codebook_size,
                codebook_sizes=codebook_sizes,
            )
        ref = self.backbone.lm_head.weight
        d_model = ref.shape[1]
        self.delay_norms = nn.ModuleList([
            RMSNorm(d_model, device=ref.device, dtype=ref.dtype)
            for _ in range(num_levels)
        ])
        self.delay_heads = nn.ModuleList([
            nn.Linear(d_model, sizes[k], bias=False,
                      device=ref.device, dtype=ref.dtype)
            for k in range(num_levels)
        ])
        for h in self.delay_heads:
            nn.init.normal_(h.weight, std=0.02)
        self.delay_num_levels = num_levels
        self.delay_codebook_size = codebook_size
        self.use_delay_pattern = True

    def forward_delay(
        self,
        input_ids: torch.Tensor,
        codes: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        speaker_ids: Optional[torch.Tensor] = None,
        prosody_vecs: Optional[torch.Tensor] = None,
        semantic_weight: float = 1.0,
        eos_weight: float = 1.0,
        plan_bins: Optional[torch.Tensor] = None,
        plan_schedule: Optional[torch.Tensor] = None,
    ) -> dict:
        speaker_emb = None
        inject = None
        if self.use_speaker_conditioning and speaker_ids is not None:
            speaker_emb = self.speaker_encoder(speaker_ids)
            if self.use_speaker_input:
                inject = self.speaker_input_proj(speaker_emb)
        if self.use_prosody_input and prosody_vecs is not None:
            ref = next(self.backbone.parameters())
            p = self.prosody_input_proj(
                self.prosody_encoder(prosody_vecs.to(device=ref.device,
                                                     dtype=ref.dtype)))
            inject = p if inject is None else inject + p
        if inject is not None:
            self.backbone.backbone.embedding.set_speaker(inject)
        self._stage_plan(plan_bins, plan_schedule)
        self.backbone.backbone.embedding.set_codes(codes)
        hidden_states = self.backbone.backbone(input_ids)
        if speaker_emb is not None:
            hidden_states = self.adaln(hidden_states, speaker_emb)
        logits = self.backbone.lm_head(hidden_states)

        text_loss = None
        audio_loss = None
        total_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            if eos_weight != 1.0:
                per_tok = nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1), ignore_index=-100, reduction="none")
                flat = shift_labels.view(-1)
                tw = torch.ones_like(per_tok)
                reg = self.token_registry
                stop_ids = torch.tensor([reg.audio_end_id, reg.eos_id],
                                        device=flat.device)
                tw[torch.isin(flat, stop_ids)] = eos_weight
                valid = flat != -100
                text_loss = (per_tok * tw)[valid].sum() / tw[valid].sum()
            else:
                text_loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
            audio_loss = torch.tensor(
                0.0, device=hidden_states.device, dtype=torch.float32)
            h = hidden_states[:, :-1, :]
            tgt = codes[:, 1:, :]
            for k in range(self.delay_num_levels):
                tk = tgt[..., k].masked_fill(tgt[..., k] < 0, -100)
                if (tk >= 0).sum() == 0:
                    continue
                lk = self.delay_heads[k](self.delay_norms[k](h)).float()
                w = semantic_weight if k == 0 else 1.0
                audio_loss = audio_loss + w * loss_fct(
                    lk.reshape(-1, lk.shape[-1]),
                    tk.reshape(-1),
                )
            total_loss = text_loss + audio_loss

        return {
            "loss": total_loss,
            "main_loss": audio_loss,
            "mtp_loss": text_loss,
            "logits": logits,
            "hidden_states": hidden_states,
        }

    def enable_depth_module(self, num_levels: int = 8, codebook_size: int = 2048,
                            d_depth: int = 1024, depth_layers: int = 2,
                            depth_feedback: str = "all", cross_frame: int = 0,
                            codebook_sizes=None, depth_arch: str = "gru",
                            depth_heads: int = 8, level_decay: float = 1.0,
                            depth_cond: str = "add"):
        emb = self.backbone.backbone.embedding
        if not isinstance(emb, DelayEmbedding):
            self.backbone.backbone.embedding = DelayEmbedding(
                emb, num_levels=num_levels, codebook_size=codebook_size,
                codebook_sizes=codebook_sizes,
            )
        ref = self.backbone.lm_head.weight
        self.depth_module = DepthModule(
            d_model=ref.shape[1], num_levels=num_levels,
            codebook_size=codebook_size, d_depth=d_depth,
            num_layers=depth_layers, cross_frame=cross_frame,
            codebook_sizes=codebook_sizes,
            depth_arch=depth_arch, depth_heads=depth_heads,
            level_decay=level_decay, depth_cond=depth_cond,
            device=ref.device, dtype=ref.dtype)
        self.depth_num_levels = num_levels
        self.depth_codebook_size = codebook_size
        self.depth_level_sizes = self.depth_module.level_sizes
        self.depth_feedback = depth_feedback
        self.depth_cross_frame = cross_frame
        self.use_depth_module = True

    def enable_flow_head(self, latent_dim: int = 512, cb0_size: int = 2048,
                         d_hidden: int = 1024, n_layers: int = 4,
                         flow_weight: float = 1.0):
        ref = self.backbone.lm_head.weight
        self.flow_head = ContinuousLatentHead(
            d_model=ref.shape[1], latent_dim=latent_dim, cb0_size=cb0_size,
            d_hidden=d_hidden, n_layers=n_layers, device=ref.device)
        self.flow_weight = flow_weight
        self.use_flow_head = True

    def enable_hybrid_attention(self, top_k: int = 4, head_dim: int = 64,
                                rope_base: float = 10000.0):
        layers = self.backbone.backbone.layers
        n = len(layers)
        assert 0 < top_k < n, f"top_k={top_k} out of range for {n} layers"
        ref = self.backbone.lm_head.weight
        replaced = list(range(n - top_k, n))
        for i in replaced:
            layers[i].mixer = AttentionMixer(
                d_model=ref.shape[1], head_dim=head_dim, rope_base=rope_base,
                device=ref.device, dtype=ref.dtype, layer_idx=i)
        self.hybrid_replaced_layers = replaced
        self.use_hybrid_attention = True
        return replaced

    def forward_depth(
        self,
        input_ids: torch.Tensor,
        codes: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        speaker_ids: Optional[torch.Tensor] = None,
        prosody_vecs: Optional[torch.Tensor] = None,
        semantic_weight: float = 1.0,
        eos_weight: float = 1.0,
        target_latents: Optional[torch.Tensor] = None,
        loss_weights: Optional[torch.Tensor] = None,
        pause_exit_weight: float = 1.0,
        pause_codes: Optional[torch.Tensor] = None,
        plan_bins: Optional[torch.Tensor] = None,
        plan_schedule: Optional[torch.Tensor] = None,
    ) -> dict:
        speaker_emb = None
        inject = None
        if self.use_speaker_conditioning and speaker_ids is not None:
            speaker_emb = self.speaker_encoder(speaker_ids)
            if self.use_speaker_input:
                inject = self.speaker_input_proj(speaker_emb)
        if self.use_prosody_input and prosody_vecs is not None:
            ref = next(self.backbone.parameters())
            p = self.prosody_input_proj(
                self.prosody_encoder(prosody_vecs.to(device=ref.device,
                                                     dtype=ref.dtype)))
            inject = p if inject is None else inject + p
        if inject is not None:
            self.backbone.backbone.embedding.set_speaker(inject)
        self._stage_plan(plan_bins, plan_schedule)
        feed = codes
        if self.depth_feedback == "semantic":
            feed = codes.clone()
            feed[..., 1:] = -1
        self.backbone.backbone.embedding.set_codes(feed)
        hidden_states = self.backbone.backbone(input_ids)
        if speaker_emb is not None:
            hidden_states = self.adaln(hidden_states, speaker_emb)
        logits = self.backbone.lm_head(hidden_states)

        text_loss = None
        depth_loss = None
        total_loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            if eos_weight != 1.0 or loss_weights is not None:
                per_tok = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1), ignore_index=-100, reduction="none")
                flat = shift_labels.view(-1)
                tw = torch.ones_like(per_tok)
                if eos_weight != 1.0:
                    reg = self.token_registry
                    stop_ids = torch.tensor([reg.audio_end_id, reg.eos_id],
                                            device=flat.device)
                    tw[torch.isin(flat, stop_ids)] = eos_weight
                if loss_weights is not None:
                    tw = tw * loss_weights[..., 1:].contiguous().view(-1).to(tw.dtype)
                valid_t = flat != -100
                text_loss = (per_tok * tw)[valid_t].sum() / tw[valid_t].sum().clamp(min=1e-8)
            else:
                text_loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1), ignore_index=-100)

            h = hidden_states[:, :-1, :]
            tgt = codes[:, 1:, :]
            prev = codes[:, :-1, :]
            valid = (tgt >= 0).all(dim=-1)
            flow_loss = None
            if valid.any():
                h_v = h[valid]
                t_v = tgt[valid]
                p_v = prev[valid] if getattr(
                    self.depth_module, "cross_frame", 0) > 0 else None
                fw = None
                if pause_exit_weight != 1.0 and pause_codes is not None:
                    pc = pause_codes.to(t_v.device)
                    prev0 = prev[valid][:, 0]
                    exit_ = torch.isin(prev0, pc) & ~torch.isin(t_v[:, 0], pc)
                    fw = torch.where(exit_, torch.full_like(prev0, 0, dtype=torch.float32)
                                     + float(pause_exit_weight), torch.ones_like(prev0, dtype=torch.float32))
                depth_loss = self.depth_module.forward_loss(
                    h_v, t_v, semantic_weight=semantic_weight, prev_codes=p_v,
                    frame_weights=fw)
                if self.use_flow_head and target_latents is not None:
                    tl = target_latents[:, :, 1:].permute(0, 2, 1)
                    flow_loss = self.flow_head.flow_loss(
                        h_v, t_v[:, 0], tl[valid])
            else:
                depth_loss = torch.tensor(
                    0.0, device=hidden_states.device, dtype=torch.float32)
            total_loss = text_loss + depth_loss
            if flow_loss is not None:
                total_loss = total_loss + self.flow_weight * flow_loss

        return {
            "loss": total_loss,
            "main_loss": depth_loss,
            "mtp_loss": text_loss,
            "flow_loss": flow_loss,
            "logits": logits,
            "hidden_states": hidden_states,
        }

    def enable_speaker_conditioning(self, speaker_dim: int = 256, num_speakers: int = 1000,
                                    input_injection: bool = False):
        d_model = self.backbone.config.d_model
        ref = next(self.backbone.parameters())
        self.speaker_encoder = SpeakerEncoder(
            num_speakers=num_speakers, speaker_dim=speaker_dim
        ).to(device=ref.device, dtype=ref.dtype)
        self.adaln = AdaLN(d_model, speaker_dim=speaker_dim).to(
            device=ref.device, dtype=ref.dtype
        )
        self.use_speaker_conditioning = True
        self.use_speaker_input = bool(input_injection)
        if input_injection:
            self.speaker_input_proj = nn.Linear(speaker_dim, d_model).to(
                device=ref.device, dtype=ref.dtype)
            nn.init.zeros_(self.speaker_input_proj.weight)
            nn.init.zeros_(self.speaker_input_proj.bias)

    def enable_plan_injection(self, pitch_bins: Optional[int] = None,
                              duration_bins: Optional[int] = None,
                              energy_bins: Optional[int] = None,
                              d_plan: int = 256,
                              scaffold_dropout: float = 0.3):
        emb = self.backbone.backbone.embedding
        if not isinstance(emb, DelayEmbedding):
            raise RuntimeError(
                "enable_plan_injection() requires the DelayEmbedding input "
                "channel — call enable_depth_module() (or enable_delay_pattern"
                "()) first. Without it there is no per-position hook and the "
                "plan would be silently ignored.")
        from tokenizer import PITCH_BINS, DURATION_BINS, ENERGY_BINS
        ref = next(self.backbone.parameters())
        self.plan_injector = PlanInjector(
            d_model=self.backbone.config.d_model,
            n_pitch_bins=PITCH_BINS if pitch_bins is None else pitch_bins,
            n_duration_bins=DURATION_BINS if duration_bins is None else duration_bins,
            n_energy_bins=ENERGY_BINS if energy_bins is None else energy_bins,
            d_plan=d_plan, scaffold_dropout=scaffold_dropout,
            device=ref.device, dtype=ref.dtype)
        self.plan_bin_sizes = self.plan_injector.bin_sizes
        self.use_plan_injection = True

    def _stage_plan(self, plan_bins, plan_schedule):
        if plan_bins is None and plan_schedule is None:
            return
        if not self.use_plan_injection:
            _warn_plan_without_injection()
            return
        if plan_bins is None or plan_schedule is None:
            raise ValueError(
                "plan_bins and plan_schedule must be supplied together "
                f"(got plan_bins={'set' if plan_bins is not None else 'None'}, "
                f"plan_schedule="
                f"{'set' if plan_schedule is not None else 'None'}) — see "
                "model.build_plan_schedule.")
        self.backbone.backbone.embedding.set_plan(
            self.plan_injector(plan_bins, plan_schedule))

    def enable_prosody_conditioning(self, prosody_dim: int = 6,
                                    prosody_hidden: int = 256):
        d_model = self.backbone.config.d_model
        ref = next(self.backbone.parameters())
        self.prosody_encoder = nn.Sequential(
            nn.Linear(prosody_dim, prosody_hidden), nn.SiLU(),
        ).to(device=ref.device, dtype=ref.dtype)
        self.prosody_input_proj = nn.Linear(prosody_hidden, d_model).to(
            device=ref.device, dtype=ref.dtype)
        nn.init.zeros_(self.prosody_input_proj.weight)
        nn.init.zeros_(self.prosody_input_proj.bias)
        self.use_prosody_input = True

    def _resize_embeddings(self):
        current_vocab_size = self.backbone.backbone.embedding.weight.shape[0]
        new_vocab_size = len(self.tokenizer)

        pad_multiple = getattr(self.backbone.config, "pad_vocab_size_multiple", 16)
        if new_vocab_size % pad_multiple != 0:
            new_vocab_size += pad_multiple - (new_vocab_size % pad_multiple)

        if new_vocab_size == current_vocab_size:
            print("No embedding resize needed.")
            return

        print(f"Resizing embeddings: {current_vocab_size} -> {new_vocab_size}")

        d_model = self.backbone.config.d_model
        n_copy = min(current_vocab_size, new_vocab_size)

        old_emb = self.backbone.backbone.embedding
        new_emb = nn.Embedding(
            new_vocab_size, d_model, device=self.device, dtype=self.dtype
        )

        with torch.no_grad():
            new_emb.weight[:n_copy] = old_emb.weight[:n_copy]
            if new_vocab_size > current_vocab_size:
                mean_embed = old_emb.weight.mean(dim=0)
                noise = torch.randn(
                    new_vocab_size - n_copy, d_model,
                    device=self.device, dtype=self.dtype,
                ) * 0.02
                new_emb.weight[n_copy:] = mean_embed + noise

        self.backbone.backbone.embedding = new_emb

        new_head = nn.Linear(
            d_model, new_vocab_size, bias=False, device=self.device, dtype=self.dtype
        )

        with torch.no_grad():
            new_head.weight[:n_copy] = old_emb.weight[:n_copy]
            if new_vocab_size > current_vocab_size:
                new_head.weight[n_copy:] = torch.randn(
                    new_vocab_size - n_copy, d_model,
                    device=self.device, dtype=self.dtype,
                ) * 0.02

        self.backbone.lm_head = new_head

        if hasattr(self.backbone.config, "tie_embeddings"):
            self.backbone.config.tie_embeddings = False

        self.backbone.config.vocab_size = new_vocab_size

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        speech_mask: Optional[torch.BoolTensor] = None,
        loss_weights: Optional[torch.Tensor] = None,
        residual_codes: Optional[torch.Tensor] = None,
        speaker_ids: Optional[torch.Tensor] = None,
        codes: Optional[torch.Tensor] = None,
        prosody_vecs: Optional[torch.Tensor] = None,
        semantic_weight: float = 1.0,
        eos_weight: float = 1.0,
        target_latents: Optional[torch.Tensor] = None,
        plan_bins: Optional[torch.Tensor] = None,
        plan_schedule: Optional[torch.Tensor] = None,
        pause_exit_weight: float = 1.0,
        pause_codes: Optional[torch.Tensor] = None,
    ) -> dict:
        if codes is not None:
            if self.use_depth_module:
                return self.forward_depth(
                    input_ids, codes, labels=labels, speaker_ids=speaker_ids,
                    prosody_vecs=prosody_vecs,
                    semantic_weight=semantic_weight, eos_weight=eos_weight,
                    target_latents=target_latents, loss_weights=loss_weights,
                    pause_exit_weight=pause_exit_weight, pause_codes=pause_codes,
                    plan_bins=plan_bins, plan_schedule=plan_schedule,
                )
            return self.forward_delay(
                input_ids, codes, labels=labels, speaker_ids=speaker_ids,
                prosody_vecs=prosody_vecs,
                semantic_weight=semantic_weight, eos_weight=eos_weight,
                plan_bins=plan_bins, plan_schedule=plan_schedule,
            )

        if plan_bins is not None or plan_schedule is not None:
            raise RuntimeError(
                "plan injection requires the delay/depth path (`codes` in the "
                "batch + enable_depth_module()/enable_delay_pattern()); the "
                "flat forward has no per-position input channel.")

        hidden_states = self.backbone.backbone(input_ids)

        if self.use_speaker_conditioning and speaker_ids is not None:
            speaker_emb = self.speaker_encoder(speaker_ids)
            hidden_states = self.adaln(hidden_states, speaker_emb)

        logits = self.backbone.lm_head(hidden_states)

        main_loss = None
        mtp_loss = None
        total_loss = None

        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            if loss_weights is not None:
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
                per_token_loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                ).view(shift_logits.size(0), shift_logits.size(1))

                shift_weights = loss_weights[..., 1:].contiguous()
                weighted_loss = per_token_loss * shift_weights

                main_loss = weighted_loss.sum() / shift_weights.sum().clamp(min=1.0)
            else:
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                main_loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )

            if speech_mask is None:
                speech_mask = self.token_registry.build_speech_mask(labels)

            mtp_loss = self.mtp_module.compute_loss(
                hidden_states=hidden_states,
                labels=labels,
                speech_mask=speech_mask,
                residual_codes=residual_codes,
                crh_gamma=self.crh_gamma,
                spec_gamma=self.mtp_gamma,
            )

            total_loss = main_loss + mtp_loss

        return {
            "loss": total_loss,
            "main_loss": main_loss,
            "mtp_loss": mtp_loss,
            "logits": logits,
            "hidden_states": hidden_states,
        }

    def generate(self, input_ids: torch.Tensor, max_length: int = 2000, **kwargs):
        return self.backbone.generate(input_ids, max_length=max_length, **kwargs)
