# delay_dataset.py: The frame-aligned multi-codebook dataset the depth-module models train on: codebook 0 in the
# backbone stream, codebooks 1-9 per frame, the three think-block tiers, lazy JSONL loading.

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import torch
from torch.utils.data import Dataset

import json
import logging
import os
from pathlib import Path

from tokenizer import TokenRegistry

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100
EMPTY_CODE = -1
NO_WORD = -1

PLAN_COT_MODES = ("plan", "tags+plan", "full")
LEGACY_COT_MODES = ("prose", "tags", "none")
COT_MODES = LEGACY_COT_MODES + PLAN_COT_MODES

PLAN_SCHEDULE_MODES = ("stretch", "pack")

PLAN_FRAME_RATE_HZ = 21.53
NUM_LEVELS = int(os.environ.get("MVC_NUM_CODEC_LEVELS", "8"))
CODEBOOK_SIZE = 2048


def build_plan_schedule(durations: Sequence[int], n_frames: int,
                        mode: str = "stretch") -> List[int]:
    if mode not in PLAN_SCHEDULE_MODES:
        raise ValueError(
            f"plan schedule mode {mode!r} not in {PLAN_SCHEDULE_MODES}")
    n_frames = int(n_frames)
    out = [NO_WORD] * max(0, n_frames)
    n_words = len(durations)
    if n_words == 0 or n_frames <= 0:
        return out

    durs = [int(d) for d in durations]
    if any(d < 0 for d in durs):
        raise ValueError(f"negative word duration in plan: {durs}")
    total = sum(durs)
    if total <= 0:
        raise ValueError("plan durations sum to 0 frames — the keystone token "
                         "(§3.1) is degenerate; the sidecar is broken")

    if mode == "pack":
        pos = 0
        for w, d in enumerate(durs):
            if pos >= n_frames:
                break
            for t in range(pos, min(n_frames, pos + d)):
                out[t] = w
            pos += d
        return out

    prev, cum = 0, 0
    for w, d in enumerate(durs):
        cum += d
        boundary = n_frames if w == n_words - 1 else int(round(n_frames * cum / total))
        if n_frames - prev >= n_words - w:
            boundary = max(prev + 1, min(boundary, n_frames - (n_words - 1 - w)))
        else:
            boundary = max(prev, min(boundary, n_frames))
        for t in range(prev, boundary):
            out[t] = w
        prev = boundary
    return out


def _plan_container(row: dict, field: str):
    raw = row.get(field)
    if raw is None:
        return None, False
    if isinstance(raw, dict):
        if raw.get("ok") is False:
            return None, True
        words = raw.get("words", raw.get("plan"))
        if words is None:
            raise ValueError(
                f"row[{field!r}] is a dict without 'words'/'plan'; keys="
                f"{sorted(raw)[:8]}")
        raw = words
    if not isinstance(raw, (list, tuple)):
        raise ValueError(
            f"row[{field!r}] must be a list of word dicts (or a sidecar object "
            f"carrying one), got {type(raw).__name__}")
    return (list(raw) or None), False


def normalise_plan_words(raw_words: Sequence[dict], *,
                         n_pitch_bins: int, n_duration_bins: int,
                         n_energy_bins: int,
                         frame_rate: float = PLAN_FRAME_RATE_HZ,
                         where: str = "") -> List[Dict[str, int]]:
    out: List[Dict[str, int]] = []
    for i, w in enumerate(raw_words):
        tag = f"{where}word {i}"
        if not isinstance(w, dict):
            raise ValueError(f"{tag}: expected a dict, got {type(w).__name__}")

        def _req_int(key, hi):
            if key not in w or w[key] is None:
                raise ValueError(f"{tag}: missing {key!r} (keys={sorted(w)[:8]})")
            try:
                v = int(w[key])
            except (TypeError, ValueError):
                raise ValueError(f"{tag}: {key}={w[key]!r} is not an integer")
            if not 0 <= v < hi:
                raise ValueError(f"{tag}: {key}={v} out of range [0,{hi})")
            return v

        dur_bin = _req_int("dur_bin", n_duration_bins)
        energy_bin = _req_int("energy_bin", n_energy_bins)

        pb = w.get("pitch_bin", None)
        if pb is None:
            pitch_bin = None
        else:
            try:
                pb = int(pb)
            except (TypeError, ValueError):
                raise ValueError(f"{tag}: pitch_bin={w['pitch_bin']!r} is not an integer")
            if pb == n_pitch_bins:
                pitch_bin = None
            elif 0 <= pb < n_pitch_bins:
                pitch_bin = pb
            else:
                raise ValueError(
                    f"{tag}: pitch_bin={pb} out of range [0,{n_pitch_bins}] "
                    f"({n_pitch_bins} = the unvoiced sentinel)")

        if w.get("dur_frames") is not None:
            frames = int(w["dur_frames"])
        elif w.get("dur_s") is not None:
            frames = max(1, int(round(float(w["dur_s"]) * frame_rate)))
        else:
            raise ValueError(
                f"{tag}: needs 'dur_frames' (or 'dur_s'). The schedule (§3.2) "
                f"and the decode window (§3.4) both count FRAMES, so a plan "
                f"without them cannot be consumed — re-run "
                f"scripts/plan_extract.py quantise, which emits dur_frames.")
        if frames < 1:
            raise ValueError(f"{tag}: dur_frames={frames} must be >= 1")

        out.append({"pitch_bin": pitch_bin, "dur_bin": dur_bin,
                    "energy_bin": energy_bin, "dur_frames": frames})
    return out


class DelayMimiDataset(Dataset):
    LAZY_THRESHOLD_BYTES = 2 * 1024 ** 3

    def __init__(self, data_path: str, tokenizer, registry: TokenRegistry,
                 max_seq_len: int = 2048, max_frames: int = 0,
                 cfg_dropout: float = 0.0, aligned: bool = False,
                 lazy: bool = None, stage: int = 1,
                 reasoning_weight: float = 0.3, max_reasoning_tokens: int = 512,
                 default_speaker_id: int = None,
                 audio_continue_weight: float = 0.0,
                 cot_mode: str = "prose",
                 plan_field: str = "plan",
                 plan_weight: float = 1.0,
                 tag_weight: float = None,
                 plan_schedule_mode: str = "stretch",
                 plan_lookahead: int = 1,
                 plan_frame_rate: float = PLAN_FRAME_RATE_HZ,
                 min_plan_coverage: float = 0.0,
                 plan_audit_rows: int = 2000):
        self.tokenizer = tokenizer
        self.registry = registry
        self.max_seq_len = max_seq_len
        self.cfg_dropout = cfg_dropout
        assert stage in (1, 2, "auto"), f"stage must be 1, 2 or 'auto', got {stage!r}"
        self.stage = stage
        if cot_mode in PLAN_COT_MODES:
            self.plan_mode = True
        else:
            assert cot_mode in ("prose", "tags", "none"), cot_mode
            self.plan_mode = False
        self.cot_mode = cot_mode
        self.reasoning_weight = reasoning_weight
        self.max_reasoning_tokens = max_reasoning_tokens
        self.plan_weight = float(plan_weight)
        self.tag_weight = None if tag_weight is None else float(tag_weight)
        if plan_schedule_mode not in PLAN_SCHEDULE_MODES:
            raise ValueError(f"plan_schedule_mode must be one of "
                             f"{PLAN_SCHEDULE_MODES}, got {plan_schedule_mode!r}")
        self.plan_schedule_mode = plan_schedule_mode
        if plan_lookahead not in (0, 1):
            raise ValueError(f"plan_lookahead must be 0 or 1, got {plan_lookahead}")
        self.plan_lookahead = int(plan_lookahead)
        self.plan_field = plan_field
        self.plan_frame_rate = float(plan_frame_rate)
        reg_pitch = getattr(registry, "pitch_ids", []) or []
        reg_dur = getattr(registry, "duration_ids", []) or []
        reg_energy = getattr(registry, "energy_ids", []) or []
        self._n_pitch_bins = len(reg_pitch)
        self._n_duration_bins = len(reg_dur)
        self._n_energy_bins = len(reg_energy)
        if self.plan_mode and not getattr(registry, "plan_enabled", False):
            raise RuntimeError(
                f"cot_mode={cot_mode!r} needs the P²-CoT plan vocabulary, but this "
                f"tokenizer has none (MVC_ENABLE_PLAN_TOKENS is off). Export "
                f"MVC_ENABLE_PLAN_TOKENS=1 before python starts (and across the "
                f"singularity boundary as SINGULARITYENV_MVC_ENABLE_PLAN_TOKENS) "
                f"and rebuild the tokenizer. Refusing to run a plan arm with no "
                f"plan tokens — it would train as a plain CoT arm and report "
                f"nothing wrong.")
        if self.plan_mode and stage == 1:
            raise ValueError(
                f"cot_mode={cot_mode!r} with stage=1 emits no <THINK> block, so "
                f"no plan would ever reach the model. Use stage=2 (or 'auto' for "
                f"a mixed corpus).")
        self.n_rows_seen = 0
        self.n_rows_missing_plan = 0
        self.n_rows_plan_rejected = 0
        self.n_rows_plan_truncated = 0
        self.n_rows_reasoning_dropped = 0
        self.audio_continue_weight = audio_continue_weight
        self.default_speaker_id = default_speaker_id
        self.aligned = aligned
        self.data_path = data_path
        if lazy is None:
            lazy = Path(data_path).stat().st_size > self.LAZY_THRESHOLD_BYTES
        self.lazy = lazy
        self._fh = None
        self._fh_pid = None
        self.data: List[dict] = []
        self.offsets: List[int] = []
        if self.lazy:
            with open(data_path, "rb") as f:
                pos = f.tell()
                for line in f:
                    if line.strip():
                        self.offsets.append(pos)
                    pos = f.tell()
            logger.info(f"DelayMimiDataset[LAZY]: {len(self.offsets)} rows "
                        f"indexed from {data_path} (max_seq_len={max_seq_len})")
        else:
            with open(data_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    resid = d.get("residual_codes")
                    if not resid or len(resid) != NUM_LEVELS - 1:
                        continue
                    if max_frames and len(d["speech_tokens"]) > max_frames:
                        continue
                    self.data.append(d)
            logger.info(f"DelayMimiDataset: {len(self.data)} rows from {data_path} "
                        f"(max_seq_len={max_seq_len})")

        self.plan_audit: Optional[Dict[str, Any]] = None
        if self.plan_mode:
            self.plan_audit = self._audit_plan_coverage(int(plan_audit_rows),
                                                        float(min_plan_coverage))

    def _audit_plan_coverage(self, n_max: int, min_coverage: float) -> Dict[str, Any]:
        n_rows = len(self)
        limit = n_rows if n_max <= 0 else min(n_rows, n_max)
        scanned = with_plan = rejected = n_words = 0
        for i in range(limit):
            d = self._row(i)
            scanned += 1
            words, was_rejected = _plan_container(d, self.plan_field)
            if was_rejected:
                rejected += 1
            elif words:
                with_plan += 1
                n_words += len(words)
        coverage = (with_plan / scanned) if scanned else 0.0
        audit = {"scanned": scanned, "rows": n_rows, "with_plan": with_plan,
                 "rejected": rejected, "coverage": coverage,
                 "sampled": limit < n_rows,
                 "mean_words": (n_words / with_plan) if with_plan else 0.0}
        msg = (f"DelayMimiDataset[PLAN] cot_mode={self.cot_mode}: "
               f"{with_plan}/{scanned} rows carry row[{self.plan_field!r}] "
               f"({100 * coverage:.1f}%), {rejected} rejected by the extractor's "
               f"own gates, mean {audit['mean_words']:.1f} words/plan"
               + (f" (SAMPLED: first {scanned} of {n_rows} rows)"
                  if audit["sampled"] else ""))
        if with_plan == 0:
            raise RuntimeError(
                msg + " — a plan arm with NO plans is always a bug. Point "
                      "--data_path at a plan-bearing corpus, or use a legacy "
                      "cot_mode.")
        if coverage < min_coverage:
            raise RuntimeError(
                msg + f" — below the required min_plan_coverage={min_coverage:.2f}.")
        (logger.warning if coverage < 0.95 else logger.info)(msg)
        return audit

    def __len__(self):
        return len(self.offsets) if self.lazy else len(self.data)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_fh"] = None
        state["_fh_pid"] = None
        return state

    def _row(self, idx: int) -> dict:
        if not self.lazy:
            return self.data[idx]
        if self._fh is None or self._fh_pid != os.getpid():
            self._fh = open(self.data_path, "rb")
            self._fh_pid = os.getpid()
        self._fh.seek(self.offsets[idx])
        d = json.loads(self._fh.readline())
        resid = d.get("residual_codes")
        if not resid or len(resid) != NUM_LEVELS - 1:
            raise ValueError(
                f"malformed row {idx} in {self.data_path} (lazy mode trusts the "
                f"assembler — run scripts/assemble_corpus.py validation)")
        return d

    def _plan_token_groups(self, words: Sequence[Dict[str, int]]) -> List[List[int]]:
        reg = self.registry
        groups: List[List[int]] = []
        for w in words:
            group = [reg.plan_word_sep_id]
            if w["pitch_bin"] is not None:
                group.append(reg.pitch_bin_to_id(w["pitch_bin"]))
            group.append(reg.duration_bin_to_id(w["dur_bin"]))
            group.append(reg.energy_bin_to_id(w["energy_bin"]))
            groups.append(group)
        return groups

    def _plan_ids(self, groups: Sequence[Sequence[int]]) -> List[int]:
        if not groups:
            return []
        ids = [self.registry.plan_start_id]
        for group in groups:
            ids.extend(group)
        ids.append(self.registry.plan_end_id)
        return ids

    def _think_block(self, prose_ids: Sequence[int], tag_ids: Sequence[int],
                     plan_groups: Sequence[Sequence[int]]) -> List[int]:
        content = list(prose_ids) + list(tag_ids) + self._plan_ids(plan_groups)
        if not content:
            return []
        return [self.registry.think_start_id] + content + [self.registry.think_end_id]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        d = self._row(idx)
        reg = self.registry

        if self.cfg_dropout > 0.0 and torch.rand(1).item() < self.cfg_dropout:
            prompt_ids = []
        else:
            prompt_ids = self.tokenizer.encode(d["prompt"], add_special_tokens=False)

        self.n_rows_seen += 1
        plan_words: List[Dict[str, int]] = []
        if self.plan_mode:
            raw_plan, plan_rejected = _plan_container(d, self.plan_field)
            if raw_plan:
                plan_words = normalise_plan_words(
                    raw_plan,
                    n_pitch_bins=self._n_pitch_bins,
                    n_duration_bins=self._n_duration_bins,
                    n_energy_bins=self._n_energy_bins,
                    frame_rate=self.plan_frame_rate,
                    where=f"{self.data_path} row {idx}: ")
            if plan_rejected:
                self.n_rows_plan_rejected += 1
            if not plan_words:
                self.n_rows_missing_plan += 1

        if self.stage == "auto":
            row_stage = 2 if str(d.get("reasoning") or "").strip() else 1
            if row_stage == 1 and plan_words:
                row_stage = 2
        else:
            row_stage = self.stage

        prose_ids: List[int] = []
        tag_ids: List[int] = []
        plan_groups: List[List[int]] = []
        if row_stage == 2:
            if self.cot_mode in ("prose", "full"):
                rtext = (d.get("reasoning") or "").strip()
                for tag in ("<THINK>", "</THINK>"):
                    rtext = rtext.replace(tag, "")
                rtext = rtext.strip()
                if rtext:
                    prose_ids = self.tokenizer.encode(
                        rtext, add_special_tokens=False)[: self.max_reasoning_tokens]

            if self.cot_mode == "tags" or self.cot_mode in ("tags+plan", "full"):
                if (not self.plan_mode) or any(
                        d.get(k) for k in ("emotion_label", "pace", "pitch")):
                    ttext = (f"EMO={d.get('emotion_label') or 'neutral'} "
                             f"PACE={d.get('pace') or 'normal'} "
                             f"PITCH={d.get('pitch') or 'normal'}")
                    for tag in ("<THINK>", "</THINK>"):
                        ttext = ttext.replace(tag, "")
                    ttext = ttext.strip()
                    if ttext:
                        tag_ids = self.tokenizer.encode(
                            (" " + ttext) if prose_ids else ttext,
                            add_special_tokens=False)[: self.max_reasoning_tokens]

            if plan_words:
                plan_groups = self._plan_token_groups(plan_words)

        think_block = self._think_block(prose_ids, tag_ids, plan_groups)

        levels = [d["speech_tokens"]] + d["residual_codes"]
        T = len(levels[0])
        stagger = 0 if self.aligned else NUM_LEVELS - 1
        span = T + stagger

        fixed = 2 + len(prompt_ids) + 1 + 2
        budget = self.max_seq_len - fixed
        if self.plan_mode and think_block and len(think_block) + span > budget:
            if prose_ids:
                prose_ids = []
                self.n_rows_reasoning_dropped += 1
                think_block = self._think_block(prose_ids, tag_ids, plan_groups)
            if len(think_block) + span > budget and tag_ids:
                tag_ids = []
                think_block = self._think_block(prose_ids, tag_ids, plan_groups)
            dropped = 0
            while plan_groups and len(think_block) + stagger + 1 > budget:
                plan_groups.pop()
                plan_words.pop()
                dropped += 1
                think_block = self._think_block(prose_ids, tag_ids, plan_groups)
            if dropped:
                self.n_rows_plan_truncated += 1

        overhead = 2 + len(prompt_ids) + len(think_block) + 1 + 2
        max_span = self.max_seq_len - overhead
        if span > max_span:
            T = max(0, max_span - stagger)
            span = T + stagger
            levels = [row[:T] for row in levels]
        if T <= 0:
            T, span = 0, 0

        frame_to_word: List[int] = []
        if plan_groups:
            frame_to_word = build_plan_schedule(
                [w["dur_frames"] for w in plan_words], T, self.plan_schedule_mode)
            last = max(frame_to_word) if frame_to_word else NO_WORD
            if last < len(plan_groups) - 1:
                del plan_groups[last + 1:]
                del plan_words[last + 1:]
                self.n_rows_plan_truncated += 1
                think_block = self._think_block(prose_ids, tag_ids, plan_groups)
        plan_ids = self._plan_ids(plan_groups)

        codes = [[EMPTY_CODE] * NUM_LEVELS for _ in range(span)]
        if self.aligned:
            for s in range(span):
                for k in range(NUM_LEVELS):
                    codes[s][k] = levels[k][s]
        else:
            for s in range(span):
                for k in range(NUM_LEVELS):
                    t = s - k
                    if 0 <= t < T:
                        codes[s][k] = levels[k][t]

        audio_placeholder = reg.audio_start_id
        input_ids = (
            [reg.bos_id, reg.user_prompt_id] + prompt_ids + think_block
            + [reg.audio_start_id]
            + [audio_placeholder] * span
            + [reg.audio_end_id, reg.eos_id]
        )
        prompt_end = 2 + len(prompt_ids)
        think_lo = prompt_end
        think_hi = prompt_end + len(think_block)
        audio_start = think_hi + 1
        audio_end = audio_start + span

        labels = list(input_ids)
        for i in range(prompt_end):
            labels[i] = IGNORE_INDEX
        if self.audio_continue_weight <= 0.0:
            for i in range(audio_start, audio_end):
                labels[i] = IGNORE_INDEX

        loss_weights = [1.0] * len(input_ids)
        for i in range(think_lo, think_hi):
            loss_weights[i] = self.reasoning_weight
        tags_lo = think_lo + 1 + len(prose_ids)
        tags_hi = tags_lo + len(tag_ids)
        plan_lo = tags_hi
        plan_hi = plan_lo + len(plan_ids)
        if self.tag_weight is not None:
            for i in range(tags_lo, tags_hi):
                loss_weights[i] = self.tag_weight
        for i in range(plan_lo, plan_hi):
            loss_weights[i] = self.plan_weight
        if self.audio_continue_weight > 0.0:
            for i in range(audio_start, audio_end):
                loss_weights[i] = self.audio_continue_weight

        full_codes = [[EMPTY_CODE] * NUM_LEVELS for _ in range(len(input_ids))]
        for s in range(span):
            full_codes[audio_start + s] = codes[s]

        plan_schedule = None
        plan_slots = None
        plan_durations = None
        if plan_groups:
            sched = [NO_WORD] * len(input_ids)
            for t, w in enumerate(frame_to_word):
                if w == NO_WORD:
                    continue
                pos = audio_start + t - self.plan_lookahead
                if pos >= audio_start - 1:
                    sched[pos] = w
            plan_schedule = sched
            plan_slots = []
            cursor = plan_lo + 1
            for group in plan_groups:
                has_pitch = len(group) == 4
                p_pos = cursor + 1 if has_pitch else NO_WORD
                d_pos = cursor + (2 if has_pitch else 1)
                e_pos = d_pos + 1
                plan_slots.append([p_pos, d_pos, e_pos])
                cursor += len(group)
            plan_durations = [w["dur_frames"] for w in plan_words]

        sid = d.get("speaker_id")
        if sid is None:
            sid = self.default_speaker_id
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "codes": torch.tensor(full_codes, dtype=torch.long),
            "loss_weights": torch.tensor(loss_weights, dtype=torch.float32),
            "speaker_id": sid,
            "prosody_vec": d.get("prosody_vec"),
            "audio_start": audio_start,
            "audio_end": audio_end,
            "plan_schedule": (None if plan_schedule is None
                              else torch.tensor(plan_schedule, dtype=torch.long)),
            "plan_slots": (None if plan_slots is None
                           else torch.tensor(plan_slots, dtype=torch.long)),
            "plan_durations": (None if plan_durations is None
                               else torch.tensor(plan_durations, dtype=torch.long)),
        }


@dataclass
class DelayCollator:
    pad_token_id: int

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [b["input_ids"] for b in batch], batch_first=True,
            padding_value=self.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(
            [b["labels"] for b in batch], batch_first=True,
            padding_value=IGNORE_INDEX)
        B, S = input_ids.shape
        codes = torch.full((B, S, NUM_LEVELS), EMPTY_CODE, dtype=torch.long)
        for i, b in enumerate(batch):
            codes[i, : b["codes"].shape[0]] = b["codes"]
        out = {
            "input_ids": input_ids,
            "labels": labels,
            "codes": codes,
            "attention_mask": (input_ids != self.pad_token_id).long(),
        }
        if any("loss_weights" in b for b in batch):
            lw = torch.ones(B, S, dtype=torch.float32)
            for i, b in enumerate(batch):
                if "loss_weights" in b:
                    lw[i, : b["loss_weights"].shape[0]] = b["loss_weights"]
            out["loss_weights"] = lw
        if any(b.get("speaker_id") is not None for b in batch):
            out["speaker_ids"] = torch.tensor(
                [b.get("speaker_id") or 0 for b in batch], dtype=torch.long)
        if any(b.get("prosody_vec") is not None for b in batch):
            out["prosody_vecs"] = torch.tensor(
                [b.get("prosody_vec") or [0.0] * 6 for b in batch],
                dtype=torch.float32)
        if any(b.get("plan_schedule") is not None for b in batch):
            sched = torch.full((B, S), NO_WORD, dtype=torch.long)
            n_words = max((int(b["plan_slots"].shape[0])
                           for b in batch if b.get("plan_slots") is not None),
                          default=0)
            slots = torch.full((B, n_words, 3), NO_WORD, dtype=torch.long)
            durations = torch.zeros((B, n_words), dtype=torch.long)
            lengths = torch.zeros(B, dtype=torch.long)
            for i, b in enumerate(batch):
                ps = b.get("plan_schedule")
                if ps is None:
                    continue
                sched[i, : ps.shape[0]] = ps
                sl = b["plan_slots"]
                slots[i, : sl.shape[0]] = sl
                du = b["plan_durations"]
                durations[i, : du.shape[0]] = du
                lengths[i] = sl.shape[0]
            out["plan_schedule"] = sched
            out["plan_slots"] = slots
            out["plan_durations"] = durations
            out["plan_lengths"] = lengths
        return out
