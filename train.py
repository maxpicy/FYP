# train.py: Training for every stage (DDP, losses, optimiser, schedule, checkpoints) and the checkpoint loader
# the inference code shares; it reads training .pt files and released .safetensors alike.

import os
import sys
import json
import math
import random
import inspect
import logging
import argparse
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

if "dill" not in sys.modules:
    import types as _types
    import importlib.machinery as _im
    _dill_shim = _types.ModuleType("dill")
    _dill_shim.extend = lambda *a, **k: None
    _dill_shim.__version__ = "0.0.0-shim"
    _dill_shim.__spec__ = _im.ModuleSpec("dill", loader=None)
    sys.modules["dill"] = _dill_shim

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, Dataset
from torch.nn.parallel import DistributedDataParallel as DDP

import config
from model import MambaCoTModel
from dataset import MambaCoTTTSDataset, MambaCoTDataCollator
from tokenizer import TokenRegistry

logger = logging.getLogger(__name__)

STAGE_DEFAULTS = {
    1: {
        "max_steps": 50_000,
        "lr": 3e-4,
        "lr_lora": 1e-4,
        "lr_min": 1e-5,
        "warmup_steps": 1_000,
        "batch_size": 4,
        "grad_accum": 4,
        "use_weighted_loss": False,
        "dataset_stage": 1,
    },
    2: {
        "max_steps": 100_000,
        "lr": 5e-5,
        "lr_lora": None,
        "lr_min": 1e-6,
        "warmup_steps": 2_000,
        "batch_size": 4,
        "grad_accum": 4,
        "use_weighted_loss": True,
        "dataset_stage": 2,
    },
    3: {
        "max_steps": 50_000,
        "lr": 1e-5,
        "lr_lora": None,
        "lr_min": 1e-6,
        "warmup_steps": 500,
        "batch_size": 4,
        "grad_accum": 4,
        "use_weighted_loss": True,
        "dataset_stage": 2,
    },
    4: {
        "max_steps": 10_000,
        "lr": 5e-6,
        "lr_lora": None,
        "lr_min": 1e-7,
        "warmup_steps": 200,
        "batch_size": 2,
        "grad_accum": 4,
        "use_weighted_loss": False,
        "dataset_stage": 2,
    },
    5: {
        "max_steps": 5_000,
        "lr": 1e-4,
        "lr_lora": None,
        "lr_min": 1e-6,
        "warmup_steps": 100,
        "batch_size": 4,
        "grad_accum": 4,
        "use_weighted_loss": False,
        "dataset_stage": 2,
    },
}


class LoRALinear(nn.Module):
    def __init__(self, original: nn.Linear, rank: int = 64,
                 alpha: int = 64, dropout: float = 0.05):
        super().__init__()
        self.original = original
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = original.in_features
        out_features = original.out_features

        original.weight.requires_grad = False
        if original.bias is not None:
            original.bias.requires_grad = False

        self.lora_A = nn.Parameter(
            torch.randn(rank, in_features, device=original.weight.device,
                        dtype=original.weight.dtype) * 0.01
        )
        self.lora_B = nn.Parameter(
            torch.zeros(out_features, rank, device=original.weight.device,
                        dtype=original.weight.dtype)
        )
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.original, name)

    def forward(self, x):
        result = self.original(x)
        lora_out = F.linear(self.lora_dropout(x), self.lora_A)
        lora_out = F.linear(lora_out, self.lora_B)
        return result + lora_out * self.scaling

    @property
    def weight(self):
        return self.original.weight + (self.lora_B @ self.lora_A) * self.scaling

    def extra_repr(self):
        return (f"in={self.original.in_features}, out={self.original.out_features}, "
                f"rank={self.rank}, alpha={self.alpha}")


def apply_lora(model: MambaCoTModel, rank: int = 64, alpha: int = 64,
               target_modules: List[str] = None, dropout: float = 0.05):
    if target_modules is None:
        target_modules = ["in_proj", "out_proj"]

    for param in model.backbone.parameters():
        param.requires_grad = False

    model.backbone.backbone.embedding.weight.requires_grad = True
    model.backbone.lm_head.weight.requires_grad = True

    lora_count = 0
    for name, module in model.backbone.named_modules():
        for target in target_modules:
            if name.endswith(target) and isinstance(module, nn.Linear):
                parts = name.rsplit(".", 1)
                if len(parts) == 2:
                    parent_name, attr_name = parts
                    parent = dict(model.backbone.named_modules())[parent_name]
                else:
                    parent = model.backbone
                    attr_name = parts[0]

                lora_module = LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)
                setattr(parent, attr_name, lora_module)
                lora_count += 1
                logger.info(f"Applied LoRA to {name} (rank={rank})")
                break

    for param in model.mtp_module.parameters():
        param.requires_grad = True

    logger.info(f"Applied {lora_count} LoRA adapters. "
                f"Backbone frozen, embeddings/head/MTP trainable.")

    return lora_count


def count_parameters(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

PLAN_TIERS = ("none", "plan", "tags+plan", "full")

PLAN_COT_MODES = ("plan", "tags+plan", "full")
LEGACY_COT_MODES = ("prose", "tags", "none")

PLAN_DATASET_DEFAULTS = {
    "plan_weight": 1.0,
    "tag_weight": None,
    "plan_schedule_from": "gt",
    "min_plan_coverage": 0.0,
    "style_tier": "none",
    "style_dropout": 0.0,
}

PLAN_BIN_ORDER = ("pitch", "duration", "energy")
PLAN_BIN_ABSENT = -1

PLAN_DRY_STEPS = 200


def _is_plan_param(name: str) -> bool:
    return any(part.startswith("plan_") for part in name.split("."))


def _plan_kwargs_for(fn, kwargs: Dict[str, Any], what: str,
                     defaults: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    params = inspect.signature(fn).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(kwargs)
    accepted = {k: v for k, v in kwargs.items() if k in params}
    lost = [k for k in kwargs
            if k not in params
            and (defaults is None or kwargs[k] != defaults.get(k))]
    if lost:
        raise RuntimeError(
            f"{what} does not accept {lost} — those P²-CoT settings were "
            f"requested on the command line and would be SILENTLY IGNORED. "
            f"Its signature is ({', '.join(params)}). Either the flag names or "
            f"the {what} signature drifted; fix the seam, do not drop the flag."
        )
    return accepted


def plan_dataset_kwargs(args) -> Dict[str, Any]:
    tier = getattr(args, "plan_tier", "none")
    style = getattr(args, "style_tier", "none")
    if tier == "none" and style == "none":
        return {}
    tagw = getattr(args, "plan_tag_weight", None)
    kw: Dict[str, Any] = {
        "plan_weight": float(getattr(args, "plan_loss_weight", 1.0)),
        "tag_weight": None if tagw is None else float(tagw),
        "plan_schedule_from": getattr(args, "plan_schedule_from", "gt"),
        "min_plan_coverage": float(getattr(args, "min_plan_coverage", 0.0)),
    }
    if style != "none":
        kw["style_tier"] = style
        kw["style_dropout"] = float(getattr(args, "style_dropout", 0.0))
    return kw


def plan_bins_from_slots(input_ids, plan_slots, registry):
    families = (registry.pitch_ids, registry.duration_ids, registry.energy_ids)
    bases = []
    for name, ids in zip(PLAN_BIN_ORDER, families):
        ids = list(ids or [])
        if not ids:
            raise RuntimeError(
                f"plan_bins_from_slots: the tokenizer has no {name} bin ids "
                f"(MVC_ENABLE_PLAN_TOKENS off for this process?) — the batch "
                f"carries a plan the vocabulary cannot express.")
        if ids != list(range(ids[0], ids[0] + len(ids))):
            raise RuntimeError(
                f"plan_bins_from_slots: {name} ids are not contiguous ({ids[:4]}"
                f"...), so id-minus-base would mis-decode every bin.")
        bases.append((ids[0], len(ids)))

    slots = plan_slots.to(input_ids.device)
    if slots.dim() != 3 or slots.shape[-1] != len(PLAN_BIN_ORDER):
        raise ValueError(f"plan_slots must be [B, W, {len(PLAN_BIN_ORDER)}], "
                         f"got {tuple(slots.shape)}")
    B, W, C = slots.shape
    valid = slots >= 0
    tok = input_ids.gather(1, slots.clamp(min=0).reshape(B, W * C)).reshape(B, W, C)

    bins = torch.full_like(slots, PLAN_BIN_ABSENT)
    for c, (lo, n) in enumerate(bases):
        b = tok[..., c] - lo
        bad = valid[..., c] & ((b < 0) | (b >= n))
        if bool(bad.any()):
            i, w = (bad.nonzero()[0]).tolist()
            raise RuntimeError(
                f"plan_slots[{i}][{w}][{c}] points at position "
                f"{int(slots[i, w, c])} holding token id {int(tok[i, w, c])}, "
                f"which is not a {PLAN_BIN_ORDER[c]} bin token "
                f"(ids {lo}..{lo + n - 1}). The collator's plan layout and the "
                f"tokenizer's plan block disagree.")
        bins[..., c] = torch.where(valid[..., c], b,
                                   torch.full_like(b, PLAN_BIN_ABSENT))
    return bins


def plan_forward_params(model) -> set:
    m = model.module if isinstance(model, DDP) else model
    return {p for p in inspect.signature(m.forward).parameters
            if p.startswith("plan_")}


def plan_forward_kwargs(batch, accepted: set, registry, device):
    plan_keys = {k: v for k, v in batch.items()
                 if k.startswith("plan_") and v is not None}
    if not plan_keys:
        return {}, []

    consumed = set()
    if ("plan_bins" in accepted and "plan_bins" not in plan_keys
            and "plan_slots" in plan_keys):
        plan_keys["plan_bins"] = plan_bins_from_slots(
            batch["input_ids"].to(device), plan_keys["plan_slots"], registry)
        consumed.add("plan_slots")
    if "plan_bins" in accepted and "plan_bins" not in plan_keys:
        raise RuntimeError(
            f"the model's forward expects plan_bins but this batch carries "
            f"neither plan_bins nor plan_slots (keys: {sorted(plan_keys)}) — "
            f"the injection would receive nothing and the plan tier would be "
            f"silently inert.")

    out = {k: (v.to(device) if torch.is_tensor(v) else v)
           for k, v in plan_keys.items() if k in accepted}
    unused = sorted(k for k in plan_keys
                    if k not in accepted and k not in consumed)
    return out, unused


def absent_plan_kwargs(input_ids, accepted: set, device) -> Dict[str, Any]:
    if "plan_bins" not in accepted or "plan_schedule" not in accepted:
        return {}
    B, S = input_ids.shape
    return {
        "plan_bins": torch.full((B, 1, len(PLAN_BIN_ORDER)), PLAN_BIN_ABSENT,
                                dtype=torch.long, device=device),
        "plan_schedule": torch.full((B, S), PLAN_BIN_ABSENT, dtype=torch.long,
                                    device=device),
    }


def plan_corpus_stats(dataset, registry, sample: int = 512,
                      seed: int = 20260817, max_seq_len: int = 0) -> dict:
    n = len(dataset)
    if n == 0:
        raise RuntimeError("plan_corpus_stats: the dataset is empty")
    k = min(sample, n)
    idx = (list(range(n)) if k == n
           else random.Random(seed).sample(range(n), k))

    n_with, plan_lens, max_len, n_over = 0, [], 0, 0
    speech_fracs = []
    with torch.random.fork_rng(devices=[]):
        for i in idx:
            item = dataset[i]
            ids = item["input_ids"]
            ids = ids.tolist() if hasattr(ids, "tolist") else list(ids)
            n_plan = sum(1 for t in ids if registry.is_plan_token(int(t)))
            if n_plan:
                n_with += 1
                plan_lens.append(n_plan)
            max_len = max(max_len, len(ids))
            if max_seq_len and len(ids) > max_seq_len:
                n_over += 1
            durs = item.get("plan_durations")
            if durs is not None:
                sigma = float(sum(durs.tolist() if hasattr(durs, "tolist")
                                  else durs))
                frames = int(item["audio_end"]) - int(item["audio_start"])
                if sigma > 0 and frames > 0:
                    speech_fracs.append(sigma / frames)
    speech_fracs.sort()
    return {
        "n_rows": n,
        "n_sampled": k,
        "n_with_plan": n_with,
        "n_without_plan": k - n_with,
        "frac_with_plan": n_with / k,
        "mean_plan_tokens": (sum(plan_lens) / len(plan_lens)) if plan_lens else 0.0,
        "max_plan_tokens": max(plan_lens) if plan_lens else 0,
        "max_seq_len_seen": max_len,
        "n_over_window": n_over,
        "speech_frac_median": (speech_fracs[len(speech_fracs) // 2]
                               if speech_fracs else None),
        "n_speech_frac": len(speech_fracs),
    }


def log_plan_stats(dataset, registry, args, sample: int = 512, rank: int = 0,
                   is_ddp: bool = False) -> dict:
    stats = plan_corpus_stats(dataset, registry, sample=sample,
                              max_seq_len=getattr(args, "max_seq_len", 0))
    if rank == 0:
        logger.info(
            "PLAN TIER: --plan_tier %s | injection=%s | schedule_from=%s | "
            "loss_weight=%s | scaffold_dropout=%s",
            getattr(args, "plan_tier", "none"),
            bool(getattr(args, "plan_injection", False)),
            getattr(args, "plan_schedule_from", "gt"),
            getattr(args, "plan_loss_weight", 1.0),
            getattr(args, "plan_scaffold_dropout", 0.0))
        logger.info(
            "PLAN TIER: %d/%d sampled rows carry a plan (%.1f%% of %d rows); "
            "mean %.1f plan tokens/row (max %d)",
            stats["n_with_plan"], stats["n_sampled"],
            100.0 * stats["frac_with_plan"], stats["n_rows"],
            stats["mean_plan_tokens"], stats["max_plan_tokens"])
        logger.info(
            "PLAN TIER: longest assembled sequence %d tokens "
            "(--max_seq_len %d); %d sampled row(s) over the window",
            stats["max_seq_len_seen"], getattr(args, "max_seq_len", 0),
            stats["n_over_window"])
        if stats["speech_frac_median"] is not None:
            logger.info(
                "PLAN TIER: rho = SumD/T median %.3f over %d plan row(s) — pass "
                "`--plan_speech_ratio %.3f` to scripts/gen_fish_cot_samples.py "
                "when decoding with --plan_schedule stretch, or the program is "
                "compressed into the first %.0f%% of the utterance and the last "
                "word is held for the rest (a schedule the model never saw).",
                stats["speech_frac_median"], stats["n_speech_frac"],
                stats["speech_frac_median"], 100.0 * stats["speech_frac_median"])

    if stats["n_with_plan"] == 0:
        raise RuntimeError(
            f"--plan_tier {getattr(args, 'plan_tier', 'none')} but NONE of "
            f"{stats['n_sampled']} sampled rows contains a single plan token. "
            f"This run would train a plan arm on a plan-free corpus and report "
            f"it as one. Check that the corpus has plan sidecars, that "
            f"MVC_ENABLE_PLAN_TOKENS=1 was exported for THIS process, and that "
            f"the dataset accepted plan_tier.")
    if stats["frac_with_plan"] < 0.5:
        logger.warning(
            "PLAN TIER: only %.1f%% of sampled rows carry a plan — the arm is "
            "mostly plan-free and its contrast with the control is diluted "
            "accordingly. Intended only if the corpus is deliberately mixed.",
            100.0 * stats["frac_with_plan"])
    if (is_ddp and getattr(args, "plan_injection", False)
            and stats["frac_with_plan"] < 1.0):
        raise RuntimeError(
            f"--plan_injection under DDP needs EVERY batch to carry the plan "
            f"channel, but only {100.0 * stats['frac_with_plan']:.1f}% of "
            f"sampled rows have a plan. A plan-free batch drops the injector "
            f"out of the graph and DDP kills all ranks ('parameters that were "
            f"not used in producing loss'). Use a fully-planned corpus, or "
            f"make the collator emit an all-absent plan for every batch.")
    if stats["n_over_window"]:
        logger.warning(
            "PLAN TIER: %d/%d sampled rows exceed --max_seq_len %d — the plan "
            "block pushes them past the window and their audio is TRUNCATED "
            "(master-plan risk R2). Shorten Tier-A text or raise the window.",
            stats["n_over_window"], stats["n_sampled"],
            getattr(args, "max_seq_len", 0))
    return stats


class DPODataset(Dataset):
    def __init__(self, data_path: str, tokenizer, registry: TokenRegistry,
                 max_seq_len: int = 2048):
        self.tokenizer = tokenizer
        self.registry = registry
        self.max_seq_len = max_seq_len

        self.data = []
        with open(data_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.data.append(json.loads(line))

        self.bos_id = registry.bos_id
        self.eos_id = registry.eos_id
        self.user_prompt_id = registry.user_prompt_id
        self.think_start_id = registry.think_start_id
        self.think_end_id = registry.think_end_id
        self.audio_start_id = registry.audio_start_id
        self.audio_end_id = registry.audio_end_id

        logger.info(f"Loaded {len(self.data)} DPO preference pairs from {data_path}")

    def __len__(self):
        return len(self.data)

    def _build_sequence(self, item: dict, speech_tokens_key: str):
        prompt_text = item["prompt"]
        speech_values = item[speech_tokens_key]

        speech_ids = [self.registry.speech_value_to_id(v) for v in speech_values]

        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)

        reasoning_ids = []
        reasoning_text = item.get("reasoning", "")
        if reasoning_text:
            clean = reasoning_text.strip()
            if clean.startswith("<THINK>"):
                clean = clean[len("<THINK>"):].strip()
            if clean.endswith("</THINK>"):
                clean = clean[:-len("</THINK>")].strip()
            if clean:
                reasoning_ids = self.tokenizer.encode(clean, add_special_tokens=False)

        if reasoning_ids:
            sequence = (
                [self.bos_id, self.user_prompt_id]
                + prompt_ids
                + [self.think_start_id]
                + reasoning_ids
                + [self.think_end_id]
                + [self.audio_start_id]
                + speech_ids
                + [self.audio_end_id, self.eos_id]
            )
            prompt_end = 2 + len(prompt_ids)
        else:
            sequence = (
                [self.bos_id, self.user_prompt_id]
                + prompt_ids
                + [self.audio_start_id]
                + speech_ids
                + [self.audio_end_id, self.eos_id]
            )
            prompt_end = 2 + len(prompt_ids)

        if len(sequence) > self.max_seq_len:
            sequence = sequence[:self.max_seq_len]

        input_ids = torch.tensor(sequence, dtype=torch.long)

        labels = input_ids.clone()
        labels[:prompt_end] = -100

        return input_ids, labels

    def __getitem__(self, idx):
        item = self.data[idx]
        chosen_ids, chosen_labels = self._build_sequence(item, "chosen_speech_tokens")
        rejected_ids, rejected_labels = self._build_sequence(item, "rejected_speech_tokens")

        return {
            "input_ids_w": chosen_ids,
            "labels_w": chosen_labels,
            "input_ids_l": rejected_ids,
            "labels_l": rejected_labels,
        }


@dataclass
class DPODataCollator:
    pad_token_id: int

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        result = {}
        for key in ["input_ids_w", "labels_w", "input_ids_l", "labels_l"]:
            tensors = [item[key] for item in batch]
            pad_val = self.pad_token_id if "input" in key else -100
            padded = torch.nn.utils.rnn.pad_sequence(
                tensors, batch_first=True, padding_value=pad_val
            )
            result[key] = padded
        return result


def get_sequence_logprobs(model: MambaCoTModel, input_ids: torch.Tensor,
                          labels: torch.Tensor) -> torch.Tensor:
    outputs = model(input_ids=input_ids)
    logits = outputs["logits"]

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    log_probs = F.log_softmax(shift_logits, dim=-1)
    gather_labels = shift_labels.clone()
    gather_labels[gather_labels == -100] = 0
    per_token = log_probs.gather(-1, gather_labels.unsqueeze(-1)).squeeze(-1)

    mask = (shift_labels != -100).float()
    return (per_token * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)


def compute_dpo_loss(model: MambaCoTModel, ref_model: MambaCoTModel,
                     batch: Dict[str, torch.Tensor],
                     beta: float = 0.5, alpha: float = 0.01,
                     kl_ceiling: float = 5.0) -> tuple:
    logprobs_w = get_sequence_logprobs(model, batch["input_ids_w"], batch["labels_w"])
    logprobs_l = get_sequence_logprobs(model, batch["input_ids_l"], batch["labels_l"])

    with torch.no_grad():
        ref_logprobs_w = get_sequence_logprobs(ref_model, batch["input_ids_w"], batch["labels_w"])
        ref_logprobs_l = get_sequence_logprobs(ref_model, batch["input_ids_l"], batch["labels_l"])

    log_ratio_w = logprobs_w - ref_logprobs_w
    log_ratio_l = logprobs_l - ref_logprobs_l
    dpo_loss = -F.logsigmoid(beta * (log_ratio_w - log_ratio_l)).mean()

    entropy_reg = alpha * (logprobs_w.mean() + logprobs_l.mean()) / 2.0

    kl_w = (logprobs_w - ref_logprobs_w).mean()
    kl_l = (logprobs_l - ref_logprobs_l).mean()
    kl_mean = (kl_w + kl_l) / 2.0
    kl_penalty = F.relu(kl_mean - kl_ceiling)

    total_loss = dpo_loss + entropy_reg + kl_penalty

    metrics = {
        "dpo_loss": dpo_loss.item(),
        "entropy_reg": entropy_reg.item(),
        "kl_mean": kl_mean.item(),
        "kl_penalty": kl_penalty.item(),
        "logprobs_w": logprobs_w.mean().item(),
        "logprobs_l": logprobs_l.mean().item(),
    }

    return total_loss, metrics


def _cot_embedding_rows(data_path, model, args) -> set:
    reg = model.token_registry
    tok = model.tokenizer
    mode = getattr(args, "cot_mode", None)
    if mode != "tags":
        raise RuntimeError(
            f"--train_cot_embeddings_only expects --cot_mode tags (got "
            f"{mode!r}): prose reasoning would whitelist most of the "
            f"vocabulary, which is not the frozen-backbone floor this arm is "
            f"meant to measure.")
    keep = {reg.think_start_id, reg.think_end_id}
    plan_structural = set(getattr(reg, "plan_token_ids", ()) or ())
    keep |= plan_structural
    prompt_tokens = set()
    seen, n_cot = set(), 0
    with open(data_path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            p = str(d.get("prompt") or "")
            if p:
                prompt_tokens.update(tok.encode(p, add_special_tokens=False))
            if not str(d.get("reasoning") or "").strip():
                continue
            n_cot += 1
            txt = (f"EMO={d.get('emotion_label') or 'neutral'} "
                   f"PACE={d.get('pace') or 'normal'} "
                   f"PITCH={d.get('pitch') or 'normal'}")
            if txt not in seen:
                seen.add(txt)
                keep.update(tok.encode(txt, add_special_tokens=False))

    for extra in (getattr(args, "f1_exclude_prompts_from", None) or []):
        if not os.path.exists(extra):
            continue
        with open(extra, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                p = str(d.get("prompt") or "")
                if p:
                    prompt_tokens.update(
                        tok.encode(p, add_special_tokens=False))

    structural = {reg.think_start_id, reg.think_end_id} | plan_structural
    dropped = (keep - structural) & prompt_tokens
    keep = (keep - prompt_tokens) | structural
    if len(keep) <= len(structural) or not n_cot:
        raise RuntimeError(
            f"no trainable reasoning tokens in {data_path} ({n_cot} CoT rows, "
            f"{len(dropped)} tag tokens dropped for also appearing in prompts) "
            f"— this arm would train nothing beyond the structural tokens.")
    logger.info(
        f"F1 whitelist: {len(keep)} rows ({len(dropped)} tag tokens dropped "
        f"because they also occur in prompt text, which is what keeps the "
        f"base-domain forward bitwise identical)")
    return keep


def build_optimizer(model: MambaCoTModel, args) -> torch.optim.Optimizer:
    if args.stage == 1 and getattr(args, "full_finetune", False):
        hybrid_prefixes = tuple(
            f"backbone.backbone.layers.{i}.mixer"
            for i in getattr(model, "hybrid_replaced_layers", []))

        def _is_fresh(name: str) -> bool:
            return (
                name.startswith("backbone.backbone.embedding")
                or name.startswith("backbone.lm_head")
                or name.startswith("mtp_module")
                or name.startswith("depth_module")
                or name.startswith("flow_head")
                or _is_plan_param(name)
                or (bool(hybrid_prefixes)
                    and name.startswith(hybrid_prefixes))
            )

        backbone_params = []
        fresh_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if _is_fresh(name):
                fresh_params.append(param)
            else:
                backbone_params.append(param)

        param_groups = [
            {"params": backbone_params, "lr": args.lr_backbone},
            {"params": fresh_params, "lr": args.lr_new_params},
        ]
        logger.info(
            f"Stage 1 full-FT optimizer: {len(backbone_params)} backbone params "
            f"@ {args.lr_backbone}, {len(fresh_params)} fresh "
            f"(embed/lm_head/MTP) params @ {args.lr_new_params}"
        )
    elif args.stage == 1:
        lora_params = []
        new_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "lora_A" in name or "lora_B" in name:
                lora_params.append(param)
            else:
                new_params.append(param)

        param_groups = [
            {"params": new_params, "lr": args.lr_new_params},
            {"params": lora_params, "lr": args.lr_lora},
        ]
        logger.info(f"Stage 1 optimizer: {len(new_params)} new params @ {args.lr_new_params}, "
                     f"{len(lora_params)} LoRA params @ {args.lr_lora}")
    else:
        lr = args.lr or STAGE_DEFAULTS[args.stage]["lr"]
        spk, rest = [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            (spk if "speaker_encoder" in name else rest).append(param)
        lr_speaker = getattr(args, "lr_speaker", 0.0)
        if lr_speaker and spk:
            param_groups = [{"params": rest, "lr": lr},
                            {"params": spk, "lr": lr_speaker}]
            logger.info(f"Stage {args.stage} optimizer: {len(rest)} params @ {lr}, "
                        f"{len(spk)} SPEAKER params @ {lr_speaker}")
        else:
            trainable = rest + spk
            param_groups = [{"params": trainable, "lr": lr}]
            logger.info(f"Stage {args.stage} optimizer: {len(trainable)} params @ {lr}")

    optimizer = torch.optim.AdamW(
        param_groups, betas=(0.9, 0.95),
        weight_decay=getattr(args, "weight_decay", 0.1),
        eps=getattr(args, "adam_eps", 1e-8),
    )
    return optimizer


def build_scheduler(optimizer: torch.optim.Optimizer, args):
    lr_min_ratio = args.lr_min / max(args.lr or STAGE_DEFAULTS[args.stage]["lr"], 1e-10)
    grad_accum = max(1, getattr(args, "grad_accum", 1) or 1)
    total_updates = max(1, args.max_steps // grad_accum)
    warmup_updates = max(1, args.warmup_steps // grad_accum)

    def lr_lambda(update):
        if update < warmup_updates:
            return update / max(1, warmup_updates)
        progress = (update - warmup_updates) / max(1, total_updates - warmup_updates)
        return max(lr_min_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def next_batch(data_iter, loader):
    try:
        return next(data_iter), data_iter
    except StopIteration:
        data_iter = iter(loader)
        return next(data_iter), data_iter


def train_loop(model, loader, optimizer, scheduler, device, args, rank, start_step=0):
    model.train()
    use_weighted_loss = (
        STAGE_DEFAULTS[args.stage]["use_weighted_loss"]
        or getattr(args, "semantic_weight", 1.0) != 1.0
        or getattr(args, "eos_weight", 1.0) != 1.0
    )

    if start_step > 0 and isinstance(getattr(loader, "sampler", None), DistributedSampler):
        loader.sampler.set_epoch(max(1, start_step // max(1, len(loader))))
    data_iter = iter(loader)

    optimizer.zero_grad()
    running_loss = 0.0
    running_main = 0.0
    running_mtp = 0.0
    running_flow = 0.0
    flow_codec = getattr(args, "_flow_codec", None)

    plan_accepted = plan_forward_params(model)
    plan_registry = getattr(model.module if isinstance(model, DDP) else model,
                            "token_registry", None)
    plan_seen = False

    for step in range(start_step + 1, args.max_steps + 1):
        batch, data_iter = next_batch(data_iter, loader)

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        speech_mask = (batch["speech_mask"].to(device)
                       if "speech_mask" in batch else None)
        loss_weights = (batch["loss_weights"].to(device)
                        if use_weighted_loss and "loss_weights" in batch else None)
        residual_codes = batch.get("residual_codes")
        if residual_codes is not None:
            residual_codes = residual_codes.to(device)
        speaker_ids = batch.get("speaker_ids")
        if speaker_ids is not None:
            speaker_ids = speaker_ids.to(device)
            mx = int(speaker_ids.max().item())
            if mx >= args.num_speakers:
                raise RuntimeError(
                    f"speaker_id {mx} >= --num_speakers {args.num_speakers}: "
                    f"the corpus carries ids outside the embedding table — "
                    f"re-map speakers or expand the table "
                    f"(scripts/expand_speaker_table.py)")
        prosody_vecs = batch.get("prosody_vecs")
        if prosody_vecs is not None:
            prosody_vecs = prosody_vecs.to(device)

        plan_batch, plan_unused = plan_forward_kwargs(
            batch, plan_accepted, plan_registry, device)
        if plan_batch and not plan_seen:
            plan_seen = True
            if plan_unused and rank == 0:
                logger.warning(
                    "PLAN TIER: collator keys %s are not parameters of "
                    "model.forward and are unused during training (expected for "
                    "the decode-window channels; investigate anything else).",
                    plan_unused)
        elif not plan_seen and getattr(args, "plan_injection", False):
            raw = sorted(k for k in batch if k.startswith("plan_"))
            if raw:
                raise RuntimeError(
                    f"--plan_injection is on and the collator emitted {raw}, but "
                    f"none of it reached model.forward, which accepts "
                    f"{sorted(plan_accepted) or 'no plan parameter'}. The "
                    f"injection would train on an all-zero channel and the run "
                    f"would look like a plan arm that merely did not work.")
            if step - start_step >= PLAN_DRY_STEPS:
                raise RuntimeError(
                    f"--plan_injection is on but NO batch in the first "
                    f"{PLAN_DRY_STEPS} steps carried a plan channel. The corpus "
                    f"has no usable plans (the startup audit samples rows; this "
                    f"sees every batch), so the injection is training on "
                    f"nothing.")
        if not plan_batch and getattr(args, "plan_injection", False):
            plan_batch = absent_plan_kwargs(batch["input_ids"], plan_accepted,
                                            device)

        target_latents = None
        if flow_codec is not None and "codes" in batch:
            with torch.no_grad():
                cg = batch["codes"].to(device).clamp(min=0).transpose(1, 2)
                target_latents = flow_codec.codes_to_latents(cg).float()

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            if "codes" in batch:
                outputs = model(
                    input_ids=input_ids,
                    labels=labels,
                    speaker_ids=speaker_ids,
                    codes=batch["codes"].to(device),
                    prosody_vecs=prosody_vecs,
                    semantic_weight=getattr(args, "semantic_weight", 1.0),
                    eos_weight=getattr(args, "eos_weight", 1.0),
                    pause_exit_weight=getattr(args, "pause_exit_weight", 1.0),
                    pause_codes=getattr(args, "pause_codes", None),
                    target_latents=target_latents,
                    loss_weights=(batch["loss_weights"].to(device)
                                  if "loss_weights" in batch else None),
                    **plan_batch,
                )
            else:
                outputs = model(
                    input_ids=input_ids,
                    labels=labels,
                    speech_mask=speech_mask,
                    loss_weights=loss_weights,
                    residual_codes=residual_codes,
                    speaker_ids=speaker_ids,
                    **plan_batch,
                )
            l2sp_w = getattr(args, "l2sp_weight", 0.0)
            l2sp_term = None

            loss = outputs["loss"] / args.grad_accum

        loss.backward()

        if step % args.grad_accum == 0:
            if l2sp_w > 0:
                _um = model.module if isinstance(model, DDP) else model
                _ref = getattr(_um, "_l2sp_ref", None)
                if _ref is not None:
                    _acc = 0.0
                    with torch.no_grad():
                        for _n, _p in _um.backbone.named_parameters():
                            _r = _ref.get(_n)
                            if _r is None or _p.grad is None:
                                continue
                            _d = _p.detach() - _r.to(_p.device).to(_p.dtype)
                            _p.grad.add_(_d, alpha=2.0 * l2sp_w)
                            _acc += float(_d.pow(2).sum())
                    l2sp_term = torch.tensor(l2sp_w * _acc)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=1.0,
            )
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        running_loss += outputs["loss"].item()
        running_main += outputs["main_loss"].item()
        running_mtp += outputs["mtp_loss"].item()
        _fl = outputs.get("flow_loss")
        running_flow += _fl.item() if _fl is not None else 0.0

        if rank == 0 and step % args.log_every == 0:
            avg_loss = running_loss / args.log_every
            avg_main = running_main / args.log_every
            avg_mtp = running_mtp / args.log_every
            avg_flow = running_flow / args.log_every
            lr_current = optimizer.param_groups[0]["lr"]

            l2sp_str = (f" | l2sp: {l2sp_term.item():.4f}"
                        if l2sp_term is not None else "")
            flow_str = f" | flow: {avg_flow:.4f}" if flow_codec is not None else ""
            logger.info(
                f"[Stage {args.stage}] Step {step}/{args.max_steps} | "
                f"Loss: {avg_loss:.4f} (main: {avg_main:.4f}, mtp: {avg_mtp:.4f}) | "
                f"LR: {lr_current:.2e}{l2sp_str}{flow_str}"
            )

            running_loss = 0.0
            running_main = 0.0
            running_mtp = 0.0
            running_flow = 0.0

        if rank == 0 and step % args.save_every == 0:
            save_checkpoint(model, optimizer, scheduler, step, args)

        if args.val_data_path and rank == 0 and step % args.val_every == 0:
            val_loss = validate(model, args, device)
            logger.info(f"[Stage {args.stage}] Step {step} | Val Loss: {val_loss:.4f}")
            model.train()

    if rank == 0:
        save_checkpoint(model, optimizer, scheduler, args.max_steps, args)

    logger.info(f"[Stage {args.stage}] Training complete ({args.max_steps} steps).")


def dpo_train_loop(model, ref_model, loader, optimizer, scheduler, device, args, rank):
    model.train()
    ref_model.eval()
    data_iter = iter(loader)

    optimizer.zero_grad()
    running_loss = 0.0

    for step in range(1, args.max_steps + 1):
        batch, data_iter = next_batch(data_iter, loader)

        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            total_loss, metrics = compute_dpo_loss(
                model=model,
                ref_model=ref_model,
                batch=batch,
                beta=args.dpo_beta,
                alpha=args.dpo_alpha,
                kl_ceiling=args.dpo_kl_ceiling,
            )
            scaled_loss = total_loss / args.grad_accum

        scaled_loss.backward()

        if step % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=1.0,
            )
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        running_loss += total_loss.item()

        if rank == 0 and step % args.log_every == 0:
            avg_loss = running_loss / args.log_every
            lr_current = optimizer.param_groups[0]["lr"]

            logger.info(
                f"[Stage 4 DPO] Step {step}/{args.max_steps} | "
                f"Loss: {avg_loss:.4f} | "
                f"DPO: {metrics['dpo_loss']:.4f} | "
                f"Entropy: {metrics['entropy_reg']:.4f} | "
                f"KL: {metrics['kl_mean']:.4f} | "
                f"LR: {lr_current:.2e}"
            )
            running_loss = 0.0

        if rank == 0 and step % args.save_every == 0:
            save_checkpoint(model, optimizer, scheduler, step, args)

    if rank == 0:
        save_checkpoint(model, optimizer, scheduler, args.max_steps, args)

    logger.info(f"[Stage 4 DPO] Training complete ({args.max_steps} steps).")


def save_checkpoint(model, optimizer, scheduler, step, args):
    os.makedirs(args.output_dir, exist_ok=True)
    path = os.path.join(args.output_dir, f"checkpoint_step_{step}.pt")

    model_to_save = model.module if isinstance(model, DDP) else model

    checkpoint = {
        "step": step,
        "stage": args.stage,
        "model_state_dict": model_to_save.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "args": vars(args),
    }

    torch.save(checkpoint, path)
    logger.info(f"Saved checkpoint to {path}")

_CRITICAL_PROJ = (
    "in_proj.weight", "out_proj.weight",
    "query_key_value.weight", "attention.dense.weight",
)


def _remap_plain_keys_for_lora(sd, model_sd):
    out = {}
    n_remapped = 0
    for k, v in sd.items():
        if k not in model_sd:
            head, _, leaf = k.rpartition(".")
            candidate = f"{head}.original.{leaf}" if head else k
            if candidate in model_sd:
                out[candidate] = v
                n_remapped += 1
                continue
        out[k] = v
    return out, n_remapped

_VOCAB_PAD_SLACK = 256


def _vocab_row_keys(model, model_sd) -> set:
    reg = getattr(model, "token_registry", None)
    tk = getattr(reg, "tokenizer", None)
    if tk is None:
        return set()
    try:
        v_model = int(len(tk))
    except TypeError:
        return set()
    return {k for k, t in model_sd.items()
            if getattr(t, "ndim", 0) == 2
            and v_model <= int(t.shape[0]) < v_model + _VOCAB_PAD_SLACK}

_SPEAKER_TABLE_SUFFIX = "speaker_encoder.embedding.weight"


def _is_safetensors(path) -> bool:
    return str(path).endswith(".safetensors")


def read_checkpoint(path, map_location="cpu", mmap=False):
    if not _is_safetensors(path):
        return torch.load(path, map_location=map_location, weights_only=False, mmap=mmap)
    from safetensors import safe_open
    from safetensors.torch import load_file
    with safe_open(str(path), framework="pt") as f:
        meta = f.metadata() or {}
    out = {"model_state_dict": load_file(str(path), device=str(map_location or "cpu"))}
    if str(meta.get("step", "")).isdigit():
        out["step"] = int(meta["step"])
    return out


def checkpoint_tensor_shapes(path):
    if _is_safetensors(path):
        from safetensors import safe_open
        with safe_open(str(path), framework="pt") as f:
            return {k: tuple(f.get_slice(k).get_shape()) for k in f.keys()}
    peek = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    sd = peek.get("model_state_dict", peek) if isinstance(peek, dict) else {}
    return {k: tuple(getattr(v, "shape", ())) for k, v in sd.items()}


def checkpoint_speaker_rows(path):
    if _is_safetensors(path):
        try:
            shapes = checkpoint_tensor_shapes(path)
        except Exception:
            return None
        return next((int(s[0]) for k, s in shapes.items()
                     if k.endswith(_SPEAKER_TABLE_SUFFIX) and len(s) == 2), None)
    try:
        peek = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except Exception:
        return None
    sd = peek.get("model_state_dict", peek) if isinstance(peek, dict) else {}
    for k, v in sd.items():
        if k.endswith(_SPEAKER_TABLE_SUFFIX) and getattr(v, "ndim", 0) == 2:
            return int(v.shape[0])
    return None


def _splice_plan_vocab_rows(old, new_like, plan_lo: int, n_plan: int,
                            v_old_real: int):
    v_old_pad, v_new_pad = int(old.shape[0]), int(new_like.shape[0])
    v_old_real = int(v_old_real)
    if tuple(old.shape[1:]) != tuple(new_like.shape[1:]):
        raise RuntimeError(
            f"vocab splice: trailing dims differ ({tuple(old.shape)} vs "
            f"{tuple(new_like.shape)}) — this is not a pure vocabulary resize.")
    if not 0 <= plan_lo <= v_old_real:
        raise RuntimeError(f"vocab splice: insertion point {plan_lo} outside "
                           f"[0, {v_old_real}]")
    if v_old_real > v_old_pad:
        raise RuntimeError(
            f"vocab splice: checkpoint tensor has {v_old_pad} rows but the "
            f"pre-plan vocabulary is {v_old_real} tokens — this checkpoint was "
            f"not produced by the vocabulary this model was built with "
            f"(MVC_NUM_SPEECH_TOKENS / MVC_NUM_STYLE_TOKENS mismatch?).")
    tail = v_old_real - plan_lo
    if plan_lo + n_plan + tail > v_new_pad:
        raise RuntimeError(
            f"vocab splice does not fit: {plan_lo} kept + {n_plan} plan + "
            f"{tail} shifted = {plan_lo + n_plan + tail} rows > the model's "
            f"{v_new_pad}.")
    out = new_like.clone()
    src = old.to(device=out.device, dtype=out.dtype)
    out[:plan_lo] = src[:plan_lo]
    out[plan_lo + n_plan:plan_lo + n_plan + tail] = src[plan_lo:v_old_real]
    return out


def load_checkpoint(path, model, optimizer=None, scheduler=None, device="cuda",
                    splice_plan_vocab: bool = False,
                    allow_speaker_table_reinit: bool = False):
    checkpoint = read_checkpoint(path, map_location=device)

    sd = checkpoint["model_state_dict"]
    model_sd = model.state_dict()

    ckpt_has_lora = any("lora_A" in k or "lora_B" in k for k in sd)
    model_has_lora = any("lora_A" in k or "lora_B" in k for k in model_sd)
    if ckpt_has_lora and not model_has_lora:
        raise RuntimeError(
            f"Checkpoint {path} contains LoRA adapters but the model has no "
            f"LoRA modules — loading would silently drop all adapter weights. "
            f"Either apply LoRA before loading (--stage 1 does this), or merge "
            f"the adapters first: scripts/merge_lora.py --checkpoint {path}"
        )

    ckpt_plan_keys = [k for k in sd if _is_plan_param(k)]
    model_plan_keys = [k for k in model_sd if _is_plan_param(k)]
    if ckpt_plan_keys and not model_plan_keys:
        raise RuntimeError(
            f"Checkpoint {path} carries {len(ckpt_plan_keys)} P²-CoT plan-tier "
            f"tensor(s) (e.g. {ckpt_plan_keys[0]}) but the model has none — "
            f"loading would silently drop the whole plan injection and train/"
            f"decode as an unplanned arm. Rebuild the model with "
            f"--plan_injection (and export MVC_ENABLE_PLAN_TOKENS=1 so the "
            f"vocabulary matches), or load a pre-plan checkpoint."
        )

    if model_has_lora and not ckpt_has_lora:
        sd, n_remapped = _remap_plain_keys_for_lora(sd, model_sd)
        if n_remapped:
            logger.info("Remapped %d plain checkpoint key(s) onto LoRA "
                        "'.original.' targets (full-FT/merged ckpt -> "
                        "LoRA-wrapped model)", n_remapped)

    replaced = getattr(model, "hybrid_replaced_layers", None) or []
    if replaced:
        pref = tuple(f"backbone.backbone.layers.{i}.mixer." for i in replaced)
        ckpt_mixer_is_mamba = any(k.startswith(pref) and "A_log" in k for k in sd)
        if ckpt_mixer_is_mamba:
            hyb_dropped = [k for k in sd if k.startswith(pref)]
            sd = {k: v for k, v in sd.items() if k not in hyb_dropped}
            logger.info("Hybrid warm-start: dropped %d Mamba mixer key(s) for "
                        "replaced layers %s (attention mixers keep fresh init)",
                        len(hyb_dropped), replaced)

    dropped = [k for k, v in sd.items()
               if k in model_sd and tuple(model_sd[k].shape) != tuple(v.shape)]

    vocab_keys = _vocab_row_keys(model, model_sd)
    vocab_dropped = [k for k in dropped if k in vocab_keys]
    if vocab_dropped:
        reg = getattr(model, "token_registry", None)
        plan_lo = getattr(reg, "plan_token_id_min", None)
        n_plan = len(getattr(reg, "plan_token_ids", ()) or ())
        v_new_real = len(reg.tokenizer) if getattr(reg, "tokenizer", None) else 0
        v_old_real = v_new_real - n_plan
        can_splice = (plan_lo is not None and n_plan > 0 and v_new_real > 0
                      and all(int(sd[k].shape[0]) >= v_old_real
                              and int(model_sd[k].shape[0]) >= v_new_real
                              and int(model_sd[k].shape[0]) > int(sd[k].shape[0])
                              for k in vocab_dropped))
        if splice_plan_vocab and can_splice:
            for k in vocab_dropped:
                sd[k] = _splice_plan_vocab_rows(sd[k], model_sd[k], plan_lo,
                                                n_plan, v_old_real)
            dropped = [k for k in dropped if k not in vocab_dropped]
            logger.warning(
                "PLAN TIER: spliced %d vocab-sized tensor(s) from %s — ids "
                "[0,%d) unchanged, ids [%d,%d) shifted up by %d, the %d plan "
                "rows and any padding keep the model's fresh init: %s",
                len(vocab_dropped), path, plan_lo, plan_lo, v_old_real, n_plan,
                n_plan, ", ".join(sorted(vocab_dropped)))
        else:
            shapes = ", ".join(
                f"{k}: ckpt{tuple(sd[k].shape)} vs model{tuple(model_sd[k].shape)}"
                for k in sorted(vocab_dropped)[:4])
            raise RuntimeError(
                f"{len(vocab_dropped)} VOCABULARY-sized tensor(s) in checkpoint "
                f"{path} have a different row count than the model ({shapes}). "
                f"Dropping them would silently re-initialise the token embedding "
                f"and/or the LM head — a cold start wearing a warm start's name. "
                f"This is what MVC_ENABLE_PLAN_TOKENS=1 does to a pre-P²-CoT "
                f"checkpoint (the plan block adds {n_plan or 'N'} rows below the "
                f"speech block). Fix by either: (a) passing "
                f"splice_plan_vocab=True / --splice_plan_vocab to remap the rows "
                f"exactly" + ("" if can_splice else
                              " (NOT applicable here — the row delta is not the "
                              "plan-block size)") +
                f", or (b) matching the gate/MVC_NUM_SPEECH_TOKENS this "
                f"checkpoint was trained under.")

    if dropped:
        arch = [k for k in dropped if k.endswith(_CRITICAL_PROJ)]
        if arch:
            raise RuntimeError(
                f"{len(arch)} projection weight(s) in checkpoint {path} have a "
                f"different SHAPE than the model (e.g. {arch[0]}) — the model was "
                f"built with a different architecture than the one that produced "
                f"this checkpoint. Check --hybrid_attention_top_k / --backbone."
            )
        spk = [k for k in dropped if k.endswith(_SPEAKER_TABLE_SUFFIX)]
        if spk and not allow_speaker_table_reinit:
            k = spk[0]
            raise RuntimeError(
                f"Speaker table {k} in checkpoint {path} is "
                f"{tuple(sd[k].shape)} but the model was built as "
                f"{tuple(model_sd[k].shape)} (--num_speakers). Dropping it "
                f"re-initialises EVERY voice from scratch, and with input "
                f"injection that random vector enters the backbone at every "
                f"step — the 2026-09-21 WER battery ran 32768-voice checkpoints "
                f"through 8192-row tables this way. Pass --num_speakers "
                f"{int(sd[k].shape[0])} (inference); for a warm start that grows "
                f"the table run scripts/expand_speaker_table.py first; "
                f"--allow_speaker_table_reinit keeps the old silent drop on "
                f"purpose."
            )
        logger.warning("Dropping %d shape-mismatched checkpoint key(s) (re-init "
                       "from scratch): %s", len(dropped),
                       ", ".join(dropped[:6]) + (" ..." if len(dropped) > 6 else ""))
        sd = {k: v for k, v in sd.items() if k not in dropped}

    incompat = model.load_state_dict(sd, strict=False)

    unexpected = list(incompat.unexpected_keys)
    critical = [k for k in unexpected if k.endswith(_CRITICAL_PROJ)]
    if critical:
        raise RuntimeError(
            f"{len(critical)} projection weight(s) in checkpoint {path} match no "
            f"model key and would be silently dropped (e.g. {critical[0]}) — this "
            f"is the full-FT-ckpt/LoRA-model mismatch that produces base-weight "
            f"babble. Load with the matching LoRA rank (0 for full-FT checkpoints)."
        )
    if unexpected:
        logger.warning("load_state_dict ignored %d unexpected checkpoint key(s): %s",
                       len(unexpected),
                       ", ".join(unexpected[:6]) + (" ..." if len(unexpected) > 6 else ""))
    missing = [k for k in incompat.missing_keys
               if not (("lora_A" in k or "lora_B" in k) and not ckpt_has_lora)]

    plan_missing = [k for k in incompat.missing_keys if _is_plan_param(k)]
    if plan_missing:
        logger.warning(
            "PLAN TIER: %d plan tensor(s) received NO value from %s and keep "
            "their fresh init%s: %s", len(plan_missing), path,
            " (expected on a warm start from a pre-plan checkpoint)"
            if not ckpt_plan_keys else " — the checkpoint HAS plan tensors, so "
            "this is a real mismatch, not a graft",
            ", ".join(plan_missing))

    if missing:
        logger.warning("%d model key(s) received no checkpoint value (kept init): %s",
                       len(missing),
                       ", ".join(missing[:6]) + (" ..." if len(missing) > 6 else ""))

    graft_pref = tuple(f"backbone.backbone.layers.{i}.mixer." for i in replaced)

    def _depth_stack(keys):
        return {k for k in keys
                if ".depth_tf." in k or ".gru." in k or ".depth_ssm." in k}
    ck_stack, md_stack = _depth_stack(sd), _depth_stack(model_sd)
    depth_swap = bool(ck_stack) and bool(md_stack) and not (ck_stack & md_stack)
    if depth_swap:
        logger.warning("Depth head ARCHITECTURE SWAP detected (checkpoint and "
                       "model use different depth stacks) — the depth module "
                       "starts from fresh init by design; the backbone still "
                       "loads and is still guarded.")

    plan_graft = bool(model_plan_keys) and not ckpt_plan_keys

    critical_missing = [k for k in missing
                        if k.endswith(_CRITICAL_PROJ)
                        and k not in dropped
                        and not (graft_pref and k.startswith(graft_pref))
                        and not (depth_swap and k.startswith("depth_module."))
                        and not (plan_graft and _is_plan_param(k))]
    if critical_missing:
        raise RuntimeError(
            f"{len(critical_missing)} projection weight(s) received NO value from "
            f"checkpoint {path} (e.g. {critical_missing[0]}) and would train from "
            f"fresh init. Check that --backbone / --model_name / LoRA rank match "
            f"the checkpoint that produced it."
        )

    logger.info(f"Loaded model from {path} (step {checkpoint.get('step', '?')})")

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint.get("step", 0)


def validate(model, args, device):
    model.eval()

    unwrapped = model.module if isinstance(model, DDP) else model
    plan_kw = plan_dataset_kwargs(args)
    if getattr(args, "delay_pattern", False) or getattr(args, "depth_module", False):
        from delay_dataset import DelayMimiDataset, DelayCollator
        val_dataset = DelayMimiDataset(
            data_path=args.val_data_path,
            tokenizer=unwrapped.tokenizer,
            registry=unwrapped.token_registry,
            max_seq_len=args.max_seq_len,
            aligned=getattr(args, "depth_module", False),
            stage=args.dataset_stage,
            cot_mode=getattr(args, "cot_mode", "prose"),
            reasoning_weight=getattr(args, "reasoning_weight", 0.3),
            audio_continue_weight=getattr(args, "audio_continue_weight", 0.0),
            default_speaker_id=getattr(args, "default_speaker_id", None),
            **_plan_kwargs_for(DelayMimiDataset.__init__, plan_kw,
                               "DelayMimiDataset", PLAN_DATASET_DEFAULTS),
        )
        val_collator = DelayCollator(pad_token_id=unwrapped.token_registry.pad_id)
    else:
        val_dataset = MambaCoTTTSDataset(
            data_path=args.val_data_path,
            tokenizer=unwrapped.tokenizer,
            registry=unwrapped.token_registry,
            stage=args.dataset_stage,
            cot_mode=getattr(args, "cot_mode", "prose"),
            max_seq_len=args.max_seq_len,
            **_plan_kwargs_for(MambaCoTTTSDataset.__init__, plan_kw,
                               "MambaCoTTTSDataset", PLAN_DATASET_DEFAULTS),
        )
        val_collator = MambaCoTDataCollator(
            pad_token_id=unwrapped.token_registry.pad_id,
            registry=unwrapped.token_registry,
            stage=args.dataset_stage,
        )
    use_weighted_loss = STAGE_DEFAULTS[args.stage]["use_weighted_loss"]
    plan_accepted = plan_forward_params(unwrapped)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=val_collator, num_workers=2,
    )

    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            plan_batch, _ = plan_forward_kwargs(
                batch, plan_accepted, unwrapped.token_registry, device)
            if "codes" in batch:
                speaker_ids = batch.get("speaker_ids")
                prosody_vecs = batch.get("prosody_vecs")
                outputs = unwrapped(
                    input_ids=input_ids, labels=labels,
                    speaker_ids=speaker_ids.to(device) if speaker_ids is not None else None,
                    codes=batch["codes"].to(device),
                    prosody_vecs=prosody_vecs.to(device) if prosody_vecs is not None else None,
                    semantic_weight=getattr(args, "semantic_weight", 1.0),
                    eos_weight=getattr(args, "eos_weight", 1.0),
                    pause_exit_weight=getattr(args, "pause_exit_weight", 1.0),
                    pause_codes=getattr(args, "pause_codes", None),
                    loss_weights=(batch["loss_weights"].to(device)
                                  if "loss_weights" in batch else None),
                    **plan_batch,
                )
            else:
                speech_mask = batch["speech_mask"].to(device)
                loss_weights = (batch["loss_weights"].to(device)
                                if use_weighted_loss else None)
                outputs = unwrapped(
                    input_ids=input_ids, labels=labels,
                    speech_mask=speech_mask, loss_weights=loss_weights,
                    **plan_batch,
                )
            total_loss += outputs["loss"].item()
            num_batches += 1

    return total_loss / max(num_batches, 1)


def decode_speech_tokens(token_ids: list, registry: TokenRegistry) -> list:
    codec_values = []
    for tid in token_ids:
        if registry.is_speech_token(tid):
            codec_values.append(registry.speech_id_to_value(tid))
    return codec_values


def freeze_for_mtp_healing(model):
    for param in model.parameters():
        param.requires_grad = False

    trainable_count = 0
    for name, param in model.mtp_module.named_parameters():
        param.requires_grad = True
        trainable_count += param.numel()

    total = sum(p.numel() for p in model.parameters())
    logger.info(
        f"MTP Healing: frozen all params, unfrozen MTP heads. "
        f"Trainable: {trainable_count:,} / {total:,} "
        f"({100 * trainable_count / total:.3f}%)"
    )
    return trainable_count


def mtp_healing_loop(model, loader, optimizer, scheduler, device, args, rank):
    model.train()
    data_iter = iter(loader)

    optimizer.zero_grad()
    running_mtp_loss = 0.0

    for step in range(1, args.max_steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        speech_mask = batch["speech_mask"].to(device)
        residual_codes = batch.get("residual_codes")
        if residual_codes is not None:
            residual_codes = residual_codes.to(device)

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(
                input_ids=input_ids,
                labels=labels,
                speech_mask=speech_mask,
                loss_weights=None,
                residual_codes=residual_codes,
            )

            mtp_loss = outputs["mtp_loss"]
            if mtp_loss is None or mtp_loss.item() == 0.0:
                continue

            scaled_loss = mtp_loss / args.grad_accum

        scaled_loss.backward()

        if step % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=1.0,
            )
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        running_mtp_loss += mtp_loss.item()

        if rank == 0 and step % args.log_every == 0:
            avg_mtp = running_mtp_loss / args.log_every
            lr_current = optimizer.param_groups[0]["lr"]

            logger.info(
                f"[Stage 5 MTP Healing] Step {step}/{args.max_steps} | "
                f"MTP Loss: {avg_mtp:.4f} | LR: {lr_current:.2e}"
            )
            running_mtp_loss = 0.0

        if rank == 0 and step % args.save_every == 0:
            save_checkpoint(model, optimizer, scheduler, step, args)

    if rank == 0:
        save_checkpoint(model, optimizer, scheduler, args.max_steps, args)

    logger.info(f"[Stage 5 MTP Healing] Complete ({args.max_steps} steps).")


def setup_distributed():
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        return rank, local_rank, world_size, device, True
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return 0, 0, 1, device, False


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Mamba-CoT-TTS Multi-Stage Training Pipeline"
    )

    parser.add_argument("--stage", type=int, required=True, choices=[1, 2, 3, 4, 5],
                        help="Training stage (1=codec align, 2=CoT pretrain, 3=SFT, 4=DPO, 5=MTP healing)")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to training JSONL data")

    parser.add_argument("--backbone", type=str, default="mamba",
                        choices=["mamba", "transformer"],
                        help="Architecture arm for the thesis comparison: "
                             "'mamba' (0%% attention, state-spaces/mamba2-1.3b) "
                             "or 'transformer' (100%%, EleutherAI/pythia-1.4b — "
                             "matched Pile corpus / GPT-NeoX tokenizer / d_model "
                             "2048). Pair with --model_name. See backbones.py.")
    parser.add_argument("--model_name", type=str, default="state-spaces/mamba2-1.3b",
                        help="HuggingFace model ID for Mamba-2 backbone")
    parser.add_argument("--output_dir", type=str, default="./checkpoints",
                        help="Directory for checkpoints")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--lr_speaker", type=float, default=0.0,
                        help="separate LR for the speaker embedding at stages "
                             "2-4 (0 = share the main LR). A row that is FRESH "
                             "at Stage 2 must be LEARNED, not refined, and the "
                             "fine-tuning LR cannot do that in one epoch - see "
                             "ledger S120.")
    parser.add_argument("--f1_exclude_prompts_from", nargs="*", default=None,
                        help="extra JSONL files whose `prompt` tokens are "
                             "excluded from the F1 whitelist. Pass the eval "
                             "prompt sets: a tag token that never appears in a "
                             "TRAINING prompt can still appear in an eval one, "
                             "and training it would perturb exactly the rows "
                             "the base-preservation gate is measured on.")
    parser.add_argument("--train_cot_embeddings_only", action="store_true",
                        help="WORKSTREAM F1 (the frozen-backbone floor): train "
                             "ONLY the token-embedding rows the THINK region "
                             "uses; every other parameter is frozen. Base-"
                             "domain generation is then preserved by "
                             "construction, not by passing a gate — unlike "
                             "LoRA, which could still rewrite every projection "
                             "and failed worst of all Track-A arms (S138). "
                             "Requires --cot_mode tags and --weight_decay 0.")
    parser.add_argument("--freeze_speaker", action="store_true",
                        help="Pin the speaker table (requires_grad=False) after "
                             "any --speaker_init_from copy. NOT the same as "
                             "--lr_speaker 0, which is falsy and silently puts "
                             "the table back on the main LR.")
    parser.add_argument("--speaker_init_from", type=str, default=None,
                        help="copy TRAINED speaker rows onto fresh ids before "
                             "training, as 'new:src,new:src' (e.g. "
                             "'4200:1,4201:2'). A random row is off-manifold for "
                             "a backbone trained to expect real speaker vectors; "
                             "starting from a trained voice makes the fine-tune "
                             "a refinement instead of a search.")
    parser.add_argument("--init_from", type=str, default=None,
                        help="Initialize MODEL WEIGHTS ONLY from this checkpoint "
                             "(fresh optimizer/scheduler, step 0). Use when the "
                             "module set or loss objective changed (e.g. adding "
                             "speaker input-injection) so the old optimizer state "
                             "no longer matches. New zero-init params stay zero.")

    parser.add_argument("--batch_size", type=int, default=None,
                        help="Per-GPU batch size (default: stage-specific)")
    parser.add_argument("--grad_accum", type=int, default=None,
                        help="Gradient accumulation steps (default: stage-specific)")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Total training steps (default: stage-specific)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Training seed for a seed replicate (2026-09-27): seeds python / numpy / torch "
                             "(offset by rank) AND the DistributedSampler, so the data order changes too. "
                             "Default None = the historical behaviour (no explicit seeding, sampler seed 0), "
                             "so every earlier run's command reproduces unchanged.")
    parser.add_argument("--warmup_steps", type=int, default=None,
                        help="LR warmup steps (default: stage-specific)")
    parser.add_argument("--max_seq_len", type=int, default=2048,
                        help="Maximum sequence length")
    parser.add_argument("--log_every", type=int, default=10,
                        help="Log metrics every N steps")
    parser.add_argument("--save_every", type=int, default=5000,
                        help="Save checkpoint every N steps")
    parser.add_argument("--val_every", type=int, default=1000,
                        help="Validate every N steps")

    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate (default: stage-specific)")
    parser.add_argument("--lr_new_params", type=float, default=3e-4,
                        help="Stage 1: LR for new embeddings/head/MTP "
                             "(also used for the fresh embedding/lm_head/MTP "
                             "group under --full_finetune)")
    parser.add_argument("--lr_lora", type=float, default=1e-4,
                        help="Stage 1: LR for LoRA adapters")
    parser.add_argument("--lr_backbone", type=float, default=5e-5,
                        help="Stage 1 --full_finetune: LR for the PRETRAINED "
                             "Mamba-2 SSM backbone (layers + norm_f). Kept well "
                             "below lr_new_params because full fine-tuning a "
                             "pretrained backbone at the new-param LR (3e-4) "
                             "diverges. Recommended 5e-5 (ceiling ~1e-4).")
    parser.add_argument("--lr_min", type=float, default=None,
                        help="Minimum LR for cosine decay (default: stage-specific)")

    parser.add_argument("--full_finetune", action="store_true",
                        help="Stage 1: skip LoRA and full fine-tune the whole "
                             "model (unfreeze ALL params). Pretrained SSM "
                             "backbone trains @ --lr_backbone (5e-5), the fresh "
                             "embedding/lm_head/MTP @ --lr_new_params (3e-4). "
                             "When absent, behavior is byte-identical to the "
                             "LoRA path. (Track B: test whether limited adapter "
                             "capacity, not the codec, was the Stage-1 bottleneck.)")
    parser.add_argument("--fullft_bf16_optim", action="store_true",
                        help="--full_finetune only: keep params/optimizer in "
                             "bf16 instead of upcasting to fp32 master weights. "
                             "Lower memory, but bf16 second-moment accumulation "
                             "over 1.7B params can confound convergence. Default "
                             "(off) = fp32 master weights for stability.")
    parser.add_argument("--use_lora", action="store_true",
                        help="Force LoRA in stages >1 (default: only stage 1 uses "
                             "LoRA; stages 2+ full-finetune). Phase A': LoRA in "
                             "stage 2 freezes the backbone so the Stage-1 cb0 "
                             "articulation survives the small-corpus CoT fine-tune "
                             "(full-FT on ~2.9k utts destroyed cb0 — job 14733469).")
    parser.add_argument("--lora_rank", type=int, default=64,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=64,
                        help="LoRA alpha (scaling = alpha/rank)")
    parser.add_argument("--lora_target_modules", nargs="+",
                        default=["in_proj", "out_proj"],
                        help="Module names to apply LoRA (Mamba-2 SSM projections)")
    parser.add_argument("--num_crh_heads", type=int, default=0,
                        help="Codebook-residual heads. 0 = X-Codec2 prototype "
                             "(disabled). Set to (codec.num_codebooks - 1) "
                             "for multi-codebook codecs: e.g. 7 for Mimi-8.")
    parser.add_argument("--crh_codebook_size", type=int, default=2048,
                        help="Per-codebook size for CRH targets. Mimi: 2048.")
    parser.add_argument("--speaker_conditioning", action="store_true",
                        help="Enable AdaLN speaker conditioning (v1, post-backbone; "
                             "zero-init so it starts as identity). Requires "
                             "speaker_id in the data (scripts/build_speaker_map.py).")
    parser.add_argument("--num_speakers", type=int, default=4000,
                        help="SpeakerEncoder table size; must exceed the highest "
                             "speaker_id in the data (LibriTTS-R clean+other ~2.5K)")
    parser.add_argument("--allow_speaker_table_reinit", action="store_true",
                        help="Let a weights-only warm start DROP a shape-mismatched "
                             "speaker table (every voice re-initialised). Off by "
                             "default since 2026-09-21; the sanctioned way to grow "
                             "the table is scripts/expand_speaker_table.py.")
    parser.add_argument("--speaker_dim", type=int, default=256)
    parser.add_argument("--prosody_vec_injection", action="store_true",
                        help="PB-12: condition the delay-pattern renderer on a "
                             "6-dim mechanical prosody vector (rows' "
                             "prosody_vec field from scripts/"
                             "prep_prosody_vectors.py), injected zero-init "
                             "through the SSM input channel.")
    parser.add_argument("--speaker_input_injection", action="store_true",
                        help="Also inject the speaker embedding into the backbone "
                             "INPUT (carried through the SSM recurrence) for "
                             "persistent frame-to-frame timbre; fixes the post-"
                             "backbone AdaLN's within-utterance voice drift. "
                             "Zero-init -> exact resume from a non-injection ckpt.")
    parser.add_argument("--semantic_weight", type=float, default=1.0,
                        help="Loss upweight on level-0 (semantic) positions of a "
                             "flattened multi-codebook stream; needs --level_cycle")
    parser.add_argument("--weight_decay", type=float, default=0.1,
                        help="AdamW weight decay. fish-speech uses 0.0; 0.1 is "
                             "this project's historical default.")
    parser.add_argument("--adam_eps", type=float, default=1e-8,
                        help="AdamW eps. fish-speech uses 1e-5.")
    parser.add_argument("--depth_arch", choices=["gru", "mamba2", "transformer"],
                        default="gru",
                        help="depth-head architecture over the codebook axis. "
                             "'transformer' matches the reference "
                             "(fish-speech DualAR uses n_fast_layer=4 "
                             "TRANSFORMER layers); 'gru' is the original.")
    parser.add_argument("--depth_level_decay", type=float, default=1.0,
                        help="geometric decay of the per-codebook depth loss "
                             "weight, w(k)=decay^(k-1) for k>=1 (Fish S2 eq.3 "
                             "puts capacity on the coarse acoustic codebooks). "
                             "1.0 = the old flat weighting.")
    parser.add_argument("--depth_cond", choices=["add", "prefix"], default="add",
                        help="how the backbone hidden state conditions the depth "
                             "stack. 'prefix' = Fish S2 §2.2 (h_slow as a TOKEN "
                             "at depth position 0, one SHARED codebook embedding "
                             "table, explicit level identity). 'add' = ours "
                             "(h broadcast onto every step, per-level tables).")
    parser.add_argument("--depth_heads", type=int, default=8,
                        help="attention heads in the transformer depth head.")
    parser.add_argument("--audio_continue_weight", type=float, default=0.0,
                        help="Supervise the LM head across the AUDIO span at this "
                             "weight instead of IGNORE_INDEX (ledger S79). The "
                             "stop head is queried every audio frame at inference "
                             "but, at 0.0, is trained at none of them — p(stop) "
                             "then drifts to 15-59x its base rate and generation "
                             "halts at the first comma. ~0.1 supplies the missing "
                             "'continue' evidence without swamping the text CE. "
                             "NB: non-zero SHIFTS absolute val loss, so it "
                             "re-baselines any cross-run comparison.")
    parser.add_argument("--pause_exit_weight", type=float, default=1.0,
                        help="S200 remedy (backtrack experiment): multiply the cb0 "
                             "depth-CE on frames whose target is a SPEECH code and "
                             "whose predecessor was a PAUSE code - the pause->speech "
                             "transition the collapsed head never makes. Weighted "
                             "mean normalised by the weights (eos_weight convention); "
                             "1.0 = off, bit-identical. Needs --pause_codes_file.")
    parser.add_argument("--pause_codes_file", default="",
                        help="JSON with a `codes` list (or a bare list) of cb0 pause "
                             "codes, e.g. data/silence_codes_fish_s2.json.")
    parser.add_argument("--eos_weight", type=float, default=1.0,
                        help="Loss upweight on </AUDIO>/<EOS> positions "
                             "(stopping/alignment supervision)")
    parser.add_argument("--level_cycle", type=int, default=0,
                        help="Codebook levels per frame in the flattened stream "
                             "(8 for mimi8); used by --semantic_weight")
    parser.add_argument("--cfg_dropout", type=float, default=0.0,
                        help="Classifier-free-guidance prompt dropout probability "
                             "(e.g. 0.1): drop the text condition so an "
                             "unconditional stream is learned for guided decoding")
    parser.add_argument("--delay_pattern", action="store_true",
                        help="Arm B: MusicGen-style delay multi-codebook path — "
                             "one position per frame, K summed level embeddings "
                             "+ K heads. Consumes *_mimi (multi-codebook) JSONLs "
                             "directly; --semantic_weight applies to level 0.")
    parser.add_argument("--depth_module", action="store_true",
                        help="PB-14 Lever 3: RQ-style recurrent depth module — "
                             "frame-aligned grid (no stagger); a GRU generates "
                             "cb0..7 sequentially per frame from the backbone "
                             "hidden state (explicit intra-frame conditioning). "
                             "Consumes the same *_mimi JSONLs as --delay_pattern.")
    parser.add_argument("--depth_dim", type=int, default=1024,
                        help="depth module width")
    parser.add_argument("--depth_layers", type=int, default=2,
                        help="depth module GRU layers")
    parser.add_argument("--depth_feedback", choices=["all", "semantic"],
                        default="all",
                        help="what the TEMPORAL stream sees of each frame: all "
                             "8 levels (Moshi-style) or cb0 only (decode-time "
                             "acoustic sampling cannot contaminate the "
                             "recurrence)")
    parser.add_argument("--depth_cross_frame", type=int, default=0,
                        help="S30 fan fix: number of PREVIOUS-frame coarse "
                             "acoustic levels (cb1..cbN) whose codes condition "
                             "this frame's depth unroll — the cross-FRAME "
                             "channel the per-frame GRU reset lacks. The fan is "
                             "carried by cb1..3 (jitter probe), so 3 is the "
                             "recommended value. 0 = legacy per-frame behavior.")
    parser.add_argument("--codec_num_levels", type=int, default=8,
                        help="number of codec levels for the delay/depth paths "
                             "(Mimi: 8; Fish S2: 10)")
    parser.add_argument("--codebook_sizes", type=str, default=None,
                        help="comma-separated per-level vocab sizes for "
                             "heterogeneous codecs (Fish S2: "
                             "'4096,1024,1024,1024,1024,1024,1024,1024,1024,1024'). "
                             "Default None = uniform 2048 (Mimi)")
    parser.add_argument("--freeze_depth_module", action="store_true",
                        help="S25 fan fix: freeze the depth module during "
                             "Stage-2 fine-tuning so the init checkpoint's "
                             "acoustic rendering (e.g. dial-108's clean "
                             "flatmod) survives unchanged — the CoT/tags "
                             "conditioning trains into the backbone only")
    parser.add_argument("--flow_head", action="store_true",
                        help="PB-14 option A: attach a continuous flow-matching "
                             "head that predicts the Mimi RVQ-recon latent "
                             "(dim 512) per frame — decoded through the FROZEN "
                             "Mimi vocoder, no argmax. Additive to --depth_module "
                             "(discrete cb0 kept for WER/stop/CoT); loads the "
                             "codec to compute fp32 latent targets on the fly.")
    parser.add_argument("--flow_hidden", type=int, default=1024,
                        help="flow head MLP width")
    parser.add_argument("--flow_layers", type=int, default=4,
                        help="flow head residual MLP blocks")
    parser.add_argument("--flow_weight", type=float, default=1.0,
                        help="weight of the flow loss in the total")
    parser.add_argument("--hybrid_attention_top_k", type=int, default=0,
                        help="Phase 8: replace the top-K Mamba blocks' mixers "
                             "with causal self-attention + RoPE (0 = pure Mamba). "
                             "Fresh-init; on a pure-ckpt warm start the replaced "
                             "layers' Mamba weights are dropped. The co-equal "
                             "comparison arm uses K=4.")
    parser.add_argument("--flow_only", action="store_true",
                        help="freeze everything except the flow head — the cheap, "
                             "safe A-frozen first pass (discrete cb0/WER untouched; "
                             "only ~15M flow params train). Drop it for a full-FT "
                             "run where the backbone can also adapt to the head.")
    parser.add_argument("--mtp_num_heads", type=int, default=3,
                        help="Shared-weight speculative MTP head offsets trained "
                             "as an auxiliary loss. Default 3 (backward compat). "
                             "Set 0 to disable MTP entirely — recommended for "
                             "Stage-1 codec alignment so all LoRA capacity goes to "
                             "the next-token (main) objective that generation uses. "
                             "MTP is restored/trained in Stage-5 healing.")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout rate")

    parser.add_argument("--dpo_beta", type=float, default=0.5,
                        help="DPO temperature coefficient")
    parser.add_argument("--dpo_alpha", type=float, default=0.01,
                        help="DPO entropy regularization weight")
    parser.add_argument("--dpo_kl_ceiling", type=float, default=5.0,
                        help="DPO KL divergence ceiling")
    parser.add_argument("--ref_model_path", type=str, default=None,
                        help="Stage 4: path to reference model checkpoint")

    parser.add_argument("--l2sp_weight", type=float, default=0.0,
                        help="PB-04 (c): L2-SP anchor weight — quadratic pull "
                             "of backbone params toward their --init_from "
                             "snapshot (cb0-articulation preservation). "
                             "Typical 1e-4..1e-3; 0 = off.")
    parser.add_argument("--cot_mode", default=None,
                        choices=list(LEGACY_COT_MODES) + list(PLAN_COT_MODES),
                        help="form of the <THINK> block: the full delivery "
                             "prose, EMO/PACE/PITCH tags only, or no reasoning "
                             "(default: prose); plus the P²-CoT plan-bearing "
                             "forms plan / tags+plan / full, which are the same "
                             "thing --plan_tier selects. All read the SAME "
                             "corpus file, so the arms differ only in this.")

    parser.add_argument("--plan_tier", default="none", choices=list(PLAN_TIERS),
                        help="which CoT tiers the <THINK> block carries — the "
                             "C0-C3 factorial as one flag: none (C0, today's "
                             "behaviour), plan (C1, prosodic program only), "
                             "tags+plan (C2), full (C3, prose+tags+plan). "
                             "Sets --cot_mode to the same value; anything but "
                             "'none' needs MVC_ENABLE_PLAN_TOKENS=1.")
    parser.add_argument("--plan_injection", action="store_true",
                        help="consume the plan during the AUDIO block via "
                             "duration-scheduled input injection (§3.2) — the "
                             "attention-free analogue of RALL-E's "
                             "duration-guided masking. Adds a zero-init "
                             "projection, so enabling it on a fresh model is an "
                             "exact no-op until trained.")
    parser.add_argument("--plan_scaffold_dropout", type=float, default=0.3,
                        help="probability of dropping the injection for a whole "
                             "utterance during training (§3.2). Keeps a "
                             "prefix-only consumption path alive, so "
                             "injection-OFF stays a valid MEASUREMENT condition "
                             "(risk R3) instead of the scaffold being the only "
                             "way the plan is read. Inert without "
                             "--plan_injection.")
    parser.add_argument("--splice_plan_vocab", action="store_true",
                        help="on --init_from ONLY: remap a pre-P²-CoT "
                             "checkpoint's embedding/lm_head rows into the "
                             "plan-enabled vocabulary (rows below the plan "
                             "block keep their id, rows above shift up by "
                             "|plan block|, the plan rows keep their fresh "
                             "init). REQUIRED for any warm start under "
                             "MVC_ENABLE_PLAN_TOKENS=1; without it the "
                             "vocab-sized mismatch is a hard error rather than "
                             "a silent re-init of the whole embedding.")
    parser.add_argument("--plan_loss_weight", type=float, default=1.0,
                        help="text-CE weight (λ) on the <PLAN> region "
                             "(the dataset's `plan_weight`). 1.0 = the speech "
                             "weight: the model must AUTHOR the program at "
                             "inference, unlike Tier-A prose at "
                             "--reasoning_weight 0.3.")
    parser.add_argument("--plan_tag_weight", type=float, default=None,
                        help="text-CE weight (λ) on the Tier-B1 EMO/PACE/PITCH "
                             "tags (the dataset's `tag_weight`). Unset = "
                             "inherit --reasoning_weight, which is exactly what "
                             "the 9-arm `tags` runs did, so legacy arms are "
                             "unchanged. The master plan §3.1 specifies 1.0 for "
                             "the tag-bearing plan tiers (C2 tags+plan, C3 "
                             "full) — pass it explicitly there, or those tags "
                             "train at 0.3 with nothing in the log to say so.")
    parser.add_argument("--min_plan_coverage", type=float, default=0.0,
                        help="minimum fraction of corpus rows that must carry a "
                             "usable plan, enforced by the dataset's "
                             "construction-time audit. 0 = report only (today's "
                             "behaviour); the master plan's data gate is 0.95.")
    parser.add_argument("--plan_schedule_from", default="gt",
                        choices=["gt", "self"],
                        help="whose durations schedule the injection during "
                             "training: 'gt' teacher-forces the aligned corpus "
                             "plan (Stage A); 'self' anneals onto the model's "
                             "own emitted plan (Stage-B scheduled sampling "
                             "against exposure bias).")
    parser.add_argument("--style_tier", default="none", choices=["none", "gt"],
                        help="utterance-level style tokens (§3.3): 'gt' puts "
                             "the row's own VQ style tokens in the prefix (the "
                             "GST trick). Requires MVC_NUM_STYLE_TOKENS>0.")
    parser.add_argument("--style_dropout", type=float, default=0.1,
                        help="conditioning-dropout probability on the style "
                             "tokens, so the 'none' inference mode (CoT does "
                             "all the work) stays trained. Inert while "
                             "--style_tier is none.")
    parser.add_argument("--dataset_stage",
                        type=lambda s: s if s == "auto" else int(s),
                        default=None, choices=[1, 2, "auto"],
                        help="Override the sequence-assembly format independent of "
                             "the training stage. E.g. '--stage 2 --dataset_stage 1' "
                             "trains with Stage-2 hyperparameters on reasoning-free "
                             "(Stage-1 format) sequences — the no-CoT control arm, "
                             "differing from the CoT arm ONLY in data.")
    parser.add_argument("--reasoning_weight", type=float, default=0.3,
                        help="Depth Stage-2 CoT: text-CE weight (λ_r) on the "
                             "<THINK> reasoning region (1.0 = speech).")
    parser.add_argument("--default_speaker_id", type=int, default=None,
                        help="Speaker id for rows lacking one (the CoT corpus) — "
                             "holds the voice constant across CoT/no-CoT arms.")
    parser.add_argument("--val_data_path", type=str, default=None,
                        help="Path to validation JSONL data")

    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level")

    args = parser.parse_args()
    args.pause_codes = None
    if getattr(args, "pause_exit_weight", 1.0) != 1.0:
        if not getattr(args, "pause_codes_file", ""):
            raise SystemExit("--pause_exit_weight != 1.0 needs --pause_codes_file "
                             "(a weight with no code set would be a silent no-op)")
        with open(args.pause_codes_file, encoding="utf-8") as _pf:
            _pc = json.load(_pf)
        _pc = _pc["codes"] if isinstance(_pc, dict) else _pc
        args.pause_codes = torch.tensor(sorted(set(int(c) for c in _pc)), dtype=torch.long)
        print(f"[pause-exit] weight {args.pause_exit_weight} on pause->speech cb0 "
              f"transitions; {args.pause_codes.numel()} pause codes from "
              f"{args.pause_codes_file}", flush=True)

    defaults = STAGE_DEFAULTS[args.stage]
    if args.batch_size is None:
        args.batch_size = defaults["batch_size"]
    if args.grad_accum is None:
        args.grad_accum = defaults["grad_accum"]
    if args.max_steps is None:
        args.max_steps = defaults["max_steps"]
    if args.warmup_steps is None:
        args.warmup_steps = defaults["warmup_steps"]
    if args.lr is None:
        args.lr = defaults["lr"]
    if args.lr_min is None:
        args.lr_min = defaults["lr_min"]
    if args.dataset_stage is None:
        args.dataset_stage = defaults["dataset_stage"]

    _resolve_plan_args(args)
    return args


def _resolve_plan_args(args):
    tier = getattr(args, "plan_tier", "none")

    if tier != "none":
        if args.cot_mode is not None and args.cot_mode != tier:
            raise SystemExit(
                f"--plan_tier {tier} and --cot_mode {args.cot_mode} name "
                f"different conditions. They select the same thing: pass one, "
                f"or pass both with the same value.")
        args.cot_mode = tier
    elif args.cot_mode in PLAN_COT_MODES:
        args.plan_tier = tier = args.cot_mode
    if args.cot_mode is None:
        args.cot_mode = "prose"

    plan_on = tier != "none" or getattr(args, "plan_injection", False)
    if plan_on and not config.ENABLE_PLAN_TOKENS:
        raise SystemExit(
            f"--plan_tier {tier}"
            f"{' --plan_injection' if getattr(args, 'plan_injection', False) else ''} "
            f"needs the plan vocabulary, but MVC_ENABLE_PLAN_TOKENS is off. "
            f"Export MVC_ENABLE_PLAN_TOKENS=1 for this process (and the same "
            f"MVC_NUM_STYLE_TOKENS as the checkpoint) before launching.")
    if getattr(args, "plan_injection", False) and tier == "none":
        raise SystemExit(
            "--plan_injection with --plan_tier none: there is no plan in the "
            "sequence to inject, so the module would train on nothing. Pass a "
            "tier (plan / tags+plan / full).")
    if plan_on and getattr(args, "stage", 1) in (4, 5):
        raise SystemExit(
            f"the plan tier is not wired into the stage-{args.stage} loop "
            f"({'DPO' if args.stage == 4 else 'MTP healing'} forwards no plan "
            f"channel), so it would be silently absent from training. Plan arms "
            f"run in stages 1-3.")
    d = float(getattr(args, "plan_scaffold_dropout", 0.0))
    if not 0.0 <= d < 1.0:
        raise SystemExit(f"--plan_scaffold_dropout must be in [0, 1), got {d} "
                         f"(1.0 would drop the injection on every utterance)")
    if getattr(args, "style_tier", "none") != "none" and config.NUM_STYLE_TOKENS <= 0:
        raise SystemExit(
            f"--style_tier {args.style_tier} needs MVC_NUM_STYLE_TOKENS>0 "
            f"(currently {config.NUM_STYLE_TOKENS}) — there is no style "
            f"vocabulary to emit.")
    s = float(getattr(args, "style_dropout", 0.0))
    if not 0.0 <= s < 1.0:
        raise SystemExit(f"--style_dropout must be in [0, 1), got {s}")
    return args


def main():
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    rank, local_rank, world_size, device, is_ddp = setup_distributed()

    if args.seed is not None:
        import numpy as _np
        random.seed(args.seed + rank)
        _np.random.seed((args.seed + rank) % (2 ** 32))
        torch.manual_seed(args.seed + rank)
        torch.cuda.manual_seed_all(args.seed + rank)
        if rank == 0:
            logger.info(f"Seed: {args.seed} (python/numpy/torch offset by rank; sampler seed {args.seed})")

    if rank == 0:
        logger.info(f"=== Mamba-CoT-TTS Training Stage {args.stage} ===")
        logger.info(f"Device: {device} | World size: {world_size}")
        logger.info(f"Data: {args.data_path}")
        logger.info(f"Steps: {args.max_steps} | Batch: {args.batch_size} | "
                     f"Grad accum: {args.grad_accum}")
        os.makedirs(args.output_dir, exist_ok=True)

    logger.info("Loading model...")
    model = MambaCoTModel(
        model_name=args.model_name,
        device=str(device),
        dtype=torch.bfloat16,
        mtp_num_heads=args.mtp_num_heads,
        num_crh_heads=args.num_crh_heads,
        crh_codebook_size=args.crh_codebook_size,
        backbone_kind=args.backbone,
    )

    if args.speaker_conditioning:
        model.enable_speaker_conditioning(
            speaker_dim=args.speaker_dim, num_speakers=args.num_speakers,
            input_injection=args.speaker_input_injection,
        )
        if rank == 0:
            logger.info(f"Speaker conditioning enabled (AdaLN v1"
                        f"{'+input-injection' if args.speaker_input_injection else ''}, "
                        f"num_speakers={args.num_speakers}, dim={args.speaker_dim})")

    if getattr(args, "prosody_vec_injection", False):
        model.enable_prosody_conditioning()
        if rank == 0:
            logger.info("Prosody-vector input injection enabled (PB-12; "
                        "zero-init — exact no-op until trained; rows without "
                        "prosody_vec condition on the corpus mean)")

    if (args.stage == 1 or args.use_lora) and not args.full_finetune:
        lora_count = apply_lora(
            model, rank=args.lora_rank, alpha=args.lora_alpha,
            target_modules=args.lora_target_modules, dropout=args.lora_dropout,
        )
        if rank == 0:
            total, trainable = count_parameters(model)
            logger.info(f"LoRA applied ({lora_count} modules). "
                        f"Trainable: {trainable:,} / {total:,} "
                        f"({100*trainable/total:.1f}%)")
    elif args.stage == 1 and args.full_finetune:
        for param in model.parameters():
            param.requires_grad = True

        if not args.fullft_bf16_optim:
            for param in model.parameters():
                if param.dtype == torch.bfloat16:
                    param.data = param.data.float()
            if rank == 0:
                logger.info("Full fine-tune: upcast params to fp32 "
                            "(fp32 master weights; bf16 compute via autocast).")

        total, trainable = count_parameters(model)
        if rank == 0:
            logger.info(f"Full fine-tune: ALL params unfrozen. "
                        f"Trainable: {trainable:,} / {total:,} "
                        f"({100*trainable/total:.1f}%)")
        assert trainable == total, (
            f"--full_finetune expected all params trainable, "
            f"got {trainable:,}/{total:,}"
        )
    elif args.stage >= 2 and not args.use_lora:
        for param in model.parameters():
            param.requires_grad = True
        if not args.fullft_bf16_optim:
            for param in model.parameters():
                if param.dtype == torch.bfloat16:
                    param.data = param.data.float()
            if rank == 0:
                logger.info("Stage>=2 full-param: upcast params to fp32 "
                            "(fp32 master weights + fp32 Adam states; "
                            "bf16 compute via autocast).")
        total, trainable = count_parameters(model)
        assert trainable == total, (
            f"stage>=2 full-param expected all params trainable, "
            f"got {trainable:,}/{total:,}"
        )
        if rank == 0:
            logger.info(f"Stage>=2 full-param: Trainable: {trainable:,} / "
                        f"{total:,} (100.0%)")

    if args.delay_pattern and args.depth_module:
        raise SystemExit("--delay_pattern and --depth_module are mutually exclusive")
    _cb_sizes = ([int(s) for s in args.codebook_sizes.split(",")]
                 if args.codebook_sizes else None)
    if _cb_sizes is not None and len(_cb_sizes) != args.codec_num_levels:
        raise SystemExit(f"--codebook_sizes has {len(_cb_sizes)} entries but "
                         f"--codec_num_levels={args.codec_num_levels}")
    if args.delay_pattern:
        model.enable_delay_pattern(num_levels=args.codec_num_levels,
                                   codebook_size=2048, codebook_sizes=_cb_sizes)
        if rank == 0:
            n = sum(p.numel() for p in model.delay_heads.parameters())
            n += sum(p.numel()
                     for p in model.backbone.backbone.embedding.level_embeds.parameters())
            logger.info(f"Delay pattern enabled (8 levels; +{n:,} new params)")
    if args.depth_module:
        model.enable_depth_module(num_levels=args.codec_num_levels,
                                  codebook_size=2048,
                                  d_depth=args.depth_dim,
                                  depth_layers=args.depth_layers,
                                  depth_feedback=args.depth_feedback,
                depth_arch=args.depth_arch, depth_heads=args.depth_heads,
                level_decay=args.depth_level_decay,
                depth_cond=args.depth_cond,
                                  cross_frame=args.depth_cross_frame,
                                  codebook_sizes=_cb_sizes)
        if rank == 0:
            n = sum(p.numel() for p in model.depth_module.parameters())
            n += sum(p.numel()
                     for p in model.backbone.backbone.embedding.level_embeds.parameters())
            _arch = getattr(args, "depth_arch", "gru").upper()
            logger.info(f"Depth module enabled (d={args.depth_dim} x "
                        f"{args.depth_layers} {_arch}, "
                        f"feedback={args.depth_feedback}, "
                        f"cross_frame={args.depth_cross_frame}; "
                        f"+{n:,} new params)")

    if getattr(args, "plan_injection", False):
        enable = getattr(model, "enable_plan_injection", None)
        if enable is None:
            raise SystemExit(
                "--plan_injection needs model.enable_plan_injection "
                "(P²-CoT B1). This build of model.py does not have it, so the "
                "flag would be accepted and do nothing.")
        enable(**_plan_kwargs_for(
            enable,
            {"scaffold_dropout": float(args.plan_scaffold_dropout),
             "schedule_from": args.plan_schedule_from},
            "model.enable_plan_injection",
            defaults={"schedule_from": "gt"}))

        plan_params = [(n, p) for n, p in model.named_parameters()
                       if _is_plan_param(n)]
        if not plan_params:
            raise RuntimeError(
                "model.enable_plan_injection created no parameter whose name "
                "starts with 'plan_'. That prefix is the wiring contract: the "
                "optimizer's fresh-params group, the checkpoint guards and this "
                "assertion all key off it, so a differently-named module would "
                "silently sit outside every one of them. Rename the module "
                "attribute (e.g. self.plan_input_proj).")
        frozen = [n for n, p in plan_params if not p.requires_grad]
        if frozen and not getattr(args, "flow_only", False):
            raise RuntimeError(
                f"{len(frozen)} plan tensor(s) are frozen right after enabling "
                f"(e.g. {frozen[0]}) — they would never train. This is the "
                f"apply_lora ordering trap: the module must be created AFTER "
                f"the backbone freeze.")
        if rank == 0:
            npl = sum(p.numel() for _, p in plan_params)
            logger.info(
                f"Plan injection enabled (duration-scheduled input injection; "
                f"scaffold_dropout={args.plan_scaffold_dropout}, "
                f"schedule_from={args.plan_schedule_from}; "
                f"+{npl:,} params in {len(plan_params)} tensors, zero-init "
                f"=> exact no-op until trained)")

    if getattr(args, "flow_head", False):
        if not args.depth_module:
            raise SystemExit("--flow_head requires --depth_module")
        from codec import load_codec
        args._flow_codec = load_codec("mimi", device=device)
        model.enable_flow_head(latent_dim=512, cb0_size=2048,
                               d_hidden=args.flow_hidden,
                               n_layers=args.flow_layers,
                               flow_weight=args.flow_weight)
        if rank == 0:
            nf = sum(p.numel() for p in model.flow_head.parameters())
            logger.info(f"Flow head enabled (dim=512, hidden={args.flow_hidden} x "
                        f"{args.flow_layers}, weight={args.flow_weight}; "
                        f"+{nf:,} new params; Mimi decoder FROZEN)")
        if getattr(args, "flow_only", False):
            for n, p in model.named_parameters():
                p.requires_grad = n.startswith("flow_head")
            if rank == 0:
                nt = sum(p.numel() for p in model.parameters() if p.requires_grad)
                logger.info(f"flow_only: froze all but flow_head ({nt:,} trainable)")

    if (getattr(args, "freeze_depth_module", False)
            and getattr(model, "depth_module", None) is not None):
        for p in model.depth_module.parameters():
            p.requires_grad = False
        if rank == 0:
            nf = sum(p.numel() for p in model.depth_module.parameters())
            logger.info(f"freeze_depth_module: {nf:,} depth params FROZEN "
                        f"(acoustic rendering pinned to the init checkpoint)")

    if getattr(args, "hybrid_attention_top_k", 0) > 0:
        replaced = model.enable_hybrid_attention(
            top_k=args.hybrid_attention_top_k)
        if rank == 0:
            na = sum(p.numel()
                     for i in replaced
                     for p in model.backbone.backbone.layers[i].mixer.parameters())
            logger.info(f"Hybrid attention enabled: layers {replaced} -> causal "
                        f"MHA+RoPE ({na:,} fresh params)")

    if args.stage == 5:
        unwrapped_model = model.module if hasattr(model, 'module') else model
        freeze_for_mtp_healing(unwrapped_model)

    ref_model = None
    if args.stage == 4:
        if args.ref_model_path is None:
            logger.error("Stage 4 requires --ref_model_path for reference policy")
            sys.exit(1)
        logger.info(f"Loading reference model from {args.ref_model_path}...")
        ref_model = MambaCoTModel(model_name=args.model_name, device=str(device), dtype=torch.bfloat16)
        load_checkpoint(args.ref_model_path, ref_model, device=str(device))
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False

    if rank == 0:
        logger.info("CoT CONDITION: cot_mode=%s | plan_tier=%s | dataset_stage=%s "
                    "— the <THINK> block carries what cot_mode says, not what "
                    "plan_tier says",
                    getattr(args, "cot_mode", None),
                    getattr(args, "plan_tier", "none"),
                    getattr(args, "dataset_stage", None))

    if rank == 0:
        total, trainable = count_parameters(model)
        logger.info(f"Parameters — Total: {total:,} | Trainable: {trainable:,}")

    dataset_stage = args.dataset_stage

    if args.stage == 4:
        dataset = DPODataset(
            data_path=args.data_path,
            tokenizer=model.tokenizer,
            registry=model.token_registry,
            max_seq_len=args.max_seq_len,
        )
        collator = DPODataCollator(pad_token_id=model.token_registry.pad_id)
    else:
        plan_kw = plan_dataset_kwargs(args)
        if args.delay_pattern or args.depth_module:
            from delay_dataset import DelayMimiDataset, DelayCollator
            dataset = DelayMimiDataset(
                data_path=args.data_path,
                cot_mode=getattr(args, "cot_mode", "prose"),
                tokenizer=model.tokenizer,
                registry=model.token_registry,
                max_seq_len=args.max_seq_len,
                cfg_dropout=args.cfg_dropout,
                aligned=args.depth_module,
                stage=dataset_stage,
                reasoning_weight=getattr(args, "reasoning_weight", 0.3),
                audio_continue_weight=getattr(args, "audio_continue_weight", 0.0),
                default_speaker_id=getattr(args, "default_speaker_id", None),
                **_plan_kwargs_for(DelayMimiDataset.__init__, plan_kw,
                                   "DelayMimiDataset", PLAN_DATASET_DEFAULTS),
            )
            collator = DelayCollator(pad_token_id=model.token_registry.pad_id)
        else:
            dataset = MambaCoTTTSDataset(
                data_path=args.data_path,
                tokenizer=model.tokenizer,
                registry=model.token_registry,
                stage=dataset_stage,
                cot_mode=getattr(args, "cot_mode", "prose"),
                max_seq_len=args.max_seq_len,
                cfg_dropout=args.cfg_dropout,
                **_plan_kwargs_for(MambaCoTTTSDataset.__init__, plan_kw,
                                   "MambaCoTTTSDataset", PLAN_DATASET_DEFAULTS),
            )
            collator = MambaCoTDataCollator(
                pad_token_id=model.token_registry.pad_id,
                registry=model.token_registry,
                stage=dataset_stage,
                semantic_weight=args.semantic_weight,
                eos_weight=args.eos_weight,
                level_cycle=args.level_cycle,
            )

    sampler = (DistributedSampler(dataset, seed=args.seed) if args.seed is not None
               else DistributedSampler(dataset)) if is_ddp else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        collate_fn=collator,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    if rank == 0:
        logger.info(f"Dataset: {len(dataset)} samples | "
                     f"Loader: {len(loader)} batches/epoch")

    if getattr(args, "plan_tier", "none") != "none":
        log_plan_stats(dataset, model.token_registry, args, rank=rank,
                       is_ddp=is_ddp)

    if getattr(args, "freeze_speaker", False):
        enc = getattr(model, "speaker_encoder", None)
        if enc is None:
            raise RuntimeError("--freeze_speaker needs --speaker_conditioning")
        for p in enc.parameters():
            p.requires_grad = False

    if getattr(args, "train_cot_embeddings_only", False):
        emb = model.backbone.backbone.embedding
        W = emb.weight
        if float(getattr(args, "weight_decay", 0.0)) != 0.0:
            raise RuntimeError(
                f"--train_cot_embeddings_only requires --weight_decay 0 "
                f"(got {args.weight_decay}): AdamW decays every element "
                f"regardless of gradient, which would move the frozen rows.")
        keep = _cot_embedding_rows(args.data_path, model, args)
        for p in model.parameters():
            p.requires_grad = False
        W.requires_grad = True
        mask = torch.zeros(W.shape[0], 1, dtype=W.dtype, device=W.device)
        mask[sorted(keep)] = 1.0
        W.register_hook(lambda g, _m=mask: g * _m)
        model._f1_rows = sorted(keep)
        if rank == 0:
            logger.info(
                f"F1: backbone FROZEN; training {len(keep)} embedding rows of "
                f"{W.shape[0]} ({100.0 * len(keep) / W.shape[0]:.2f}%) — the "
                f"tokens the THINK region uses")

    if is_ddp:
        model = DDP(model, device_ids=[local_rank])

    unwrapped = model.module if is_ddp else model
    optimizer = build_optimizer(unwrapped, args)
    scheduler = build_scheduler(optimizer, args)

    start_step = 0
    if args.resume_from:
        start_step = load_checkpoint(
            args.resume_from, unwrapped, optimizer, scheduler, device=str(device)
        )
        if rank == 0:
            logger.info(
                f"Resumed from {args.resume_from} at step {start_step} "
                f"(optimizer + scheduler state restored)"
            )
    elif args.init_from:
        load_checkpoint(args.init_from, unwrapped, device=str(device),
                        splice_plan_vocab=bool(getattr(args, "splice_plan_vocab",
                                                       False)),
                        allow_speaker_table_reinit=bool(
                            getattr(args, "allow_speaker_table_reinit", False)))
        if rank == 0:
            logger.info(f"Initialized model weights from {args.init_from} "
                        f"(fresh optimizer/scheduler, step 0)")

    if args.speaker_init_from:
        enc = getattr(unwrapped, "speaker_encoder", None)
        if enc is None:
            raise RuntimeError("--speaker_init_from needs --speaker_conditioning")
        W = enc.embedding.weight
        with torch.no_grad():
            for pair in args.speaker_init_from.split(","):
                dst, src = (int(x) for x in pair.strip().split(":"))
                if not (0 <= dst < W.shape[0] and 0 <= src < W.shape[0]):
                    raise ValueError(f"speaker row out of range in '{pair}' "
                                     f"(table is {W.shape[0]})")
                before = float(W[dst].norm())
                W[dst].copy_(W[src])
                if rank == 0:
                    logger.info(f"speaker warm-start: row {dst} <- row {src} "
                                f"(norm {before:.4f} -> {float(W[dst].norm()):.4f})")

    if getattr(args, "train_cot_embeddings_only", False):
        W = unwrapped.backbone.backbone.embedding.weight
        live = [n for n, p in unwrapped.named_parameters()
                if p.requires_grad and p is not W]
        if live:
            raise RuntimeError(
                f"F1: {len(live)} params outside the embedding are still "
                f"trainable, e.g. {live[:5]}")
        in_opt = [p for g in optimizer.param_groups for p in g["params"]]
        if len(in_opt) != 1 or in_opt[0] is not W:
            raise RuntimeError(
                f"F1: the optimizer holds {len(in_opt)} tensors; it must hold "
                f"exactly the embedding weight")
        if rank == 0:
            logger.info(f"F1 verified: 1 tensor in the optimizer, "
                        f"{len(unwrapped._f1_rows)} rows unmasked")

    if getattr(args, "freeze_speaker", False):
        enc = unwrapped.speaker_encoder
        live = [n for n, p in enc.named_parameters() if p.requires_grad]
        if live:
            raise RuntimeError(f"--freeze_speaker did not take: {live}")
        in_opt = {id(p) for g in optimizer.param_groups for p in g["params"]}
        leaked = [n for n, p in enc.named_parameters() if id(p) in in_opt]
        if leaked:
            raise RuntimeError(f"frozen speaker params are in the optimizer: {leaked}")
        if rank == 0:
            logger.info(f"speaker table FROZEN: {sum(1 for _ in enc.parameters())} "
                        f"tensors, none in the optimizer")

    if getattr(args, "l2sp_weight", 0.0) > 0:
        unwrapped._l2sp_ref = {
            n: p.detach().to("cpu", copy=True).to(torch.bfloat16)
            for n, p in unwrapped.backbone.named_parameters()
        }
        if rank == 0:
            n_ref = len(unwrapped._l2sp_ref)
            logger.info(f"L2-SP anchor snapshot: {n_ref} backbone tensors "
                        f"(bf16 on CPU, ~{sum(t.numel() for t in unwrapped._l2sp_ref.values()) * 2 / 1e9:.1f} GB), "
                        f"weight={args.l2sp_weight}")

    if args.stage == 4:
        dpo_train_loop(model, ref_model, loader, optimizer, scheduler, device, args, rank)
    elif args.stage == 5:
        mtp_healing_loop(model, loader, optimizer, scheduler, device, args, rank)
    else:
        train_loop(model, loader, optimizer, scheduler, device, args, rank,
                   start_step=start_step)

    if is_ddp:
        cleanup_distributed()

    if rank == 0:
        logger.info("Training complete.")

if __name__ == "__main__":
    main()
