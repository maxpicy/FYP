# dataset.py: Text + think block + speech tokens -> one training sequence with per-region loss weights; collator.

import json
import logging
import torch
from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import List, Dict, Any

from tokenizer import TokenRegistry, NUM_SPEECH_TOKENS

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100


class MambaCoTTTSDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer,
        registry: TokenRegistry,
        stage: int = 1,
        max_seq_len: int = 2048,
        cfg_dropout: float = 0.0,
        cot_mode: str = "prose",
    ):
        assert stage in (1, 2, "auto"), f"stage must be 1, 2 or 'auto', got {stage}"

        self.tokenizer = tokenizer
        self.registry = registry
        self.stage = stage
        self.max_seq_len = max_seq_len
        self.cfg_dropout = cfg_dropout

        self.data: List[dict] = []
        with open(data_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.data.append(json.loads(line))

        assert cot_mode in ("prose", "tags", "none"), (
            f"cot_mode={cot_mode!r} is not supported by the flat "
            f"MambaCoTTTSDataset. The P²-CoT plan tiers (plan / tags+plan / "
            f"full) carry a per-position injection schedule and run on the "
            f"delay/depth path only — launch with --depth_module (or "
            f"--delay_pattern), which routes to delay_dataset.DelayMimiDataset."
            if cot_mode in ("plan", "tags+plan", "full") else cot_mode)
        self.cot_mode = cot_mode
        if cot_mode != "prose":
            for row in self.data:
                if cot_mode == "none":
                    row["reasoning"] = ""
                else:
                    row["reasoning"] = (
                        f"EMO={row.get('emotion_label') or 'neutral'} "
                        f"PACE={row.get('pace') or 'normal'} "
                        f"PITCH={row.get('pitch') or 'normal'}")

        self.bos_id = registry.bos_id
        self.eos_id = registry.eos_id
        self.user_prompt_id = registry.user_prompt_id
        self.think_start_id = registry.think_start_id
        self.think_end_id = registry.think_end_id
        self.audio_start_id = registry.audio_start_id
        self.audio_end_id = registry.audio_end_id

        logger.info(
            f"Loaded {len(self.data)} samples from {data_path} (stage={stage}, "
            f"cot_mode={cot_mode}, max_seq_len={max_seq_len})"
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        item = self.data[idx]

        if self.stage == "auto":
            row_stage = 2 if str(item.get("reasoning", "") or "").strip() else 1
        else:
            row_stage = self.stage

        prompt_text: str = item["prompt"]
        speech_values: List[int] = item["speech_tokens"]
        residual_codes = item.get("residual_codes", None)
        speaker_id = item.get("speaker_id", None)

        clamped = False
        for i, v in enumerate(speech_values):
            if v < 0 or v >= NUM_SPEECH_TOKENS:
                speech_values[i] = max(0, min(v, NUM_SPEECH_TOKENS - 1))
                clamped = True
        if clamped:
            logger.warning(f"Sample {idx}: speech_tokens contained out-of-range values, clamped to [0, {NUM_SPEECH_TOKENS - 1}]")

        speech_ids = [self.registry.speech_value_to_id(v) for v in speech_values]

        if self.cfg_dropout > 0.0 and torch.rand(1).item() < self.cfg_dropout:
            prompt_ids = []
        else:
            prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)

        reasoning_ids = []
        if row_stage == 2:
            reasoning_text = item.get("reasoning", "")
            if reasoning_text:
                reasoning_clean = reasoning_text.strip()
                if reasoning_clean.startswith("<THINK>"):
                    reasoning_clean = reasoning_clean[len("<THINK>"):].strip()
                if reasoning_clean.endswith("</THINK>"):
                    reasoning_clean = reasoning_clean[:-len("</THINK>")].strip()
                if reasoning_clean:
                    reasoning_ids = self.tokenizer.encode(reasoning_clean, add_special_tokens=False)

        if row_stage == 1:
            sequence = (
                [self.bos_id, self.user_prompt_id]
                + prompt_ids
                + [self.audio_start_id]
                + speech_ids
                + [self.audio_end_id, self.eos_id]
            )
            prompt_end = 2 + len(prompt_ids)
            reasoning_start = prompt_end
            reasoning_end = prompt_end
            speech_start = prompt_end + 1
            speech_end = speech_start + len(speech_ids)
        else:
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
            reasoning_start = prompt_end + 1
            reasoning_end = reasoning_start + len(reasoning_ids)
            speech_start = reasoning_end + 2
            speech_end = speech_start + len(speech_ids)

        if len(sequence) > self.max_seq_len:
            overflow = len(sequence) - self.max_seq_len

            max_speech_trim = len(speech_ids)
            speech_trim = min(overflow, max_speech_trim)
            speech_ids_trimmed = speech_ids[:len(speech_ids) - speech_trim]
            overflow -= speech_trim

            reasoning_ids_trimmed = reasoning_ids
            if overflow > 0 and row_stage == 2:
                max_reasoning_trim = len(reasoning_ids)
                reasoning_trim = min(overflow, max_reasoning_trim)
                reasoning_ids_trimmed = reasoning_ids[:len(reasoning_ids) - reasoning_trim]
                overflow -= reasoning_trim

            if overflow > 0:
                logger.warning(
                    f"Sample {idx}: sequence still {overflow} tokens over max_seq_len "
                    f"after truncating speech and reasoning"
                )

            if row_stage == 1:
                sequence = (
                    [self.bos_id, self.user_prompt_id]
                    + prompt_ids
                    + [self.audio_start_id]
                    + speech_ids_trimmed
                    + [self.audio_end_id, self.eos_id]
                )
                prompt_end = 2 + len(prompt_ids)
                reasoning_start = prompt_end
                reasoning_end = prompt_end
                speech_start = prompt_end + 1
                speech_end = speech_start + len(speech_ids_trimmed)
            else:
                sequence = (
                    [self.bos_id, self.user_prompt_id]
                    + prompt_ids
                    + [self.think_start_id]
                    + reasoning_ids_trimmed
                    + [self.think_end_id]
                    + [self.audio_start_id]
                    + speech_ids_trimmed
                    + [self.audio_end_id, self.eos_id]
                )
                prompt_end = 2 + len(prompt_ids)
                reasoning_start = prompt_end + 1
                reasoning_end = reasoning_start + len(reasoning_ids_trimmed)
                speech_start = reasoning_end + 2
                speech_end = speech_start + len(speech_ids_trimmed)

        return {
            "input_ids": torch.tensor(sequence, dtype=torch.long),
            "prompt_end": prompt_end,
            "reasoning_start": reasoning_start,
            "reasoning_end": reasoning_end,
            "speech_start": speech_start,
            "speech_end": speech_end,
            "residual_codes": residual_codes,
            "speaker_id": speaker_id,
        }


@dataclass
class MambaCoTDataCollator:
    pad_token_id: int
    registry: TokenRegistry
    stage: int = 1
    lambda_r: float = 0.3
    lambda_s: float = 1.0
    semantic_weight: float = 1.0
    eos_weight: float = 1.0
    level_cycle: int = 0

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids_list = [item["input_ids"] for item in batch]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids_list, batch_first=True, padding_value=self.pad_token_id
        )
        B, T_max = input_ids.shape

        labels = input_ids.clone()

        loss_weights = torch.ones(B, T_max, dtype=torch.float32)

        for i, item in enumerate(batch):
            seq_len = len(item["input_ids"])
            prompt_end = item["prompt_end"]

            labels[i, :prompt_end] = IGNORE_INDEX

            if seq_len < T_max:
                labels[i, seq_len:] = IGNORE_INDEX

            loss_weights[i, :prompt_end] = 0.0
            if seq_len < T_max:
                loss_weights[i, seq_len:] = 0.0

            if self.stage == 2 or self.stage == "auto":
                r_start = item["reasoning_start"]
                r_end = item["reasoning_end"]
                s_start = item["speech_start"]
                s_end = item["speech_end"]

                think_delim_start = max(prompt_end, r_start - 1)
                think_delim_end = min(seq_len, r_end + 1)
                loss_weights[i, think_delim_start:think_delim_end] = self.lambda_r

                loss_weights[i, s_start - 1:seq_len] = self.lambda_s

            if self.semantic_weight != 1.0 and self.level_cycle > 0:
                s_start = item["speech_start"]
                s_end = min(item["speech_end"], seq_len)
                for pos in range(s_start, s_end, self.level_cycle):
                    loss_weights[i, pos] *= self.semantic_weight
            if self.eos_weight != 1.0:
                s_end = item["speech_end"]
                for pos in (s_end, s_end + 1):
                    if pos < seq_len:
                        loss_weights[i, pos] *= self.eos_weight

        speech_mask = self.registry.build_speech_mask(input_ids) & (labels != IGNORE_INDEX)

        attention_mask = (input_ids != self.pad_token_id).long()

        has_residual = any(item.get("residual_codes") is not None for item in batch)
        residual_codes_tensor = None
        if has_residual:
            num_layers = 0
            for item in batch:
                if item.get("residual_codes") is not None:
                    num_layers = len(item["residual_codes"])
                    break
            if num_layers > 0:
                residual_codes_tensor = torch.full((B, num_layers, T_max), -100, dtype=torch.long)
                for i, item in enumerate(batch):
                    rc = item.get("residual_codes")
                    if rc is not None:
                        s_start = item["speech_start"]
                        for layer_idx, layer_codes in enumerate(rc):
                            for t_idx, code in enumerate(layer_codes):
                                pos = s_start + t_idx
                                if pos < T_max:
                                    residual_codes_tensor[i, layer_idx, pos] = code

        speaker_ids = None
        if any(item.get("speaker_id") is not None for item in batch):
            speaker_ids = torch.tensor(
                [item.get("speaker_id", 0) for item in batch], dtype=torch.long
            )

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "speech_mask": speech_mask,
            "loss_weights": loss_weights,
            "residual_codes": residual_codes_tensor,
            "speaker_ids": speaker_ids,
        }
