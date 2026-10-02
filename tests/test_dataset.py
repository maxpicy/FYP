# test_dataset.py: JSONL parsing, sequence construction and collation.

import json
import os
import tempfile

import pytest
import torch

from config import (
    AUDIO_END_TOKEN,
    AUDIO_START_TOKEN,
    NUM_SPEECH_TOKENS,
    THINK_END_TOKEN,
    THINK_START_TOKEN,
    USER_PROMPT_TOKEN,
    get_all_special_tokens,
)
from dataset import MambaCoTDataCollator


class SimpleTokenizer:
    def __init__(self):
        self.vocab = {}
        self._next_id = 0
        self.eos_token_id = self._add("[EOS]")

        for token in get_all_special_tokens():
            self._add(token)

    def _add(self, token):
        if token not in self.vocab:
            self.vocab[token] = self._next_id
            self._next_id += 1
        return self.vocab[token]

    def encode(self, text, add_special_tokens=False, return_tensors=None):
        tokens = text.split()
        ids = [self.vocab.get(t, self._add(t)) for t in tokens]
        if return_tensors == "pt":
            return torch.tensor([ids], dtype=torch.long)
        return ids

    def convert_tokens_to_ids(self, tokens):
        return [self.vocab.get(t, self._add(t)) for t in tokens]

    def convert_ids_to_tokens(self, ids):
        inv = {v: k for k, v in self.vocab.items()}
        return [inv.get(i, f"[UNK_{i}]") for i in ids]

    def decode(self, ids, skip_special_tokens=False):
        return " ".join(self.convert_ids_to_tokens(ids))

    def __len__(self):
        return len(self.vocab)

    def add_special_tokens(self, special_tokens_dict):
        count = 0
        for tokens in special_tokens_dict.values():
            if isinstance(tokens, list):
                for t in tokens:
                    if t not in self.vocab:
                        self._add(t)
                        count += 1
        return count


class TestJSONLParsing:
    def test_valid_jsonl(self):
        data = [
            {
                "prompt": "Hello world",
                "reasoning": "Neutral greeting",
                "audio_file": "test.wav",
            },
            {
                "prompt": "Oh great",
                "reasoning": "Sarcastic [EMO:sarcastic]",
                "audio_file": "test2.wav",
                "emotion_label": "sarcastic",
            },
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
            tmpfile = f.name

        try:
            parsed = []
            with open(tmpfile, "r") as f:
                for line in f:
                    if line.strip():
                        parsed.append(json.loads(line))

            assert len(parsed) == 2
            assert parsed[0]["prompt"] == "Hello world"
            assert parsed[1]["emotion_label"] == "sarcastic"
        finally:
            os.unlink(tmpfile)

    def test_empty_lines_skipped(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({"prompt": "a", "reasoning": "", "audio_file": "x.wav"}) + "\n")
            f.write("\n")
            f.write("  \n")
            f.write(json.dumps({"prompt": "b", "reasoning": "", "audio_file": "y.wav"}) + "\n")
            tmpfile = f.name

        try:
            parsed = []
            with open(tmpfile, "r") as f:
                for line in f:
                    if line.strip():
                        parsed.append(json.loads(line))
            assert len(parsed) == 2
        finally:
            os.unlink(tmpfile)

    def test_optional_fields(self):
        data = {"prompt": "Test", "reasoning": "", "audio_file": "x.wav",
                "speech_tokens": [42, 100, 200]}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps(data) + "\n")
            tmpfile = f.name

        try:
            with open(tmpfile, "r") as f:
                item = json.loads(f.readline())
            assert item["speech_tokens"] == [42, 100, 200]
        finally:
            os.unlink(tmpfile)


class TestSequenceConstruction:
    def setup_method(self):
        self.tokenizer = SimpleTokenizer()

    def test_stage1_no_reasoning(self):
        prompt = "Hello world"
        prompt_text = f"{USER_PROMPT_TOKEN} {prompt}"
        prompt_ids = self.tokenizer.encode(prompt_text)

        audio_start_ids = self.tokenizer.encode(AUDIO_START_TOKEN)
        speech_ids = self.tokenizer.convert_tokens_to_ids(["[SPEECH_42]", "[SPEECH_100]"])
        audio_end_ids = self.tokenizer.encode(AUDIO_END_TOKEN)

        input_ids = prompt_ids + audio_start_ids + speech_ids + audio_end_ids
        input_ids.append(self.tokenizer.eos_token_id)

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        assert USER_PROMPT_TOKEN in tokens
        assert AUDIO_START_TOKEN in tokens
        assert AUDIO_END_TOKEN in tokens
        assert THINK_START_TOKEN not in tokens
        assert THINK_END_TOKEN not in tokens

    def test_stage2_with_reasoning(self):
        prompt = "Great Monday"
        reasoning = "[EMO:sarcastic] [PACE:slow]"

        prompt_ids = self.tokenizer.encode(f"{USER_PROMPT_TOKEN} {prompt}")
        reasoning_ids = self.tokenizer.encode(f"{THINK_START_TOKEN} {reasoning} {THINK_END_TOKEN}")
        audio_start_ids = self.tokenizer.encode(AUDIO_START_TOKEN)
        speech_ids = self.tokenizer.convert_tokens_to_ids(["[SPEECH_0]"])
        audio_end_ids = self.tokenizer.encode(AUDIO_END_TOKEN)

        input_ids = prompt_ids + reasoning_ids + audio_start_ids + speech_ids + audio_end_ids
        input_ids.append(self.tokenizer.eos_token_id)

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        assert THINK_START_TOKEN in tokens
        assert THINK_END_TOKEN in tokens
        assert "[EMO:sarcastic]" in tokens

    def test_speech_token_range(self):
        speech_tokens = [f"[SPEECH_{i}]" for i in [0, 512, NUM_SPEECH_TOKENS - 1]]
        ids = self.tokenizer.convert_tokens_to_ids(speech_tokens)
        assert len(ids) == 3
        assert all(isinstance(i, int) for i in ids)


@pytest.fixture(scope="module")
def real_registry():
    from tokenizer import load_base_tokenizer, expand_tokenizer, TokenRegistry
    tok = load_base_tokenizer()
    expand_tokenizer(tok)
    return TokenRegistry(tok)


class TestCollator:
    def _make_item(self, registry, input_ids, prompt_end,
                   reasoning_start=0, reasoning_end=0,
                   speech_start=0, speech_end=0,
                   residual_codes=None, speaker_id=None):
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "prompt_end": prompt_end,
            "reasoning_start": reasoning_start,
            "reasoning_end": reasoning_end,
            "speech_start": speech_start,
            "speech_end": speech_end,
            "residual_codes": residual_codes,
            "speaker_id": speaker_id,
        }

    def test_padding(self, real_registry):
        collator = MambaCoTDataCollator(
            pad_token_id=real_registry.pad_id,
            registry=real_registry,
            stage=1,
        )
        batch = [
            self._make_item(
                real_registry,
                input_ids=[real_registry.bos_id, 100, 101, real_registry.eos_id],
                prompt_end=2, speech_start=2, speech_end=4,
            ),
            self._make_item(
                real_registry,
                input_ids=[real_registry.bos_id, 100, 101, 102, 103, real_registry.eos_id],
                prompt_end=2, speech_start=2, speech_end=6,
            ),
        ]
        result = collator(batch)
        assert result["input_ids"].shape == (2, 6)
        assert result["input_ids"][0, 4].item() == real_registry.pad_id
        assert result["input_ids"][0, 5].item() == real_registry.pad_id

    def test_prompt_masked_in_labels(self, real_registry):
        collator = MambaCoTDataCollator(
            pad_token_id=real_registry.pad_id,
            registry=real_registry,
            stage=1,
        )
        batch = [
            self._make_item(
                real_registry,
                input_ids=[real_registry.bos_id, 100, 101, 102, real_registry.eos_id],
                prompt_end=3, speech_start=3, speech_end=5,
            ),
        ]
        result = collator(batch)
        labels = result["labels"]
        assert labels[0, 0].item() == -100
        assert labels[0, 1].item() == -100
        assert labels[0, 2].item() == -100
        assert labels[0, 3].item() != -100

    def test_padding_masked_in_labels(self, real_registry):
        collator = MambaCoTDataCollator(
            pad_token_id=real_registry.pad_id,
            registry=real_registry,
            stage=1,
        )
        batch = [
            self._make_item(
                real_registry,
                input_ids=[real_registry.bos_id, 100, real_registry.eos_id],
                prompt_end=1, speech_start=1, speech_end=3,
            ),
            self._make_item(
                real_registry,
                input_ids=[real_registry.bos_id, 100, 101, 102, real_registry.eos_id],
                prompt_end=1, speech_start=1, speech_end=5,
            ),
        ]
        result = collator(batch)
        labels = result["labels"]
        assert labels[0, 3].item() == -100
        assert labels[0, 4].item() == -100

    def test_attention_mask(self, real_registry):
        collator = MambaCoTDataCollator(
            pad_token_id=real_registry.pad_id,
            registry=real_registry,
            stage=1,
        )
        batch = [
            self._make_item(
                real_registry,
                input_ids=[real_registry.bos_id, 100, real_registry.eos_id],
                prompt_end=1, speech_start=1, speech_end=3,
            ),
            self._make_item(
                real_registry,
                input_ids=[real_registry.bos_id, 100, 101, 102, real_registry.eos_id],
                prompt_end=1, speech_start=1, speech_end=5,
            ),
        ]
        result = collator(batch)
        mask = result["attention_mask"]
        assert mask[0, 0].item() == 1
        assert mask[0, 2].item() == 1
        assert mask[0, 3].item() == 0
        assert mask[0, 4].item() == 0
        assert mask[1].sum().item() == 5

    def test_loss_weights_stage1(self, real_registry):
        collator = MambaCoTDataCollator(
            pad_token_id=real_registry.pad_id,
            registry=real_registry,
            stage=1,
        )
        batch = [
            self._make_item(
                real_registry,
                input_ids=[real_registry.bos_id, 100, 101, 102, real_registry.eos_id],
                prompt_end=2,
                reasoning_start=2, reasoning_end=2,
                speech_start=2, speech_end=5,
            ),
        ]
        result = collator(batch)
        weights = result["loss_weights"]
        assert weights[0, 0].item() == pytest.approx(0.0)
        assert weights[0, 1].item() == pytest.approx(0.0)
        assert weights[0, 2].item() == pytest.approx(1.0)
        assert weights[0, 3].item() == pytest.approx(1.0)

    def test_loss_weights_stage2(self, real_registry):
        lambda_r = 0.3
        lambda_s = 1.0
        collator = MambaCoTDataCollator(
            pad_token_id=real_registry.pad_id,
            registry=real_registry,
            stage=2,
            lambda_r=lambda_r,
            lambda_s=lambda_s,
        )
        batch = [
            self._make_item(
                real_registry,
                input_ids=[
                    real_registry.bos_id, real_registry.user_prompt_id, 100,
                    real_registry.think_start_id, 200, 201, real_registry.think_end_id,
                    real_registry.audio_start_id, 300, real_registry.audio_end_id,
                    real_registry.eos_id,
                ],
                prompt_end=3,
                reasoning_start=4, reasoning_end=6,
                speech_start=8, speech_end=9,
            ),
        ]
        result = collator(batch)
        weights = result["loss_weights"]
        assert weights[0, 0].item() == pytest.approx(0.0)
        assert weights[0, 1].item() == pytest.approx(0.0)
        assert weights[0, 2].item() == pytest.approx(0.0)
        assert weights[0, 3].item() == pytest.approx(lambda_r)
        assert weights[0, 4].item() == pytest.approx(lambda_r)
        assert weights[0, 5].item() == pytest.approx(lambda_r)
        assert weights[0, 7].item() == pytest.approx(lambda_s)
        assert weights[0, 8].item() == pytest.approx(lambda_s)

    def test_residual_codes_none(self, real_registry):
        collator = MambaCoTDataCollator(
            pad_token_id=real_registry.pad_id,
            registry=real_registry,
            stage=1,
        )
        batch = [
            self._make_item(
                real_registry,
                input_ids=[real_registry.bos_id, 100, real_registry.eos_id],
                prompt_end=1, speech_start=1, speech_end=3,
                residual_codes=None,
            ),
        ]
        result = collator(batch)
        assert result["residual_codes"] is None

    def test_speaker_ids_none(self, real_registry):
        collator = MambaCoTDataCollator(
            pad_token_id=real_registry.pad_id,
            registry=real_registry,
            stage=1,
        )
        batch = [
            self._make_item(
                real_registry,
                input_ids=[real_registry.bos_id, 100, real_registry.eos_id],
                prompt_end=1, speech_start=1, speech_end=3,
                speaker_id=None,
            ),
        ]
        result = collator(batch)
        assert result["speaker_ids"] is None
