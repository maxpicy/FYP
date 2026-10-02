# test_dataset_auto_stage.py: Per-row stage selection for replay mixing.

import json
import tempfile
from pathlib import Path

import torch

COT_ROW = {
    "prompt": "Oh great, another Monday.",
    "reasoning": "<THINK> Sarcasm. [EMO:sarcastic] [PACE:slow] [PITCH:low] </THINK>",
    "audio_file": "a.wav",
    "speech_tokens": [42, 917, 203, 81, 556],
}
REPLAY_ROW = {
    "prompt": "The weather is lovely.",
    "audio_file": "b.wav",
    "speech_tokens": [10, 20, 30, 40, 50, 60],
}


def _write(rows):
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
    for r in rows:
        f.write(json.dumps(r) + "\n")
    f.close()
    return f.name


def test_auto_matches_fixed_stages(tokenizer, registry):
    from dataset import MambaCoTTTSDataset

    p_mixed = _write([COT_ROW, REPLAY_ROW])
    p_cot = _write([COT_ROW])
    p_rep = _write([REPLAY_ROW])
    try:
        ds_auto = MambaCoTTTSDataset(p_mixed, tokenizer, registry, stage="auto")
        ds_s2 = MambaCoTTTSDataset(p_cot, tokenizer, registry, stage=2)
        ds_s1 = MambaCoTTTSDataset(p_rep, tokenizer, registry, stage=1)

        cot_auto, cot_fixed = ds_auto[0], ds_s2[0]
        assert torch.equal(cot_auto["input_ids"], cot_fixed["input_ids"]), \
            "auto must build the exact stage-2 sequence for a reasoning row"
        for k in ("prompt_end", "reasoning_start", "reasoning_end",
                  "speech_start", "speech_end"):
            assert cot_auto[k] == cot_fixed[k], k

        rep_auto, rep_fixed = ds_auto[1], ds_s1[0]
        assert torch.equal(rep_auto["input_ids"], rep_fixed["input_ids"]), \
            "auto must build the exact stage-1 sequence for a replay row"
        think_id = registry.think_start_id
        assert think_id not in rep_auto["input_ids"].tolist(), \
            "replay rows must NOT get an empty <THINK> block"
    finally:
        for p in (p_mixed, p_cot, p_rep):
            Path(p).unlink(missing_ok=True)


def test_collator_auto_mixed_batch(tokenizer, registry):
    from dataset import MambaCoTTTSDataset, MambaCoTDataCollator

    p_mixed = _write([COT_ROW, REPLAY_ROW])
    try:
        ds = MambaCoTTTSDataset(p_mixed, tokenizer, registry, stage="auto")
        col = MambaCoTDataCollator(pad_token_id=registry.pad_id,
                                   registry=registry, stage="auto")
        batch = col([ds[0], ds[1]])
        lw = batch["loss_weights"]

        r0 = ds[0]
        assert torch.allclose(
            lw[0, r0["reasoning_start"]:r0["reasoning_end"]],
            torch.full((r0["reasoning_end"] - r0["reasoning_start"],),
                       col.lambda_r)), "CoT row reasoning region"
        for i, item in enumerate((ds[0], ds[1])):
            assert torch.allclose(
                lw[i, item["speech_start"]:item["speech_end"]],
                torch.full((item["speech_end"] - item["speech_start"],),
                           col.lambda_s)), f"row {i} speech region"
        r1 = ds[1]
        pre_speech = lw[1, :r1["speech_start"] - 1]
        assert not torch.any(pre_speech == col.lambda_r) or col.lambda_r in (0.0, 1.0)
    finally:
        Path(p_mixed).unlink(missing_ok=True)
