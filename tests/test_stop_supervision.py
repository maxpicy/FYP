# test_stop_supervision.py: The stop-head supervision.

import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

IGNORE_INDEX = -100


def _tok():
    from tokenizer import TokenRegistry, expand_tokenizer, load_base_tokenizer
    tokenizer = load_base_tokenizer()
    expand_tokenizer(tokenizer)
    return tokenizer, TokenRegistry(tokenizer)


@pytest.fixture(scope="module")
def build():
    from delay_dataset import DelayMimiDataset

    def _make(tmpdir, weight):
        import json
        tok, reg = _tok()
        p = pathlib.Path(tmpdir) / "rows.jsonl"
        nlev = __import__("delay_dataset").NUM_LEVELS
        row = {"prompt": "hello there friend",
               "speech_tokens": [5, 6, 7, 8],
               "residual_codes": [[1, 2, 3, 4] for _ in range(nlev - 1)],
               "speaker_id": 1}
        p.write_text(json.dumps(row) + "\n", encoding="utf-8")
        return DelayMimiDataset(str(p), tok, reg, stage=1, lazy=False,
                                audio_continue_weight=weight)
    return _make


def _audio_span(item, reg):
    ids = item["input_ids"].tolist()
    start = ids.index(reg.audio_start_id) + 1
    end = ids.index(reg.audio_end_id)
    return start, end


def test_default_is_the_old_behaviour(build, tmp_path):
    _, reg = _tok()
    it = build(tmp_path, 0.0)[0]
    s, e = _audio_span(it, reg)
    assert (it["labels"][s:e] == IGNORE_INDEX).all()


def test_enabled_supervises_every_audio_frame(build, tmp_path):
    _, reg = _tok()
    it = build(tmp_path, 0.1)[0]
    s, e = _audio_span(it, reg)
    assert e > s
    assert (it["labels"][s:e] != IGNORE_INDEX).all()
    assert (it["labels"][s:e] == reg.audio_start_id).all()


def test_stop_token_stays_supervised_either_way(build, tmp_path):
    _, reg = _tok()
    for w in (0.0, 0.1):
        it = build(tmp_path, w)[0]
        ids = it["input_ids"].tolist()
        assert it["labels"][ids.index(reg.audio_end_id)] == reg.audio_end_id


def test_audio_weight_is_small_relative_to_text(build, tmp_path):
    _, reg = _tok()
    it = build(tmp_path, 0.1)[0]
    s, e = _audio_span(it, reg)
    lw = it["loss_weights"]
    assert torch.allclose(lw[s:e], torch.tensor(0.1))
    assert lw[ids_end := it["input_ids"].tolist().index(reg.audio_end_id)] == 1.0
    assert lw[ids_end] > lw[s]
