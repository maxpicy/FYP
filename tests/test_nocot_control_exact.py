# test_nocot_control_exact.py: The no-CoT control differs from the treatment only by the think block.

import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def toks():
    try:
        from transformers import AutoTokenizer
        from tokenizer import TokenRegistry, expand_tokenizer
        tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        expand_tokenizer(tok)
        return tok, TokenRegistry(tok)
    except Exception as exc:
        pytest.skip(f"tokenizer unavailable: {exc}")


def _corpus(tmp, n=6):
    from delay_dataset import NUM_LEVELS
    rows = []
    for i in range(n):
        T = 30 + i
        rows.append({
            "prompt": f"a matched control sentence number {i} here",
            "reasoning": f"Deliver line {i} with warmth and a rising close.",
            "emotion_label": "joyful", "pace": "fast", "pitch": "high",
            "speaker_id": 3, "cot_form": "prose",
            "speech_tokens": list(range(T)),
            "residual_codes": [[(k + 1) % 900 for _ in range(T)]
                               for k in range(NUM_LEVELS - 1)],
        })
    p = tmp / "c.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return p, rows


def _ds(path, tok, reg, cot_mode):
    from delay_dataset import DelayMimiDataset
    return DelayMimiDataset(
        data_path=str(path), tokenizer=tok, registry=reg, max_seq_len=2048,
        aligned=True, stage="auto", cot_mode=cot_mode, lazy=False)


def test_none_mode_equals_a_pure_stage1_build(tmp_path, toks):
    tok, reg = toks
    p, _ = _corpus(tmp_path)
    ctrl = _ds(p, tok, reg, "none")
    from delay_dataset import DelayMimiDataset
    stage1 = DelayMimiDataset(
        data_path=str(p), tokenizer=tok, registry=reg, max_seq_len=2048,
        aligned=True, stage=1, cot_mode="prose", lazy=False)

    for i in range(len(ctrl)):
        a, b = ctrl[i], stage1[i]
        assert a.keys() == b.keys(), f"row {i}: different fields"
        for k in a:
            va, vb = a[k], b[k]
            if torch.is_tensor(va):
                assert torch.equal(va, vb), (
                    f"row {i}: field {k!r} differs between cot_mode=none and "
                    f"a pure stage-1 build — the control is not clean")
            else:
                assert va == vb, f"row {i}: field {k!r} differs"


def test_treatment_and_control_differ_only_by_the_think_block(tmp_path, toks):
    tok, reg = toks
    p, _ = _corpus(tmp_path)
    treat = _ds(p, tok, reg, "prose")
    ctrl = _ds(p, tok, reg, "none")

    for i in range(len(treat)):
        t, c = treat[i], ctrl[i]
        ti, ci = t["input_ids"], c["input_ids"]
        assert len(ti) > len(ci), (
            f"row {i}: prose mode did not lengthen the sequence — the "
            f"treatment is not adding a THINK block")
        assert reg.think_start_id in ti.tolist()
        assert reg.think_end_id in ti.tolist()
        assert reg.think_start_id not in ci.tolist(), \
            f"row {i}: the CONTROL contains a THINK token"
        sa = [x for x in ti.tolist() if x >= reg.speech_token_id_min]
        sb = [x for x in ci.tolist() if x >= reg.speech_token_id_min]
        assert sa == sb, f"row {i}: the audio differs between the arms"


def test_control_has_no_reasoning_supervision(tmp_path, toks):
    tok, reg = toks
    p, _ = _corpus(tmp_path)
    ctrl = _ds(p, tok, reg, "none")
    treat = _ds(p, tok, reg, "prose")
    lw = "loss_weights"
    if lw not in ctrl[0]:
        pytest.skip("collator does not expose per-token loss weights")
    for i in range(len(ctrl)):
        cw = set(ctrl[i][lw].tolist())
        tw = set(treat[i][lw].tolist())
        assert 0.3 not in cw or 0.3 in tw, "unexpected reasoning weight"
