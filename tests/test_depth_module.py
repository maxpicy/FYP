# test_depth_module.py: The depth module and the frame-aligned grid.

import sys
from pathlib import Path

import pytest
import torch

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))


def _module(**kw):
    from model import DepthModule
    torch.manual_seed(0)
    return DepthModule(d_model=64, num_levels=8, codebook_size=32,
                       d_depth=48, num_layers=2, **kw)


def test_loss_finite_and_semantic_weight():
    m = _module()
    h = torch.randn(10, 64)
    tgt = torch.randint(0, 32, (10, 8))
    l1 = m.forward_loss(h, tgt, semantic_weight=1.0)
    l2 = m.forward_loss(h, tgt, semantic_weight=3.0)
    assert torch.isfinite(l1) and l1.item() > 0
    assert l2.item() > l1.item()
    lvl0 = l2.item() - l1.item()
    assert lvl0 > 0


def test_grad_flow_all_params():
    m = _module()
    h = torch.randn(6, 64, requires_grad=True)
    tgt = torch.randint(0, 32, (6, 8))
    loss = m.forward_loss(h, tgt)
    loss.backward()
    for name, p in m.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name
    assert h.grad is not None and h.grad.abs().sum() > 0


def test_generate_frame_valid_and_deterministic():
    m = _module()
    m.eval()
    h = torch.randn(1, 64)
    out1 = m.generate_frame(h)
    out2 = m.generate_frame(h)
    assert out1 == out2
    assert len(out1) == 8
    assert all(0 <= c < 32 for c in out1)
    out3 = m.generate_frame(h, level_temps=[0.0] + [1.0] * 7, top_k=5)
    assert len(out3) == 8 and all(0 <= c < 32 for c in out3)
    assert out3[0] == out1[0]


def test_intra_frame_conditioning_is_causal():
    m = _module()
    m.eval()
    h = torch.randn(3, 64)
    t1 = torch.randint(0, 32, (3, 8))
    t2 = t1.clone()
    t2[:, 0] = (t2[:, 0] + 1) % 32
    x1 = m._inputs_teacher_forced(h, t1)
    x2 = m._inputs_teacher_forced(h, t2)
    assert torch.equal(x1[:, 0], x2[:, 0])
    assert not torch.equal(x1[:, 1], x2[:, 1])


def test_cross_frame_backcompat_default_off():
    m = _module()
    assert not hasattr(m, "cross_embeds")
    assert not hasattr(m, "cross_start")
    assert m.cross_frame == 0
    assert m._cross_summary(torch.randint(0, 32, (4, 8))) is None
    h = torch.randn(4, 64)
    tgt = torch.randint(0, 32, (4, 8))
    x_none = m._inputs_teacher_forced(h, tgt, prev_codes=None)
    x_legacy = m._inputs_teacher_forced(h, tgt)
    assert torch.equal(x_none, x_legacy)


def test_cross_frame_train_inference_parity():
    m = _module(cross_frame=3)
    m.eval()
    h = torch.randn(1, 64)
    prev = torch.randint(0, 32, (8,)).tolist()
    gen = m.generate_frame(h, prev_codes=prev)
    tgt = torch.tensor(gen).view(1, 8)
    x = m._inputs_teacher_forced(h, tgt, prev_codes=torch.tensor(prev).view(1, 8))
    y, _ = m.gru(x)
    y = m.norm(y)
    tf = [int(m.heads[k](y[:, k]).argmax(-1).item()) for k in range(8)]
    assert tf == gen


def test_cross_frame_channel_is_live_and_gated():
    m = _module(cross_frame=3)
    a = torch.tensor([5, 1, 2, 3, 0, 0, 0, 0]).view(1, 8)
    b = torch.tensor([5, 9, 8, 7, 0, 0, 0, 0]).view(1, 8)
    assert not torch.allclose(m._cross_summary(a), m._cross_summary(b))
    cs = m.cross_start.unsqueeze(0)
    assert torch.allclose(m._cross_summary(None), cs)
    assert torch.allclose(m._cross_summary(torch.full((1, 8), -1)), cs)
    part = torch.tensor([5, 1, -1, 3, 0, 0, 0, 0]).view(1, 8)
    assert torch.allclose(m._cross_summary(part), cs)
    cb0_neg = torch.tensor([-1, 1, 2, 3, 0, 0, 0, 0]).view(1, 8)
    assert not torch.allclose(m._cross_summary(cb0_neg), cs)


def test_cross_frame_grad_flow():
    m = _module(cross_frame=3)
    h = torch.randn(12, 64)
    tgt = torch.randint(0, 32, (12, 8))
    prev = torch.randint(0, 32, (12, 8))
    prev[:3] = -1
    loss = m.forward_loss(h, tgt, prev_codes=prev)
    loss.backward()
    assert m.cross_embeds[0].weight.grad.abs().sum() > 0
    assert m.cross_start.grad.abs().sum() > 0
    for name, p in m.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name


def test_aligned_dataset_grid(tmp_path):
    import json
    from unittest.mock import MagicMock

    row = {
        "prompt": "hello world",
        "speech_tokens": [10, 11, 12, 13],
        "residual_codes": [[20 + k * 10 + t for t in range(4)] for k in range(7)],
        "speaker_id": 5,
    }
    p = tmp_path / "rows.jsonl"
    p.write_text(json.dumps(row) + "\n")

    tok = MagicMock()
    tok.encode.return_value = [101, 102]
    reg = MagicMock()
    reg.bos_id, reg.user_prompt_id = 1, 2
    reg.audio_start_id, reg.audio_end_id, reg.eos_id = 3, 4, 5

    from delay_dataset import DelayMimiDataset
    ds = DelayMimiDataset(str(p), tok, reg, max_seq_len=256, aligned=True)
    item = ds[0]
    codes = item["codes"]
    a0, a1 = item["audio_start"], item["audio_end"]
    assert a1 - a0 == 4
    for s in range(4):
        assert codes[a0 + s][0].item() == row["speech_tokens"][s]
        for k in range(1, 8):
            assert codes[a0 + s][k].item() == row["residual_codes"][k - 1][s]
    assert (codes[:a0] == -1).all() and (codes[a1:] == -1).all()

    ds2 = DelayMimiDataset(str(p), tok, reg, max_seq_len=256, aligned=False)
    item2 = ds2[0]
    assert item2["audio_end"] - item2["audio_start"] == 4 + 7

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
