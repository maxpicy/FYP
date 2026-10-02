# test_pause_exit_weight.py: The pause-exit loss weighting.

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from model import DepthModule


def _tiny():
    torch.manual_seed(0)
    return DepthModule(d_model=16, num_levels=3, codebook_size=32, d_depth=16,
                       num_layers=1, depth_arch="gru")


def test_none_and_all_ones_are_identical():
    m = _tiny()
    h = torch.randn(7, 16)
    t = torch.randint(0, 32, (7, 3))
    a = m.forward_loss(h, t)
    b = m.forward_loss(h, t, frame_weights=torch.ones(7))
    assert torch.allclose(a, b)


def test_nonuniform_weight_changes_loss_and_backprops():
    m = _tiny()
    h = torch.randn(7, 16, requires_grad=True)
    t = torch.randint(0, 32, (7, 3))
    w = torch.ones(7)
    w[2] = 5.0
    a = m.forward_loss(h, t)
    b = m.forward_loss(h, t, frame_weights=w)
    assert not torch.allclose(a, b)
    b.backward()
    assert h.grad is not None and torch.isfinite(h.grad).all()


def test_weights_are_normalised():
    m = _tiny()
    h = torch.randn(7, 16)
    t = torch.randint(0, 32, (7, 3))
    w = torch.rand(7) + 0.5
    a = m.forward_loss(h, t, frame_weights=w)
    b = m.forward_loss(h, t, frame_weights=w * 3.0)
    assert torch.allclose(a, b, atol=1e-6)


def test_transition_mask_selects_pause_to_speech_only():
    pause = torch.tensor([5, 9])
    prev0 = torch.tensor([-1, 5, 9, 5, 7, 9, 2])
    tgt0 = torch.tensor([3, 4, 9, 5, 5, 1, 8])
    exit_ = torch.isin(prev0, pause) & ~torch.isin(tgt0, pause)
    assert exit_.tolist() == [False, True, False, False, False, True, False]
    w = torch.where(exit_, torch.full_like(prev0, 0, dtype=torch.float32) + 4.0,
                    torch.ones_like(prev0, dtype=torch.float32))
    assert w.tolist() == [1.0, 4.0, 1.0, 1.0, 1.0, 4.0, 1.0]
