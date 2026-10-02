# test_level_decay.py: The per-codebook depth loss weighting.

import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

FISH_CB = [4096] + [1024] * 9


def _mk(decay):
    from model import DepthModule
    torch.manual_seed(0)
    return DepthModule(d_model=32, num_levels=10, codebook_size=1024,
                       d_depth=64, num_layers=2, cross_frame=0,
                       codebook_sizes=FISH_CB, level_decay=decay)


def _weights(m, semantic_weight=1.0):
    d = m.level_decay
    ws = [semantic_weight if k == 0 else d ** (k - 1) for k in range(m.num_levels)]
    if d != 1.0:
        ws = [w * m.num_levels / sum(ws) for w in ws]
    return ws


def test_default_is_flat_and_unchanged():
    m = _mk(1.0)
    assert m.level_decay == 1.0
    assert _weights(m)[1:] == [1.0] * 9


def test_decay_emphasises_coarse_codebooks():
    ws = _weights(_mk(0.8))[1:]
    assert all(a > b for a, b in zip(ws, ws[1:])), ws


def test_decay_preserves_total_scale():
    for d in (0.6, 0.8, 0.95):
        assert sum(_weights(_mk(d))) == pytest.approx(10.0, rel=1e-6)


@pytest.mark.parametrize("decay", [1.0, 0.8])
def test_forward_loss_finite_and_differentiable(decay):
    m = _mk(decay)
    h = torch.randn(4, 32)
    tgt = torch.stack([torch.randint(0, s, (4,)) for s in FISH_CB], dim=1)
    loss = m.forward_loss(h, tgt, semantic_weight=1.5)
    assert torch.isfinite(loss)
    loss.backward()
    assert m.h_proj.weight.grad is not None


def test_decay_actually_changes_the_loss():
    torch.manual_seed(1)
    h = torch.randn(4, 32)
    tgt = torch.stack([torch.randint(0, s, (4,)) for s in FISH_CB], dim=1)
    flat = _mk(1.0).forward_loss(h, tgt).item()
    dec = _mk(0.7).forward_loss(h, tgt).item()
    assert abs(flat - dec) > 1e-4, (flat, dec)
