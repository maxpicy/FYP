# test_depth_cond.py: The depth module's conditioning shape.

import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

FISH_CB = [4096] + [1024] * 9


def _mk(cond, arch="gru", seed=0):
    from model import DepthModule
    torch.manual_seed(seed)
    return DepthModule(d_model=32, num_levels=10, codebook_size=1024,
                       d_depth=64, num_layers=2, cross_frame=0,
                       codebook_sizes=FISH_CB, depth_arch=arch,
                       depth_cond=cond)


def _tgt(n=4):
    return torch.stack([torch.randint(0, s, (n,)) for s in FISH_CB], dim=1)


def test_add_mode_unchanged():
    m = _mk("add")
    assert hasattr(m, "code_embeds") and len(m.code_embeds) == 9
    assert not hasattr(m, "shared_embed")
    assert torch.isfinite(m.forward_loss(torch.randn(3, 32), _tgt(3)))


def test_prefix_registers_no_unused_parameters():
    m = _mk("prefix")
    assert not hasattr(m, "start"), "unused `start` still registered"
    loss = m.forward_loss(torch.randn(4, 32), _tgt())
    loss.backward()
    dead = [n for n, p in m.named_parameters()
            if p.requires_grad and p.grad is None]
    assert not dead, f"parameters with no gradient: {dead}"


def test_add_mode_still_has_start():
    m = _mk("add")
    assert hasattr(m, "start")


def test_prefix_uses_one_shared_table():
    m = _mk("prefix")
    assert hasattr(m, "shared_embed") and not hasattr(m, "code_embeds")
    assert m.shared_embed.num_embeddings == 4096
    assert m.level_embed.shape == (10, 64)


def test_prefix_trains():
    m = _mk("prefix")
    loss = m.forward_loss(torch.randn(4, 32), _tgt())
    assert torch.isfinite(loss)
    loss.backward()
    assert m.h_proj.weight.grad is not None
    assert m.shared_embed.weight.grad is not None
    assert m.level_embed.grad is not None


def test_prefix_conditioning_actually_reaches_the_output():
    m = _mk("prefix").eval()
    tgt = _tgt(1)
    with torch.no_grad():
        a = m._inputs_teacher_forced(torch.randn(1, 32), tgt)
        b = m._inputs_teacher_forced(torch.randn(1, 32) * 10, tgt)
    assert not torch.allclose(a[:, 0], b[:, 0]), "prefix position ignores h"
    torch.testing.assert_close(a[:, 1:], b[:, 1:])


def test_add_mode_broadcasts_h_everywhere():
    m = _mk("add").eval()
    tgt = _tgt(1)
    with torch.no_grad():
        a = m._inputs_teacher_forced(torch.randn(1, 32), tgt)
        b = m._inputs_teacher_forced(torch.randn(1, 32) * 10, tgt)
    assert not torch.allclose(a[:, 1:], b[:, 1:])


def test_level_identity_distinguishes_same_id_at_different_levels():
    m = _mk("prefix").eval()
    with torch.no_grad():
        e = m.shared_embed(torch.tensor([500]))
        assert not torch.allclose(e + m.level_embed[1], e + m.level_embed[2])


@pytest.mark.parametrize("arch", ["gru", "transformer"])
def test_prefix_generate_frame_in_range(arch):
    m = _mk("prefix", arch=arch).eval()
    out = m.generate_frame(torch.randn(1, 32), level_temps=[0.7] * 10)
    assert len(out) == 10
    assert all(0 <= c < s for c, s in zip(out, FISH_CB))
