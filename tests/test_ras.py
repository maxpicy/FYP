# test_ras.py: Repetition Aware Sampling.

import pathlib
import sys

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

FISH_CB = [4096] + [1024] * 9


def _depth():
    from model import DepthModule
    torch.manual_seed(0)
    return DepthModule(d_model=16, num_levels=10, codebook_size=1024,
                       d_depth=32, num_layers=1, cross_frame=0,
                       codebook_sizes=FISH_CB).eval()


def test_off_by_default_is_bit_identical():
    m = _depth()
    h = torch.randn(1, 16)
    hist = [[3] * 10 for _ in range(10)]
    torch.manual_seed(7)
    a = m.generate_frame(h, level_temps=[0.7] * 10, rep_window=hist)
    torch.manual_seed(7)
    b = m.generate_frame(h, level_temps=[0.7] * 10, rep_window=hist,
                         ras_tau=0.0)
    assert a == b


def test_escape_fires_when_history_repeats():
    m = _depth()
    h = torch.randn(1, 16)
    torch.manual_seed(11)
    off = m.generate_frame(h, level_temps=[0.7] * 10)
    hist = [[off[k]] * 10 for k in range(10)]

    torch.manual_seed(11)
    on = m.generate_frame(h, level_temps=[0.7] * 10, rep_window=hist,
                          ras_tau=0.5, ras_win=10)
    assert on != off, "escape never fired on a fully repeating history"

    torch.manual_seed(11)
    ctrl = m.generate_frame(h, level_temps=[0.7] * 10, rep_window=hist,
                            ras_tau=0.0)
    assert ctrl == off, "history alone changed the draw (penalty leaked?)"


def test_escape_ignores_the_repetition_penalty():
    import model as M
    src = pathlib.Path(M.__file__).read_text(encoding="utf-8")
    i_snap = src.index("base_logits = logits.clone()")
    i_pen = src.index("logits[idx] = torch.where(s < 0, s * rp, s / rp)")
    assert i_snap < i_pen, "base_logits snapshot taken AFTER the penalty"
    assert "F.softmax(base_logits / t" in src, "escape does not use base_logits"


def test_composes_with_the_penalty():
    m = _depth()
    hist = [[5] * 10 for _ in range(10)]
    per = [1.2] * 10
    per[1] = 1.6
    out = m.generate_frame(torch.randn(1, 16), level_temps=[0.7] * 10,
                           rep_window=hist, rep_penalty=per,
                           ras_tau=0.1, ras_win=10)
    assert len(out) == 10
    assert all(0 <= c < s for c, s in zip(out, FISH_CB))


def test_per_level_tau_is_indexed_by_level():
    m = _depth()
    hist = [[9] * 10 for _ in range(10)]
    tau = [0.0] * 10
    tau[1] = 0.1
    out = m.generate_frame(torch.randn(1, 16), level_temps=[0.7] * 10,
                           rep_window=hist, ras_tau=tau)
    assert len(out) == 10


def test_greedy_path_is_untouched():
    m = _depth()
    h = torch.randn(1, 16)
    a = m.generate_frame(h, level_temps=[0.0] * 10)
    hist = [[a[k]] * 10 for k in range(10)]
    b = m.generate_frame(h, level_temps=[0.0] * 10, rep_window=hist,
                         ras_tau=0.9)
    assert a == b, "RAS altered a greedy decode"
