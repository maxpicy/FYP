# test_silence_guard.py: The SilenceGuard against codebook 0 silence collapse.

import inspect
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from test_depth_checkpoint import SilenceGuard

SIL = json.load(open(REPO / "data/silence_codes_fish_s2.json", encoding="utf-8"))
CODES = SIL["codes"]
PAUSE = CODES[0]
SPEECH = 622


def feed(g, seq, at_last=False):
    masked = []
    for i, c in enumerate(seq):
        if g.mask_now(at_last):
            masked.append(i)
        g.update(c, at_last=at_last)
    return masked


def test_inert_on_legitimate_pauses():
    g = SilenceGuard(CODES, budget=10, stop_after=10)
    seq = [PAUSE] * 9 + [SPEECH] + [PAUSE] * 9 + [SPEECH]
    assert feed(g, seq, at_last=False) == []
    assert not g.force_stop(True)
    assert g.stats()["mask_events"] == 0 and g.stats()["max_run"] == 9


def test_mask_fires_on_exactly_the_eleventh_frame():
    g = SilenceGuard(CODES, budget=10, stop_after=0)
    seq = [PAUSE] * 10 + [SPEECH]
    assert feed(g, seq, at_last=False) == [10]
    assert g.run == 0
    s = g.stats()
    assert {k: s[k] for k in ("mask_events", "frames_masked", "max_run", "stopped")} == \
        {"mask_events": 1, "frames_masked": 1, "max_run": 10, "stopped": False}


def test_budget_rearms_after_recovery():
    g = SilenceGuard(CODES, budget=3, stop_after=0)
    seq = [PAUSE] * 3 + [SPEECH] + [PAUSE] * 3 + [SPEECH]
    assert feed(g, seq) == [3, 7]
    assert g.stats()["mask_events"] == 2


def test_no_mask_at_last_word_but_stop():
    g = SilenceGuard(CODES, budget=10, stop_after=10)
    seq = [SPEECH] + [PAUSE] * 12
    assert feed(g, seq, at_last=True) == []
    assert g.force_stop(True)
    assert not g.force_stop(False)
    assert not g.force_stop(None)


def test_no_plan_stop_only_when_enabled():
    seq = [PAUSE] * 12
    g = SilenceGuard(CODES, budget=10, stop_after=10, stop_without_plan=False)
    assert feed(g, seq, at_last=None) == [10, 11]
    g.run = 12
    assert not g.force_stop(None)
    g2 = SilenceGuard(CODES, budget=10, stop_after=10, stop_without_plan=True)
    g2.run = 12
    assert g2.force_stop(None)


def test_stop_at_last_word_requires_the_last_word_to_have_begun():
    g = SilenceGuard(CODES, budget=10, stop_after=10)
    feed(g, [PAUSE] * 12, at_last=True)
    assert not g.force_stop(True)
    feed(g, [SPEECH] + [PAUSE] * 10, at_last=True)
    assert g.force_stop(True)
    assert g.stats()["spoke_at_last"] is True


def test_stop_allowed_when_retries_exhausted_even_if_unspoken():
    g = SilenceGuard(CODES, budget=3, stop_after=3, retries=1)
    feed(g, [PAUSE] * 3, at_last=False)
    assert g.plan_rewind(False) == 3
    feed(g, [SPEECH] * 3, at_last=False)
    feed(g, [PAUSE] * 3, at_last=True)
    assert g.force_stop(True)


def test_stop_after_zero_never_stops():
    g = SilenceGuard(CODES, budget=10, stop_after=0)
    g.run = 100
    assert not g.force_stop(True)


def test_empty_code_set_refused():
    with pytest.raises(ValueError):
        SilenceGuard([], budget=10)


def test_regression_the_clip_the_user_heard():
    seq = [2499, 622, 3593, 2596, 136, 787, 3482, 2834,
           1391, 1088, 1933, 1088, 538, 1391, 1391, 1391, 2342, 1391, 1391, 1391,
           1088, 1088, 1391, 1088]
    g = SilenceGuard(CODES, budget=10, stop_after=10)
    masked = feed(g, seq, at_last=False)
    assert masked[0] == 18
    assert all(c in set(CODES) for c in seq[8:])


def test_mask_index_is_cached_long_tensor():
    torch = pytest.importorskip("torch")
    g = SilenceGuard(CODES)
    a = g.mask_index("cpu")
    b = g.mask_index("cpu")
    assert a is b and a.dtype == torch.long and a.numel() == len(set(CODES))


def test_generate_frame_accepts_cb0_mask():
    torch = pytest.importorskip("torch")
    from model import DepthModule
    sig = inspect.signature(DepthModule.generate_frame)
    assert "cb0_mask" in sig.parameters
    assert sig.parameters["cb0_mask"].default is None


def test_depth_generate_accepts_silence_guard():
    from test_depth_checkpoint import depth_generate
    sig = inspect.signature(depth_generate)
    assert "silence_guard" in sig.parameters
    assert sig.parameters["silence_guard"].default is None


def test_v1_default_never_rewinds():
    g = SilenceGuard(CODES, budget=4)
    for _ in range(8):
        g.update(PAUSE)
    assert g.plan_rewind(False) == 0
    assert g.kicking is False
    assert g.stats()["rewinds"] == 0


def test_rewind_drops_the_run_and_masks_budget_frames():
    g = SilenceGuard(CODES, budget=4, retries=2)
    for _ in range(4):
        g.update(PAUSE)
    assert g.plan_rewind(False) == 4
    assert g.run == 0 and g.kicking is True
    seen = []
    for c in (SPEECH, SPEECH, PAUSE, SPEECH, SPEECH):
        seen.append(g.mask_now(False))
        g.update(c)
    assert seen == [True, True, True, True, False]
    assert g.kicking is False
    s = g.stats()
    assert s["rewinds"] == 1 and s["frames_rewound"] == 4 and s["retries_used"] == 1
    assert s["mask_events"] == 1 and s["frames_masked"] == 4


def test_rewind_never_at_last_word():
    g = SilenceGuard(CODES, budget=3, retries=3)
    for _ in range(5):
        g.update(PAUSE)
    assert g.plan_rewind(True) == 0
    assert g.plan_rewind(None) == 5


def test_retries_exhaust_then_v1_masking_applies():
    g = SilenceGuard(CODES, budget=3, retries=1)
    for _ in range(3):
        g.update(PAUSE)
    assert g.plan_rewind(False) == 3
    for _ in range(3):
        g.mask_now(False); g.update(SPEECH)
    for _ in range(3):
        g.update(PAUSE)
    assert g.plan_rewind(False) == 0
    assert g.mask_now(False) is True
    assert g.stats()["retries_used"] == 1


def test_loop_breaker_rewind_truncates_its_window():
    from test_depth_checkpoint import LoopBreaker
    lb = LoopBreaker(win=4)
    for c in (1, 2, 1, 2, 1, 2):
        lb.update(c)
    assert len(lb.hist) == 4
    lb.rewind(2)
    assert len(lb.hist) == 2 and lb.active is False and lb.run == 0
