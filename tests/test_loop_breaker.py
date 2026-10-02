# test_loop_breaker.py: The codebook 0 loop breaker.

import random
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

tdc = pytest.importorskip("test_depth_checkpoint")
fpk = pytest.importorskip("fan_periodk")


def test_constant_sequence_is_zero():
    assert tdc.flipk_rate([7] * 40) == 0.0


def test_exact_hand_count():
    got = tdc.flipk_rate([5, 7, 5, 7, 5], k_lo=2, k_hi=4)
    assert abs(got - 100.0 * 4 / 6) < 1e-9


def test_period2_alternation_fires():
    seq = [1, 2] * 20
    assert tdc.flipk_rate(seq) > 25.0


def test_period5_cycle_fires():
    seq = [3, 8, 1, 9, 4] * 8
    assert tdc.flipk_rate(seq) > 12.0
    assert tdc.flipk_rate(seq, k_lo=2, k_hi=2) == 0.0


def test_diverse_sequence_is_low():
    rng = random.Random(7)
    seq = [rng.randrange(4096) for _ in range(200)]
    assert tdc.flipk_rate(seq) < 2.0


def test_short_sequence_no_crash():
    assert tdc.flipk_rate([]) == 0.0
    assert tdc.flipk_rate([1]) == 0.0
    assert tdc.flipk_rate([1, 2]) == 0.0


def test_parity_with_offline_scanner():
    rng = random.Random(42)
    for _ in range(5):
        seq = [rng.randrange(30) for _ in range(120)]
        import numpy as np
        c = np.asarray(seq)
        hits = tot = 0
        for k in range(2, 9):
            n = len(c) - k
            hits += fpk.flip_k(c, k) / 100.0 * n
            tot += n
        expect = 100.0 * hits / tot
        assert abs(tdc.flipk_rate(seq) - expect) < 1e-6


def test_no_trigger_below_window():
    lb = tdc.LoopBreaker(threshold=5.0, win=32)
    for c in [1, 2] * 10:
        lb.update(c)
    assert not lb.active and lb.events == 0


def test_trigger_on_loop_and_recover():
    lb = tdc.LoopBreaker(threshold=5.0, win=16)
    for c in [1, 2] * 16:
        lb.update(c)
    assert lb.active
    assert lb.events == 1
    assert lb.max_metric > 25.0
    for c in range(100, 132):
        lb.update(c)
    assert not lb.active
    assert lb.run == 0
    assert lb.events == 1


def test_force_stop_semantics():
    lb = tdc.LoopBreaker(threshold=5.0, win=16, stop_after=8)
    for c in [3, 9] * 8:
        lb.update(c)
    assert lb.active and not lb.force_stop
    for c in [3, 9] * 4:
        lb.update(c)
    assert lb.force_stop
    lb0 = tdc.LoopBreaker(threshold=5.0, win=16, stop_after=0)
    for c in [3, 9] * 40:
        lb0.update(c)
    assert lb0.active and not lb0.force_stop


def test_clean_sequence_never_intervenes():
    rng = random.Random(3)
    lb = tdc.LoopBreaker(threshold=5.0, win=32)
    for _ in range(300):
        lb.update(rng.randrange(4096))
    assert lb.events == 0 and lb.frames_intervened == 0
    assert lb.stats() == {"events": 0, "frames": 0,
                          "max_flipk": lb.stats()["max_flipk"],
                          "stopped": False}
    assert lb.stats()["max_flipk"] < 5.0


def test_stats_shape():
    lb = tdc.LoopBreaker(threshold=5.0, win=16)
    for c in [1, 2] * 20:
        lb.update(c)
    s = lb.stats()
    assert set(s) == {"events", "frames", "max_flipk", "stopped"}
    assert s["frames"] > 0 and s["stopped"] is False


@pytest.mark.parametrize("name,row,warble", [
    ("oracle", 5, True), ("oracle", 9, True),
    ("oracle", 0, False), ("oracle", 2, False),
    ("self", 5, True), ("self", 0, False),
])


def test_ear_validated_map(name, row, warble):
    import json
    path = REPO / "_fan" / "codes" / f"{name}.jsonl"
    if not path.exists():
        pytest.skip("probe codes not fetched on this machine")
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    c = rows[row]["speech_tokens"]
    win, step = 32, 8
    worst = max(tdc.flipk_rate(c[s:s + win])
                for s in range(0, max(1, len(c) - win + 1), step))
    if warble:
        assert worst > 5.0
    else:
        assert worst <= 5.0
