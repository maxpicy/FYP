# test_s1mini_render.py: The Path B renderer shim.

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from s1mini_render import FULL_VOCAB_MIN, ForcedSemantic

SEM_BEGIN = 100_000
IM_END = 7


def real_sample(logits, temperature, top_p, top_k):
    return torch.tensor([int(logits.argmax())]), "REAL"


def wide(n=152_000):
    return torch.zeros(1, 1, n)


def narrow(peak):
    x = torch.zeros(1, 1, 1024)
    x[0, 0, peak] = 1.0
    return x


def test_semantic_calls_forced_fast_calls_passthrough():
    f = ForcedSemantic(real_sample, SEM_BEGIN, IM_END)
    f.arm([11, 22])
    t, _ = f(wide(), None, None, None)
    assert int(t) == SEM_BEGIN + 11
    t2, tag = f(narrow(peak=333), None, None, None)
    assert tag == "REAL" and int(t2) == 333
    t3, _ = f(wide(), None, None, None)
    assert int(t3) == SEM_BEGIN + 11


def test_ras_double_draw_per_frame():
    f = ForcedSemantic(real_sample, SEM_BEGIN, IM_END)
    f.arm([5, 6, 7])
    got = [int(f(wide(), None, None, None)[0]) for _ in range(6)]
    assert got == [SEM_BEGIN + 5, SEM_BEGIN + 5,
                   SEM_BEGIN + 6, SEM_BEGIN + 6,
                   SEM_BEGIN + 7, SEM_BEGIN + 7]


def test_exhaustion_returns_im_end():
    f = ForcedSemantic(real_sample, SEM_BEGIN, IM_END)
    f.arm([9])
    f(wide(), None, None, None)
    f(wide(), None, None, None)
    t, _ = f(wide(), None, None, None)
    assert int(t) == IM_END


def test_rearm_resets_counter():
    f = ForcedSemantic(real_sample, SEM_BEGIN, IM_END)
    f.arm([1, 2])
    for _ in range(4):
        f(wide(), None, None, None)
    f.arm([3])
    t, _ = f(wide(), None, None, None)
    assert int(t) == SEM_BEGIN + 3


def test_fast_calls_masked_to_residual_range():
    from s1mini_render import RES_MAX
    f = ForcedSemantic(real_sample, SEM_BEGIN, IM_END)
    f.arm([1])
    x = torch.zeros(1, 1, 4096)
    x[0, 0, 2000] = 5.0
    x[0, 0, 700] = 1.0
    t, tag = f(x, None, None, None)
    assert tag == "REAL" and int(t) == 700
    t2, _ = f(narrow(peak=RES_MAX), None, None, None)
    assert int(t2) == RES_MAX


def test_fast_temps_indexed_per_level_and_counter_always_advances():
    seen = []

    def spy(logits, temperature, top_p, top_k):
        seen.append(float(temperature))
        return torch.tensor([0]), "REAL"

    temps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    f = ForcedSemantic(spy, SEM_BEGIN, IM_END, fast_temps=temps)
    f.arm([1, 2])
    for _ in range(18):
        f(narrow(peak=5), None, None, None)
    assert seen == pytest.approx(temps + temps, abs=1e-6), seen


def test_fast_temps_default_none_leaves_temperature_untouched():
    seen = []

    def spy(logits, temperature, top_p, top_k):
        seen.append(temperature)
        return torch.tensor([0]), "REAL"

    f = ForcedSemantic(spy, SEM_BEGIN, IM_END)
    f.arm([1])
    f(narrow(peak=5), "SENTINEL", None, None)
    assert seen == ["SENTINEL"]


def test_returned_probs_is_none_and_only_index_zero_used():
    f = ForcedSemantic(real_sample, SEM_BEGIN, IM_END)
    f.arm([4])
    t, p = f(wide(), None, None, None)
    assert p is None and t.shape == (1,) and t.dtype == torch.long


def test_resume_state_keeps_complete_rows_and_drops_torn_or_unseeded(tmp_path):
    import json

    from s1mini_render import resume_state

    out = tmp_path / "rendered.jsonl"
    assert resume_state(out) == (set(), [])
    lines = [
        json.dumps({"renderer": {"model": "m"}}),
        json.dumps({"renderer": {"row": 0, "seed": 5}}),
        json.dumps({"renderer": {"row": 2, "seed": 7}}),
        json.dumps({"renderer": {"row": 2, "seed": 7}}),
        '{"renderer": {"row": 3, "se',
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    done, kept = resume_state(out)
    assert done == {0, 2}
    assert [json.loads(k)["renderer"]["row"] for k in kept] == [0, 2]
