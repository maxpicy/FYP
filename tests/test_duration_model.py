# test_duration_model.py: The duration-bounded decode window.

import ast
import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
GEN = ROOT / "scripts" / "gen_fish_cot_samples.py"
FIT = ROOT / "scripts" / "fit_duration_model.py"


def _consts(src, fn):
    tree = ast.parse(src)
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef) and n.name == fn:
            return ast.unparse(n)
    raise AssertionError(f"{fn} not found")


def test_predict_frames_matches_hand_computation():
    src = GEN.read_text(encoding="utf-8")
    ns = {}
    fn = _consts(src, "predict_frames")
    exec(compile(fn, "<gen>", "exec"), ns)
    predict = ns["predict_frames"]

    dm = {"coef": [40.0, 2.0, 0.5, 7.0, -3.0]}
    base = 40 + 2 * 3 + 0.5 * 13
    assert predict(dm, {"prompt": "one two three", "pace": "slow"}) \
        == pytest.approx(base + 7.0)
    assert predict(dm, {"prompt": "one two three", "pace": "fast"}) \
        == pytest.approx(base - 3.0)
    assert predict(dm, {"prompt": "one two three"}) == pytest.approx(base)
    assert predict(dm, {"prompt": "one two three", "pace": "BRISK"}) \
        == pytest.approx(base)


def test_predict_frames_rejects_stale_model():
    src = GEN.read_text(encoding="utf-8")
    ns = {}
    exec(compile(_consts(src, "predict_frames"), "<gen>", "exec"), ns)
    with pytest.raises(ValueError):
        ns["predict_frames"]({"coef": [1.0, 2.0, 3.0]}, {"prompt": "a b"})


def test_predict_frames_has_a_floor():
    src = GEN.read_text(encoding="utf-8")
    ns = {}
    exec(compile(_consts(src, "predict_frames"), "<gen>", "exec"), ns)
    assert ns["predict_frames"]({"coef": [-1e6, 0, 0, 0, 0]},
                                {"prompt": "a b c"}) >= 20.0


def test_window_defaults_are_a_real_window():
    tree = ast.parse(GEN.read_text(encoding="utf-8"))
    got = {}
    for n in ast.walk(tree):
        if (isinstance(n, ast.Call)
                and getattr(n.func, "attr", "") == "add_argument"
                and n.args and isinstance(n.args[0], ast.Constant)
                and n.args[0].value in ("--dur_lo", "--dur_hi")):
            got[n.args[0].value] = next(
                k.value.value for k in n.keywords if k.arg == "default")
    assert got["--dur_lo"] < 1.0 < got["--dur_hi"], got
    assert got["--dur_lo"] > 0.0


def test_duration_bounding_is_off_by_default():
    tree = ast.parse(GEN.read_text(encoding="utf-8"))
    for n in ast.walk(tree):
        if (isinstance(n, ast.Call)
                and getattr(n.func, "attr", "") == "add_argument"
                and n.args and isinstance(n.args[0], ast.Constant)
                and n.args[0].value == "--duration_model"):
            default = next((k.value.value for k in n.keywords
                            if k.arg == "default"), "MISSING")
            assert default is None
            return
    raise AssertionError("--duration_model not found")


def test_max_frames_is_the_bounded_one():
    src = GEN.read_text(encoding="utf-8")
    assert "max_frames=max_fr" in src
    assert "max_frames=args.max_frames" not in src
