# test_eval_suite_gates.py: The evaluation suite reports metrics only, no pass/fail gates.

import re
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import eval_suite
from contour_metrics import contour_metrics

SUITE_SRC = (REPO / "scripts" / "eval_suite.py").read_text(encoding="utf-8")


def _voiced_signal(sr=16000, secs=2.0, f0=140.0, vib_hz=6.0, vib_st=1.5):
    t = np.arange(int(sr * secs)) / sr
    st = 3.0 * np.sin(2 * np.pi * 0.4 * t) + vib_st * np.sin(2 * np.pi * vib_hz * t)
    f = f0 * 2 ** (st / 12.0)
    phase = 2 * np.pi * np.cumsum(f) / sr
    y = 0.5 * np.sin(phase) + 0.2 * np.sin(2 * phase) + 0.1 * np.sin(3 * phase)
    return y.astype(np.float32), sr


def test_prosodic_is_exactly_the_complement_of_fast():
    y, sr = _voiced_signal()
    m = contour_metrics(y, sr)
    assert not np.isnan(m["prosodic"]) and not np.isnan(m["fast"])
    assert m["fast"] > 1.0
    assert m["prosodic"] + m["fast"] == pytest.approx(100.0, abs=1e-6)


def test_the_gate_set_is_empty():
    assert eval_suite.GATES == {}


def test_the_panel_prints_no_verdict():
    main_src = SUITE_SRC[SUITE_SRC.index("def main():"):]
    assert '"PASS"' not in main_src and "FAIL:" not in main_src
    assert "gate\")" not in main_src
    assert "No gates, no PASS/FAIL" in main_src


def test_custom_axes_are_opt_in_and_labelled_diagnostic():
    assert '"--custom_axes"' in SUITE_SRC
    m = re.search(r'add_argument\("--custom_axes".*?help="(.*?)"\)', SUITE_SRC, re.S)
    assert m and "DIAGNOSTIC" in m.group(1).upper()
    main_src = SUITE_SRC[SUITE_SRC.index("def main():"):]
    assert "if args.custom_axes:" in main_src
    assert "DIAGNOSTIC ONLY" in main_src


def test_the_wer_protocol_is_named_on_the_output():
    main_src = SUITE_SRC[SUITE_SRC.index("def main():"):]
    assert "field protocol" in main_src and "not field-comparable" in main_src
