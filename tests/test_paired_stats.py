# test_paired_stats.py: The paired statistics: sign test, Holm, contrasts.

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from paired_stats import contrast, holm, sign_p


def test_sign_test_known_values():
    assert sign_p(0, 0) == 1.0
    assert sign_p(5, 5) == 1.0
    assert sign_p(0, 10) == pytest.approx(2 / 1024)
    assert sign_p(2, 8) == pytest.approx(2 * (1 + 10 + 45) / 1024)


def test_holm_is_step_down_and_monotone():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm([0.5]) == [0.5]


def test_contrast_counts_and_interval():
    a = [0.0] * 50 + [0.2] * 50
    b = [0.1] * 100
    r = contrast(a, b)
    assert r["n"] == 100 and r["better"] == 50 and r["worse"] == 50 and r["tie"] == 0
    assert r["delta"] == pytest.approx(0.0, abs=1e-12) and r["ci"][0] < 0 < r["ci"][1] and r["sign_p"] == 1.0
    with pytest.raises(ValueError):
        contrast([0.1], [0.1, 0.2])


def test_cli_reads_eval_suite_score_files(tmp_path):
    for name, w in (("a", [0.0, 0.0, 0.5, 0.1]), ("b", [0.1, 0.2, 0.5, 0.3])):
        (tmp_path / f"{name}.json").write_text(json.dumps({"selfplan": {"per_row": {"wer": w}}}))
    out = tmp_path / "o.json"
    p = subprocess.run([sys.executable, str(ROOT / "scripts" / "paired_stats.py"), "--pair",
                        f"{tmp_path / 'a.json'}:selfplan", f"{tmp_path / 'b.json'}:selfplan", "--out", str(out)],
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    r = json.loads(out.read_text())[0]
    assert r["better"] == 3 and r["tie"] == 1 and r["delta"] == pytest.approx(-0.125)
