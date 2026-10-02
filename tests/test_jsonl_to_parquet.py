# test_jsonl_to_parquet.py: The corpus parquet conversion is lossless.

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
pq = pytest.importorskip("pyarrow.parquet")

from jsonl_to_parquet import from_row, to_row

ROWS = [
    {"prompt": "Hello there.", "reasoning": "calm, warm", "emotion_label": "neutral", "pace": "P3", "pitch": "H2",
     "speaker_id": 4200, "speech_tokens": [1, 4095, 0], "residual_codes": [[5, 6, 7]] * 9, "source": "qwen",
     "plan": {"words": ["Hello", "there."], "pitch": [0.25, -1.5e-3], "dur": [3, 4]}, "cot_form": "full"},
    {"prompt": "No think block.", "speaker_id": 12, "speech_tokens": [], "residual_codes": [[] for _ in range(9)],
     "source": "emilia", "id": "EN_B00000_S00000_W000000", "dnsmos": 3.41, "speaker": "EN_B00000_S00000",
     "plan": {}, "band": 2, "replay": True},
    {"prompt": "Odd types.", "reasoning": None, "speaker_id": "4200", "speech_tokens": [1, 2, 99999],
     "residual_codes": [[1], "x"], "plan": None, "reasoning_variants": ["a", "b"], "v2_tags": {"pace": "derived"},
     "file": "a.wav", "caption_source": "psc"},
]


@pytest.mark.parametrize("r", ROWS)
def test_row_round_trip(r):
    assert to_row(from_row(r)) == r


def test_cli_shards_and_verifies(tmp_path):
    src = tmp_path / "c.jsonl"
    rows = [dict(r, prompt=f"{r['prompt']} #{i}") for i in range(7) for r in ROWS]
    src.write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = tmp_path / "pq"
    p = subprocess.run([sys.executable, str(ROOT / "scripts" / "jsonl_to_parquet.py"), "--src", str(src), "--out",
                        str(out), "--shard_rows", "5", "--workers", "2"], capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr
    shards = sorted(out.glob("train-*.parquet"))
    assert len(shards) == 5 and shards[0].name == "train-00000-of-00005.parquet"
    back = [to_row(c) for s in shards for c in pq.read_table(s).to_pylist()]
    assert back == rows
    man = json.loads((out / "train_manifest.json").read_text())
    assert man["rows"] == 21 and man["lossless_round_trip_verified"]
