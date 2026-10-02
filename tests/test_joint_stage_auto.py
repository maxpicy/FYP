# test_joint_stage_auto.py: Per-row staging in the depth-module dataset.

import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_delay_dataset_accepts_auto_and_validates_stage():
    import inspect

    import delay_dataset as dd
    src = inspect.getsource(dd.DelayMimiDataset)
    assert 'assert stage in (1, 2, "auto")' in src, \
        "an unrecognised stage would compare unequal to 2 and silently " \
        "produce a corpus-wide stage-1 run"
    assert 'if self.stage == "auto":' in src, \
        "per-row auto staging is gone"
    assert "row_stage = 2 if str(d.get(\"reasoning\") or \"\").strip() else 1" in src
    assert "if row_stage == 2:" in src, \
        "the THINK block is gated on the dataset-wide stage again"


def test_auto_rule_matches_the_flat_dataset():
    import inspect

    import dataset as flat
    import delay_dataset as dd
    a = inspect.getsource(flat.MambaCoTTTSDataset)
    b = inspect.getsource(dd.DelayMimiDataset)
    assert 'item.get("reasoning", "") or ""' in a or \
           'item.get("reasoning"' in a, "flat dataset's auto rule changed"
    assert 'd.get("reasoning")' in b
    assert 'emotion_label' not in a.split("row_stage")[0][-400:], \
        "the flat dataset now infers stage from emotion fields"


def test_f1_whitelist_reads_the_file_not_the_dataset_object():
    src = (ROOT / "train.py").read_text(encoding="utf-8")
    assert "_cot_embedding_rows(args.data_path, model, args)" in src, \
        "F1 no longer passes the corpus PATH; a dataset object would be empty " \
        "on any corpus above DelayMimiDataset's 2 GB lazy threshold"
    i = src.index("def _cot_embedding_rows")
    body = src[i:i + 3000]
    assert "open(data_path" in body, "the whitelist no longer streams the file"
    code = body.split('"""')[2] if body.count('"""') >= 2 else body
    assert 'getattr(dataset, "data"' not in code, \
        "the whitelist reads dataset.data again"
    assert "for row in" not in code or "for line in fh" in code


def test_f1_whitelist_on_a_real_file(tmp_path):
    import types

    import train as T

    rows = [
        {"prompt": "a", "reasoning": "EMO=sad PACE=slow PITCH=low",
         "emotion_label": "sad", "pace": "slow", "pitch": "low"},
        {"prompt": "b", "reasoning": "EMO=joyful PACE=fast PITCH=high",
         "emotion_label": "joyful", "pace": "fast", "pitch": "high"},
        {"prompt": "c", "reasoning": ""},
        {"prompt": "d"},
    ]
    p = tmp_path / "mix.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    calls = []

    def enc(s, add_special_tokens=False):
        calls.append(s)
        return [1000 + len(s)]

    model = types.SimpleNamespace(
        token_registry=types.SimpleNamespace(think_start_id=7, think_end_id=8),
        tokenizer=types.SimpleNamespace(encode=enc))
    args = types.SimpleNamespace(cot_mode="tags")
    keep = T._cot_embedding_rows(str(p), model, args)

    assert {7, 8} <= keep
    tags = [c for c in calls if c.startswith("EMO=")]
    assert len(tags) == 2, f"expected 2 distinct tag strings, got {tags}"
    assert len(set(tags)) == 2, "a duplicate tag string was re-tokenised"


def test_f1_whitelist_refuses_a_corpus_with_no_reasoning(tmp_path):
    import types

    import train as T
    p = tmp_path / "plain.jsonl"
    p.write_text(json.dumps({"prompt": "x"}), encoding="utf-8")
    model = types.SimpleNamespace(
        token_registry=types.SimpleNamespace(think_start_id=7, think_end_id=8),
        tokenizer=types.SimpleNamespace(
            encode=lambda s, add_special_tokens=False: [1]))
    with pytest.raises(RuntimeError, match="no trainable reasoning tokens"):
        T._cot_embedding_rows(str(p), model,
                              types.SimpleNamespace(cot_mode="tags"))


def test_f1_whitelist_tag_string_matches_the_datasets(tmp_path):
    train_src = (ROOT / "train.py").read_text(encoding="utf-8")
    delay_src = (ROOT / "delay_dataset.py").read_text(encoding="utf-8")
    frag = 'f"EMO={d.get(\'emotion_label\') or \'neutral\'} "'
    assert frag in train_src, "train.py's tag string drifted"
    assert frag in delay_src, "delay_dataset.py's tag string drifted"
