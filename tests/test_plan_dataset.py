# test_plan_dataset.py: The three think-block tiers in the dataset.

import json
import os
import subprocess
import sys
import zlib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

torch = pytest.importorskip("torch")


class FakeTokenizer:
    UNK_ID = 0
    UNK_TOKEN = "<|endoftext|>"
    TEXT_BASE = 1000

    def __init__(self, tokens, base_vocab: int = 50277):
        self._token_to_id = {self.UNK_TOKEN: self.UNK_ID}
        self._id_to_token = {self.UNK_ID: self.UNK_TOKEN}
        for offset, token in enumerate(tokens):
            token_id = base_vocab + offset
            self._token_to_id[token] = token_id
            self._id_to_token[token_id] = token

    def __len__(self):
        return max(self._id_to_token) + 1

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, str):
            return self._token_to_id.get(tokens, self.UNK_ID)
        return [self._token_to_id.get(t, self.UNK_ID) for t in tokens]

    def convert_ids_to_tokens(self, ids):
        if isinstance(ids, int):
            return self._id_to_token.get(ids)
        return [self._id_to_token.get(i) for i in ids]

    def encode(self, text, add_special_tokens=False):
        return [self.TEXT_BASE + (zlib.crc32(w.encode()) % 4096)
                for w in text.split()]


def fake_registry():
    import tokenizer as tok
    return tok.TokenRegistry(FakeTokenizer(tok.get_all_new_tokens()))

PROMPT = "alpha beta gamma delta"
REASONING = "Say it warmly and slow down at the end."

PLAN_WORDS = [
    {"word": "alpha", "pitch_bin": 3, "dur_bin": 1, "energy_bin": 0, "dur_frames": 2},
    {"word": "beta", "pitch_bin": None, "dur_bin": 4, "energy_bin": 7, "dur_frames": 4},
    {"word": "gamma", "pitch_bin": 15, "dur_bin": 9, "energy_bin": 3, "dur_frames": 6},
]
SUM_D = 12


def _row(n_frames, *, plan=None, reasoning=REASONING, tags=True, prompt=PROMPT):
    from delay_dataset import NUM_LEVELS
    row = {
        "prompt": prompt,
        "reasoning": reasoning,
        "speaker_id": 3,
        "speech_tokens": list(range(n_frames)),
        "residual_codes": [[(k + 1) % 900 for _ in range(n_frames)]
                           for k in range(NUM_LEVELS - 1)],
    }
    if tags:
        row.update({"emotion_label": "joyful", "pace": "fast", "pitch": "high"})
    if plan is not None:
        row["plan"] = plan
    return row


def write_corpus(path, rows):
    Path(path).write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return str(path)


def make_dataset(path, registry, cot_mode, **kw):
    from delay_dataset import DelayMimiDataset
    import tokenizer as tok
    kw.setdefault("max_seq_len", 2048)
    kw.setdefault("aligned", True)
    kw.setdefault("stage", "auto")
    kw.setdefault("lazy", False)
    return DelayMimiDataset(data_path=path, tokenizer=registry.tokenizer,
                            registry=registry, cot_mode=cot_mode, **kw)


def expected_legacy_ids(reg, tok_, think_text, n_frames, aligned=True, n_levels=None):
    from delay_dataset import NUM_LEVELS
    n_levels = NUM_LEVELS if n_levels is None else n_levels
    stagger = 0 if aligned else n_levels - 1
    span = n_frames + stagger
    think = ([reg.think_start_id] + tok_.encode(think_text) + [reg.think_end_id]
             if think_text else [])
    return ([reg.bos_id, reg.user_prompt_id] + tok_.encode(PROMPT) + think
            + [reg.audio_start_id] + [reg.audio_start_id] * span
            + [reg.audio_end_id, reg.eos_id])


def probe(tmpdir) -> dict:
    import config
    from delay_dataset import DelayMimiDataset, NO_WORD

    tmpdir = Path(tmpdir)
    reg = fake_registry()
    tok_ = reg.tokenizer

    rows = [
        _row(SUM_D, plan=list(PLAN_WORDS)),
        _row(SUM_D),
        _row(2 * SUM_D, plan=list(PLAN_WORDS)),
    ]
    path = write_corpus(tmpdir / "plan.jsonl", rows)

    out = {
        "plan_start_id": reg.plan_start_id,
        "plan_end_id": reg.plan_end_id,
        "plan_word_sep_id": reg.plan_word_sep_id,
        "think_start_id": reg.think_start_id,
        "think_end_id": reg.think_end_id,
        "audio_start_id": reg.audio_start_id,
        "audio_end_id": reg.audio_end_id,
        "bos_id": reg.bos_id,
        "eos_id": reg.eos_id,
        "pitch_ids": [reg.pitch_bin_to_id(b) for b in range(config.PITCH_BINS)],
        "duration_ids": [reg.duration_bin_to_id(b) for b in range(config.DURATION_BINS)],
        "energy_ids": [reg.energy_bin_to_id(b) for b in range(config.ENERGY_BINS)],
        "prompt_ids": tok_.encode(PROMPT),
        "reasoning_ids": tok_.encode(REASONING),
        "tag_ids": tok_.encode("EMO=joyful PACE=fast PITCH=high"),
        "tag_ids_after_prose": tok_.encode(" EMO=joyful PACE=fast PITCH=high"),
        "no_word": NO_WORD,
    }

    def dump(ds, i):
        item = ds[i]
        d = {k: (v.tolist() if torch.is_tensor(v) else v) for k, v in item.items()}
        return d

    for mode in ("prose", "tags", "none", "plan", "tags+plan", "full"):
        ds = make_dataset(path, reg, mode)
        out[f"items_{mode}"] = [dump(ds, i) for i in range(len(ds))]
        out[f"counters_{mode}"] = {
            "seen": ds.n_rows_seen, "missing": ds.n_rows_missing_plan,
            "rejected": ds.n_rows_plan_rejected,
            "truncated": ds.n_rows_plan_truncated,
            "prose_dropped": ds.n_rows_reasoning_dropped,
        }
        out[f"audit_{mode}"] = ds.plan_audit

    ds_pack = make_dataset(path, reg, "plan", plan_schedule_mode="pack")
    out["items_plan_pack"] = [dump(ds_pack, i) for i in range(len(ds_pack))]

    ds_la0 = make_dataset(path, reg, "plan", plan_lookahead=0)
    out["item_plan_lookahead0"] = dump(ds_la0, 0)

    ds_w = make_dataset(path, reg, "full", reasoning_weight=0.3, tag_weight=1.0,
                        plan_weight=2.0)
    out["item_full_weighted"] = dump(ds_w, 0)

    from delay_dataset import DelayCollator
    ds_plan = make_dataset(path, reg, "plan")
    batch = DelayCollator(pad_token_id=reg.pad_id)([ds_plan[0], ds_plan[1], ds_plan[2]])
    out["batch_keys"] = sorted(batch.keys())
    out["batch"] = {k: batch[k].tolist() for k in
                    ("plan_schedule", "plan_slots", "plan_durations", "plan_lengths")}
    ds_none = make_dataset(path, reg, "none")
    out["batch_keys_plainarm"] = sorted(
        DelayCollator(pad_token_id=reg.pad_id)([ds_none[0], ds_none[1]]).keys())

    tight = make_dataset(path, reg, "full", max_seq_len=40)
    out["item_tight"] = dump(tight, 0)
    out["counters_tight"] = {"prose_dropped": tight.n_rows_reasoning_dropped,
                             "truncated": tight.n_rows_plan_truncated}
    tighter = make_dataset(path, reg, "full", max_seq_len=20)
    out["item_tighter"] = dump(tighter, 0)
    out["counters_tighter"] = {"truncated": tighter.n_rows_plan_truncated}

    rej = write_corpus(tmpdir / "rej.jsonl", [
        _row(SUM_D, plan={"words": list(PLAN_WORDS), "ok": False}),
        _row(SUM_D, plan=list(PLAN_WORDS)),
    ])
    ds_rej = make_dataset(rej, reg, "plan")
    out["rejected_audit"] = ds_rej.plan_audit
    out["rejected_item0_schedule"] = dump(ds_rej, 0)["plan_schedule"]
    out["rejected_counters"] = {"rejected": ds_rej.n_rows_plan_rejected,
                                "missing": ds_rej.n_rows_missing_plan}

    noplan = write_corpus(tmpdir / "noplan.jsonl", [_row(SUM_D), _row(SUM_D)])
    out["err_zero_coverage"] = _err(make_dataset, noplan, reg, "plan")
    out["err_stage1_plan_arm"] = _err(lambda: make_dataset(path, reg, "plan", stage=1))
    out["err_min_coverage"] = _err(
        lambda: make_dataset(path, reg, "plan", min_plan_coverage=0.95))

    bad_bin = write_corpus(tmpdir / "badbin.jsonl", [
        _row(SUM_D, plan=[{"pitch_bin": 3, "dur_bin": 99, "energy_bin": 0,
                           "dur_frames": 2}])])
    out["err_bad_dur_bin"] = _err(lambda: make_dataset(bad_bin, reg, "plan")[0])
    no_frames = write_corpus(tmpdir / "nodur.jsonl", [
        _row(SUM_D, plan=[{"pitch_bin": 3, "dur_bin": 1, "energy_bin": 0}])])
    out["err_no_dur_frames"] = _err(lambda: make_dataset(no_frames, reg, "plan")[0])
    out["err_no_dur_frames_msg"] = _msg(lambda: make_dataset(no_frames, reg, "plan")[0])
    dur_s = write_corpus(tmpdir / "durs.jsonl", [
        _row(SUM_D, plan=[{"pitch_bin": 3, "dur_bin": 1, "energy_bin": 0,
                           "dur_s": 0.5}])])
    out["dur_s_frames"] = dump(make_dataset(dur_s, reg, "plan"), 0)["plan_durations"]

    out["legacy_expected_prose"] = expected_legacy_ids(
        reg, tok_, REASONING, SUM_D)
    out["legacy_expected_tags"] = expected_legacy_ids(
        reg, tok_, "EMO=joyful PACE=fast PITCH=high", SUM_D)
    out["legacy_expected_none"] = expected_legacy_ids(reg, tok_, "", SUM_D)
    return out


def _err(fn, *args):
    try:
        fn(*args)
    except Exception as exc:
        return type(exc).__name__
    return None


def _msg(fn, *args):
    try:
        fn(*args)
    except Exception as exc:
        return str(exc)
    return ""

_CACHE: dict = {}


def _run_probe(**env_extra) -> dict:
    key = tuple(sorted(env_extra.items()))
    if key in _CACHE:
        return _CACHE[key]
    env = dict(os.environ)
    env.update(env_extra)
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve())],
        capture_output=True, text=True, cwd=str(REPO_ROOT), env=env)
    assert proc.returncode == 0, (
        f"probe subprocess failed (env={env_extra}):\n"
        f"{proc.stdout[-2000:]}\n{proc.stderr[-4000:]}")
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    _CACHE[key] = result
    return result


@pytest.fixture(scope="module")
def on() -> dict:
    return _run_probe(MVC_ENABLE_PLAN_TOKENS="1", MVC_NUM_SPEECH_TOKENS="64")


@pytest.fixture(scope="module")
def reg():
    import config
    if config.ENABLE_PLAN_TOKENS:
        pytest.skip("suite is running with MVC_ENABLE_PLAN_TOKENS=1; gate-off N/A")
    return fake_registry()


@pytest.fixture
def corpus(tmp_path):
    return write_corpus(tmp_path / "legacy.jsonl",
                        [_row(SUM_D), _row(SUM_D, reasoning="")])


class TestLegacyUnchanged:
    @pytest.mark.parametrize("mode,think_text", [
        ("prose", REASONING),
        ("tags", "EMO=joyful PACE=fast PITCH=high"),
        ("none", ""),
    ])
    def test_sequence_is_the_hand_written_answer(self, corpus, reg, mode, think_text):
        ds = make_dataset(corpus, reg, mode)
        got = ds[0]["input_ids"].tolist()
        assert got == expected_legacy_ids(reg, reg.tokenizer, think_text, SUM_D)

    def test_none_emits_no_think_block_at_all(self, corpus, reg):
        ids = make_dataset(corpus, reg, "none")[0]["input_ids"].tolist()
        assert reg.think_start_id not in ids
        assert reg.think_end_id not in ids

    def test_stage1_row_in_auto_mode_still_gets_no_think(self, corpus, reg):
        ids = make_dataset(corpus, reg, "prose")[1]["input_ids"].tolist()
        assert reg.think_start_id not in ids

    def test_legacy_loss_weights_unchanged(self, corpus, reg):
        item = make_dataset(corpus, reg, "prose", reasoning_weight=0.3)[0]
        w = item["loss_weights"].tolist()
        n_prompt = len(reg.tokenizer.encode(PROMPT))
        lo = 2 + n_prompt
        hi = lo + 1 + len(reg.tokenizer.encode(REASONING)) + 1
        assert w[lo:hi] == pytest.approx([0.3] * (hi - lo))
        assert set(w[:lo]) == {1.0} and set(w[hi:]) == {1.0}

    def test_no_plan_keys_leak_into_a_legacy_batch(self, corpus, reg):
        from delay_dataset import DelayCollator
        ds = make_dataset(corpus, reg, "prose")
        batch = DelayCollator(pad_token_id=reg.pad_id)([ds[0], ds[1]])
        assert not [k for k in batch if k.startswith("plan_")]
        assert set(batch) == {"input_ids", "labels", "codes", "attention_mask",
                              "loss_weights", "speaker_ids"}

    def test_item_exposes_plan_fields_as_none(self, corpus, reg):
        item = make_dataset(corpus, reg, "prose")[0]
        assert item["plan_schedule"] is None
        assert item["plan_slots"] is None
        assert item["plan_durations"] is None


class TestGateOffRefusesPlanArms:
    @pytest.mark.parametrize("mode", ["plan", "tags+plan", "full"])
    def test_plan_mode_without_the_vocabulary_is_fatal(self, corpus, reg, mode):
        with pytest.raises(RuntimeError, match="MVC_ENABLE_PLAN_TOKENS"):
            make_dataset(corpus, reg, mode)

    def test_unknown_cot_mode_still_rejected(self, corpus, reg):
        with pytest.raises(AssertionError):
            make_dataset(corpus, reg, "plann")


class TestSchedule:
    def test_stretch_tiles_the_span_exactly(self):
        from delay_dataset import build_plan_schedule
        assert build_plan_schedule([2, 4, 6], 12) == (
            [0] * 2 + [1] * 4 + [2] * 6)

    def test_stretch_dilates_proportionally(self):
        from delay_dataset import build_plan_schedule
        assert build_plan_schedule([2, 4, 6], 24) == (
            [0] * 4 + [1] * 8 + [2] * 12)

    def test_stretch_covers_every_frame_and_every_word(self):
        from delay_dataset import build_plan_schedule
        sched = build_plan_schedule([3, 1, 7, 2], 37)
        assert -1 not in sched and len(sched) == 37
        assert sorted(set(sched)) == [0, 1, 2, 3]
        assert sched == sorted(sched)

    def test_pack_is_reproducible_from_the_plan_alone(self):
        from delay_dataset import build_plan_schedule, NO_WORD
        assert build_plan_schedule([2, 4, 6], 24, "pack") == (
            [0] * 2 + [1] * 4 + [2] * 6 + [NO_WORD] * 12)

    def test_pack_clips_at_the_span_without_wrapping(self):
        from delay_dataset import build_plan_schedule
        assert build_plan_schedule([2, 4, 6], 5, "pack") == [0, 0, 1, 1, 1]

    def test_more_words_than_frames_does_not_crash(self):
        from delay_dataset import build_plan_schedule
        sched = build_plan_schedule([1, 1, 1, 1], 2)
        assert len(sched) == 2 and sched == sorted(sched)

    def test_empty_and_zero_cases(self):
        from delay_dataset import build_plan_schedule, NO_WORD
        assert build_plan_schedule([], 3) == [NO_WORD] * 3
        assert build_plan_schedule([1, 2], 0) == []

    def test_degenerate_durations_raise(self):
        from delay_dataset import build_plan_schedule
        with pytest.raises(ValueError):
            build_plan_schedule([0, 0], 10)
        with pytest.raises(ValueError):
            build_plan_schedule([1, -2], 10)

    def test_unknown_mode_raises(self):
        from delay_dataset import build_plan_schedule
        with pytest.raises(ValueError, match="schedule mode"):
            build_plan_schedule([1], 1, "squash")


class TestWordNormalisation:
    def _norm(self, words):
        from delay_dataset import normalise_plan_words
        return normalise_plan_words(words, n_pitch_bins=16, n_duration_bins=16,
                                    n_energy_bins=8)

    def test_unvoiced_sentinel_becomes_none(self):
        out = self._norm([{"pitch_bin": 16, "dur_bin": 0, "energy_bin": 0,
                           "dur_frames": 3}])
        assert out[0]["pitch_bin"] is None

    def test_out_of_range_bins_raise_with_coordinates(self):
        from delay_dataset import normalise_plan_words
        with pytest.raises(ValueError, match="word 1"):
            normalise_plan_words(
                [{"pitch_bin": 0, "dur_bin": 0, "energy_bin": 0, "dur_frames": 1},
                 {"pitch_bin": 0, "dur_bin": 0, "energy_bin": 9, "dur_frames": 1}],
                n_pitch_bins=16, n_duration_bins=16, n_energy_bins=8)

    def test_pitch_above_the_sentinel_raises(self):
        with pytest.raises(ValueError, match="pitch_bin"):
            self._norm([{"pitch_bin": 17, "dur_bin": 0, "energy_bin": 0,
                         "dur_frames": 1}])

    def test_seconds_are_converted_at_the_declared_frame_rate(self):
        from delay_dataset import normalise_plan_words
        out = normalise_plan_words(
            [{"pitch_bin": 0, "dur_bin": 0, "energy_bin": 0, "dur_s": 0.5}],
            n_pitch_bins=16, n_duration_bins=16, n_energy_bins=8, frame_rate=20.0)
        assert out[0]["dur_frames"] == 10


def _plan_span(on, item):
    ids = item["input_ids"]
    return ids.index(on["plan_start_id"]), ids.index(on["plan_end_id"]) + 1


class TestPlanAssembly:
    def test_legacy_arms_are_unchanged_with_the_gate_on(self, on):
        assert on["items_prose"][0]["input_ids"] == on["legacy_expected_prose"]
        assert on["items_tags"][0]["input_ids"] == on["legacy_expected_tags"]
        assert on["items_none"][0]["input_ids"] == on["legacy_expected_none"]

    def test_plan_block_is_exactly_the_program(self, on):
        item = on["items_plan"][0]
        lo, hi = _plan_span(on, item)
        expected = (
            [on["plan_start_id"]]
            + [on["plan_word_sep_id"], on["pitch_ids"][3],
               on["duration_ids"][1], on["energy_ids"][0]]
            + [on["plan_word_sep_id"], on["duration_ids"][4], on["energy_ids"][7]]
            + [on["plan_word_sep_id"], on["pitch_ids"][15],
               on["duration_ids"][9], on["energy_ids"][3]]
            + [on["plan_end_id"]])
        assert item["input_ids"][lo:hi] == expected

    def test_plan_only_arm_carries_no_prose_and_no_tags(self, on):
        item = on["items_plan"][0]
        ids = item["input_ids"]
        i = ids.index(on["think_start_id"])
        assert ids[i + 1] == on["plan_start_id"]
        assert ids[ids.index(on["plan_end_id"]) + 1] == on["think_end_id"]
        for tid in on["reasoning_ids"] + on["tag_ids"]:
            assert tid not in ids

    def test_tier_order_is_prose_then_tags_then_plan(self, on):
        ids = on["items_full"][0]["input_ids"]
        i_think = ids.index(on["think_start_id"])
        i_prose = ids.index(on["reasoning_ids"][0])
        i_tags = ids.index(on["tag_ids_after_prose"][0])
        i_plan = ids.index(on["plan_start_id"])
        i_end = ids.index(on["think_end_id"])
        assert i_think < i_prose < i_tags < i_plan < i_end

    def test_tags_plus_plan_has_no_prose(self, on):
        ids = on["items_tags+plan"][0]["input_ids"]
        assert ids[ids.index(on["think_start_id"]) + 1] == on["tag_ids"][0]
        for tid in on["reasoning_ids"]:
            assert tid not in ids

    def test_audio_content_is_identical_across_arms(self, on):
        base = on["items_none"][0]["codes"]
        for mode in ("prose", "tags", "plan", "tags+plan", "full"):
            item = on[f"items_{mode}"][0]
            codes = [c for c in item["codes"] if c[0] != -1]
            assert codes == [c for c in base if c[0] != -1], mode


class TestPerTierLossWeights:
    def test_each_tier_lands_on_exactly_its_own_positions(self, on):
        item = on["item_full_weighted"]
        w = item["loss_weights"]
        ids = item["input_ids"]
        lo = ids.index(on["think_start_id"])
        n_prose = len(on["reasoning_ids"])
        n_tags = len(on["tag_ids_after_prose"])
        p_lo, p_hi = _plan_span(on, item)
        assert w[lo] == pytest.approx(0.3)
        assert w[lo + 1: lo + 1 + n_prose] == pytest.approx([0.3] * n_prose)
        assert w[lo + 1 + n_prose: p_lo] == [1.0] * n_tags
        assert w[p_lo:p_hi] == [2.0] * (p_hi - p_lo)
        assert w[p_hi] == pytest.approx(0.3)
        assert set(w[:lo]) == {1.0} and set(w[p_hi + 1:]) == {1.0}

    def test_tag_weight_defaults_to_the_reasoning_weight(self, on):
        item = on["items_tags"][0]
        w = item["loss_weights"]
        lo = item["input_ids"].index(on["think_start_id"])
        n = 1 + len(on["tag_ids"]) + 1
        assert w[lo:lo + n] == pytest.approx([0.3] * n)

    def test_plan_tokens_are_supervised_targets(self, on):
        item = on["items_plan"][0]
        lo, hi = _plan_span(on, item)
        assert item["labels"][lo:hi] == item["input_ids"][lo:hi]


class TestScheduleAlignment:
    def test_position_to_word_map_is_exact(self, on):
        item = on["items_plan"][0]
        sched = item["plan_schedule"]
        a0 = item["audio_start"]
        expected_frames = [0] * 2 + [1] * 4 + [2] * 6
        for t, word in enumerate(expected_frames):
            assert sched[a0 + t - 1] == word, f"frame {t}"
        assert set(sched[: a0 - 1]) == {on["no_word"]}
        assert set(sched[a0 + len(expected_frames) - 1:]) == {on["no_word"]}

    def test_frame_zero_is_scheduled_on_the_audio_token(self, on):
        item = on["items_plan"][0]
        a0 = item["audio_start"]
        assert item["input_ids"][a0 - 1] == on["audio_start_id"]
        assert item["plan_schedule"][a0 - 1] == 0

    def test_lookahead_zero_shifts_by_exactly_one(self, on):
        shifted = on["item_plan_lookahead0"]["plan_schedule"]
        base = on["items_plan"][0]["plan_schedule"]
        assert shifted[1:] == base[:-1]

    def test_stretch_dilates_when_the_clip_is_longer_than_the_program(self, on):
        item = on["items_plan"][2]
        a0 = item["audio_start"]
        expected = [0] * 4 + [1] * 8 + [2] * 12
        assert item["plan_schedule"][a0 - 1: a0 - 1 + 24] == expected

    def test_pack_leaves_the_tail_unscheduled(self, on):
        item = on["items_plan_pack"][2]
        a0 = item["audio_start"]
        expected = [0] * 2 + [1] * 4 + [2] * 6 + [on["no_word"]] * 12
        assert item["plan_schedule"][a0 - 1: a0 - 1 + 24] == expected

    def test_slots_point_at_the_words_own_plan_tokens(self, on):
        item = on["items_plan"][0]
        ids = item["input_ids"]
        slots = item["plan_slots"]
        assert [ids[s[0]] for s in slots if s[0] != -1] == [
            on["pitch_ids"][3], on["pitch_ids"][15]]
        assert slots[1][0] == -1
        assert [ids[s[1]] for s in slots] == [on["duration_ids"][b]
                                              for b in (1, 4, 9)]
        assert [ids[s[2]] for s in slots] == [on["energy_ids"][b]
                                              for b in (0, 7, 3)]

    def test_durations_match_the_sidecar(self, on):
        assert on["items_plan"][0]["plan_durations"] == [2, 4, 6]

    def test_seconds_fall_back_at_the_declared_frame_rate(self, on):
        assert on["dur_s_frames"] == [11]


class TestMissingPlansAreVisible:
    def test_a_plan_less_row_degrades_to_the_stage1_layout(self, on):
        item = on["items_plan"][1]
        assert on["think_start_id"] not in item["input_ids"]
        assert item["plan_schedule"] is None
        assert item["plan_slots"] is None

    def test_the_degradation_is_counted(self, on):
        assert on["counters_plan"]["missing"] == 1
        assert on["counters_plan"]["seen"] == 3

    def test_construction_audits_coverage(self, on):
        audit = on["audit_plan"]
        assert audit["scanned"] == 3 and audit["with_plan"] == 2
        assert abs(audit["coverage"] - 2 / 3) < 1e-9
        assert audit["mean_words"] == 3.0
        assert audit["sampled"] is False

    def test_legacy_arms_run_no_audit(self, on):
        for mode in ("prose", "tags", "none"):
            assert on[f"audit_{mode}"] is None

    def test_a_corpus_with_no_plans_at_all_is_fatal(self, on):
        assert on["err_zero_coverage"] == "RuntimeError"

    def test_min_coverage_is_enforceable(self, on):
        assert on["err_min_coverage"] == "RuntimeError"

    def test_a_plan_arm_pinned_to_stage_1_is_fatal(self, on):
        assert on["err_stage1_plan_arm"] == "ValueError"

    def test_a_rejected_sidecar_is_dropped_not_used(self, on):
        assert on["rejected_item0_schedule"] is None
        assert on["rejected_counters"]["rejected"] == 1
        assert on["rejected_audit"]["rejected"] == 1
        assert on["rejected_audit"]["with_plan"] == 1

    def test_malformed_plans_raise_loudly(self, on):
        assert on["err_bad_dur_bin"] == "ValueError"
        assert on["err_no_dur_frames"] == "ValueError"
        assert "dur_frames" in on["err_no_dur_frames_msg"]
        assert "plan_extract" in on["err_no_dur_frames_msg"]


class TestTruncationPolicy:
    def test_reasoning_goes_first(self, on):
        item = on["item_tight"]
        ids = item["input_ids"]
        assert on["plan_start_id"] in ids, "the plan was dropped before the prose"
        for tid in on["reasoning_ids"]:
            assert tid not in ids
        assert on["counters_tight"]["prose_dropped"] == 1

    def test_the_window_is_respected(self, on):
        assert len(on["item_tight"]["input_ids"]) <= 40
        assert len(on["item_tighter"]["input_ids"]) <= 20

    def test_a_words_program_is_never_split(self, on):
        item = on["item_tighter"]
        ids = item["input_ids"]
        lo, hi = _plan_span(on, item)
        body = ids[lo + 1: hi - 1]
        assert body, "the whole program vanished"
        assert body[0] == on["plan_word_sep_id"]
        groups, cur = [], []
        for tid in body:
            if tid == on["plan_word_sep_id"]:
                if cur:
                    groups.append(cur)
                cur = []
            else:
                cur.append(tid)
        groups.append(cur)
        for g in groups:
            assert len(g) in (2, 3), f"partial word {g}"
            assert g[-2] in on["duration_ids"] and g[-1] in on["energy_ids"]
            if len(g) == 3:
                assert g[0] in on["pitch_ids"]

    def test_truncation_is_counted(self, on):
        assert on["counters_tighter"]["truncated"] >= 1

    def test_the_schedule_still_covers_only_surviving_words(self, on):
        item = on["item_tighter"]
        n_words = len(item["plan_slots"])
        assert max(item["plan_schedule"]) < n_words
        assert len(item["plan_durations"]) == n_words


class TestCollator:
    def test_plan_keys_appear_only_when_a_plan_is_present(self, on):
        assert not [k for k in on["batch_keys_plainarm"] if k.startswith("plan_")]
        for key in ("plan_schedule", "plan_slots", "plan_durations", "plan_lengths"):
            assert key in on["batch_keys"]

    def test_padding_never_looks_like_word_zero(self, on):
        sched = on["batch"]["plan_schedule"]
        assert len(set(len(r) for r in sched)) == 1, "ragged batch"
        assert set(sched[1]) == {-1}, "the plan-less row must be all NO_WORD"
        assert sched[0][-1] == -1

    def test_slots_and_durations_pad_correctly(self, on):
        slots = on["batch"]["plan_slots"]
        durations = on["batch"]["plan_durations"]
        lengths = on["batch"]["plan_lengths"]
        assert lengths == [3, 0, 3]
        assert slots[1] == [[-1, -1, -1]] * 3
        assert durations[1] == [0, 0, 0]
        assert durations[0] == [2, 4, 6]
        assert sum(durations[1]) == 0

    def test_schedule_rows_align_with_the_padded_input(self, on):
        sched = on["batch"]["plan_schedule"]
        item0 = on["items_plan"][0]
        a0 = item0["audio_start"]
        assert sched[0][a0 - 1] == 0
        assert sched[0][: a0 - 1] == [-1] * (a0 - 1)

if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        print(json.dumps(probe(td)))
