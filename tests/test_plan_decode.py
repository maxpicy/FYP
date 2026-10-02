# test_plan_decode.py: Plan parsing, the plan window and the injection schedule at decode time.

import importlib.util
import json
import pathlib
import sys
import types

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name):
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "scripts"))
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def g():
    try:
        return _load("gen_fish_cot_samples")
    except Exception as exc:
        pytest.skip(f"generator not importable here: {exc}")

START, END, PW = 1000, 1001, 1002
PITCH0, DUR0, ENERGY0 = 1010, 1030, 1050
N_PITCH, N_DUR, N_ENERGY = 16, 16, 8

THINK_START, THINK_END = 900, 901


def maps(g):
    return g.PlanIdMaps(
        start=START, end=END, word=PW,
        pitch={PITCH0 + b: b for b in range(N_PITCH)},
        duration={DUR0 + b: b for b in range(N_DUR)},
        energy={ENERGY0 + b: b for b in range(N_ENERGY)},
    )


def word_ids(pitch, dur, energy):
    ids = [PW]
    if pitch is not None:
        ids.append(PITCH0 + pitch)
    return ids + [DUR0 + dur, ENERGY0 + energy]


def block(*words, terminated=True):
    ids = [START]
    for w in words:
        ids += word_ids(*w)
    return ids + ([END] if terminated else [])


class FakeReg:
    think_start_id = THINK_START
    think_end_id = THINK_END
    plan_start_id = START
    plan_end_id = END
    plan_word_sep_id = PW
    plan_enabled = True
    pitch_ids = [PITCH0 + b for b in range(N_PITCH)]
    duration_ids = [DUR0 + b for b in range(N_DUR)]
    energy_ids = [ENERGY0 + b for b in range(N_ENERGY)]

    def pitch_bin_to_id(self, b):
        return self.pitch_ids[b]

    def duration_bin_to_id(self, b):
        return self.duration_ids[b]

    def energy_bin_to_id(self, b):
        return self.energy_ids[b]


def test_wellformed_plan_parses_to_exact_bins(g):
    ids = block((7, 4, 3), (None, 6, 5), (0, 15, 0))
    p = g.parse_plan_tokens(ids, maps(g))
    assert p.status == g.PLAN_STATUS_OK and p.ok
    assert p.reason == ""
    assert p.n_words == 3
    assert p.words == [
        {"pitch_bin": 7, "dur_bin": 4, "energy_bin": 3},
        {"pitch_bin": None, "dur_bin": 6, "energy_bin": 5},
        {"pitch_bin": 0, "dur_bin": 15, "energy_bin": 0},
    ]


def test_absent_plan_is_its_own_status(g):
    p = g.parse_plan_tokens([42, 43, 44], maps(g))
    assert p.status == g.PLAN_STATUS_ABSENT and not p.ok
    assert p.words == []


def test_empty_plan_block(g):
    p = g.parse_plan_tokens([START, END], maps(g))
    assert p.status == g.PLAN_STATUS_EMPTY and not p.ok


def test_unterminated_plan_keeps_its_words_but_is_not_ok(g):
    ids = block((3, 2, 1), (4, 3, 2), terminated=False)
    p = g.parse_plan_tokens(ids, maps(g))
    assert p.status == g.PLAN_STATUS_UNTERMINATED and not p.ok
    assert p.n_words == 2, "diagnostic words were dropped"
    assert "never closed" in p.reason


def test_word_missing_its_energy_token_is_malformed(g):
    ids = [START, PW, PITCH0 + 3, DUR0 + 2, END]
    p = g.parse_plan_tokens(ids, maps(g))
    assert p.status == g.PLAN_STATUS_MALFORMED and not p.ok
    assert "word 0" in p.reason and "[E:*]" in p.reason


def test_word_missing_its_duration_token_is_malformed(g):
    ids = [START, PW, PITCH0 + 3, ENERGY0 + 1, END]
    p = g.parse_plan_tokens(ids, maps(g))
    assert p.status == g.PLAN_STATUS_MALFORMED
    assert "[D:*]" in p.reason and "energy bin 1" in p.reason


def test_stray_text_token_inside_the_block_is_malformed(g):
    ids = [START] + word_ids(1, 1, 1) + [7777] + word_ids(2, 2, 2) + [END]
    p = g.parse_plan_tokens(ids, maps(g))
    assert p.status == g.PLAN_STATUS_MALFORMED
    assert "word 1" in p.reason and "non-plan token" in p.reason
    assert p.n_words == 1, "words before the failure should survive for triage"


def test_plan_not_starting_with_a_word_separator_is_malformed(g):
    p = g.parse_plan_tokens([START, DUR0 + 1, ENERGY0 + 1, END], maps(g))
    assert p.status == g.PLAN_STATUS_MALFORMED
    assert "expected [PW]" in p.reason


def test_two_plan_blocks_are_malformed(g):
    ids = block((1, 1, 1)) + block((2, 2, 2))
    p = g.parse_plan_tokens(ids, maps(g))
    assert p.status == g.PLAN_STATUS_MALFORMED
    assert "more than one <PLAN>" in p.reason


def test_every_failure_mode_has_a_distinct_status(g):
    cases = {
        g.parse_plan_tokens([42], maps(g)).status,
        g.parse_plan_tokens([START, END], maps(g)).status,
        g.parse_plan_tokens(block((1, 1, 1), terminated=False), maps(g)).status,
        g.parse_plan_tokens([START, DUR0, ENERGY0, END], maps(g)).status,
        g.parse_plan_tokens(block((1, 1, 1)), maps(g)).status,
    }
    assert len(cases) == 5


def test_plan_region_reads_the_last_think_block_only(g):
    reg = FakeReg()
    ids = ([5, 6, THINK_START, 7, THINK_END, 8]
           + [THINK_START] + block((2, 3, 4)) + [THINK_END, 99])
    body = g.plan_region(ids, reg)
    assert body == block((2, 3, 4))
    assert g.parse_plan_tokens(body, maps(g)).ok


def test_plan_region_is_empty_without_a_think_block(g):
    assert g.plan_region([1, 2, 3], FakeReg()) == []


def test_plan_region_tolerates_an_unclosed_think_block(g):
    ids = [THINK_START] + block((1, 2, 3), terminated=False)
    assert g.plan_region(ids, FakeReg())[0] == START


def test_given_plan_round_trips_through_the_prefix(g):
    words = [{"pitch_bin": 7, "dur_bin": 4, "energy_bin": 3},
             {"pitch_bin": None, "dur_bin": 9, "energy_bin": 1}]
    ids = g.plan_prefix_ids(FakeReg(), words)
    assert ids[0] == START and ids[-1] == END
    assert ids == [START, PW, PITCH0 + 7, DUR0 + 4, ENERGY0 + 3,
                   PW, DUR0 + 9, ENERGY0 + 1, END]
    back = g.parse_plan_tokens(ids, maps(g))
    assert back.ok and back.words == words


def test_dur_frames_are_the_extractor_bin_centres(g):
    pe = _load("plan_extract")
    words = [{"pitch_bin": 3, "dur_bin": 4, "energy_bin": 2},
             {"pitch_bin": None, "dur_bin": 4, "energy_bin": 1},
             {"pitch_bin": 5, "dur_bin": 6, "energy_bin": 0}]
    got = g.plan_dur_frames(words)
    assert got == pytest.approx([4.9749, 4.9749, 7.4066], abs=1e-3)
    assert g.plan_sum_frames(words) == pytest.approx(17.356, abs=1e-3)
    tokens = pe.plan_to_tokens([{"pitch_bin": 3, "dur_bin": 4, "energy_bin": 2},
                                {"pitch_bin": pe.PITCH_UNVOICED_BIN,
                                 "dur_bin": 4, "energy_bin": 1},
                                {"pitch_bin": 5, "dur_bin": 6, "energy_bin": 0}])
    assert round(g.plan_sum_frames(words)) == pe.plan_frames(tokens)


def test_dur_frames_are_monotone_in_the_bin(g):
    frames = [g.plan_dur_frames([{"pitch_bin": 0, "dur_bin": b,
                                  "energy_bin": 0}])[0] for b in range(16)]
    assert all(a < b for a, b in zip(frames, frames[1:]))


def test_unvoiced_word_still_contributes_its_duration(g):
    voiced = [{"pitch_bin": 3, "dur_bin": 8, "energy_bin": 2}]
    unvoiced = [{"pitch_bin": None, "dur_bin": 8, "energy_bin": 2}]
    assert g.plan_sum_frames(voiced) == g.plan_sum_frames(unvoiced)


def test_plan_window_arithmetic_known_answer(g):
    parsed = g.ParsedPlan([{"pitch_bin": 2, "dur_bin": 10, "energy_bin": 1}] * 10,
                          g.PLAN_STATUS_OK, "")
    total = g.plan_sum_frames(parsed.words)
    assert total == pytest.approx(163.793, abs=1e-2)
    w = g.resolve_decode_window(parsed, 20, 400, "words", dur_lo=0.9,
                                dur_hi=1.5, use_plan=True)
    assert (w.min_fr, w.max_fr) == (147, 245)
    assert w.source == "plan" and not w.fallback
    assert w.sum_frames == pytest.approx(total)


def test_plan_window_keeps_the_floor_and_the_min_plus_5_guard(g):
    parsed = g.ParsedPlan([{"pitch_bin": 0, "dur_bin": 4, "energy_bin": 0}] * 3,
                          g.PLAN_STATUS_OK, "")
    w = g.resolve_decode_window(parsed, 20, 400, "words", dur_lo=0.9,
                                dur_hi=1.5, use_plan=True)
    assert w.min_fr == 20
    assert w.max_fr == 25
    assert w.max_fr > w.min_fr


def test_window_matches_the_duration_model_arithmetic_exactly(g, monkeypatch):
    src = (ROOT / "scripts" / "gen_fish_cot_samples.py").read_text(
        encoding="utf-8")
    assert "min_fr = max(20, int(args.dur_lo * t_hat))" in src
    assert "max_fr = max(min_fr + 5, int(args.dur_hi * t_hat))" in src
    for t_hat, lo, hi in [(97.4, 0.6, 1.5), (250.0, 0.9, 1.5), (12.0, 1.0, 2.0)]:
        ref_min = max(20, int(lo * t_hat))
        ref_max = max(ref_min + 5, int(hi * t_hat))
        monkeypatch.setattr(g, "plan_sum_frames", lambda _w, _t=t_hat: _t)
        w = g.resolve_decode_window(
            types.SimpleNamespace(ok=True, words=None), 20, 400, "words",
            dur_lo=lo, dur_hi=hi, use_plan=True)
        assert (w.min_fr, w.max_fr) == (ref_min, ref_max)


@pytest.mark.parametrize("status", ["absent", "empty", "unterminated",
                                    "malformed"])


def test_unusable_plan_falls_back_visibly(g, status):
    parsed = g.ParsedPlan([], status, "because")
    w = g.resolve_decode_window(parsed, 33, 300, "duration_model", dur_lo=0.9,
                                dur_hi=1.5, use_plan=True)
    assert (w.min_fr, w.max_fr) == (33, 300), "the old window must be restored"
    assert w.fallback is True, "the fall back must be visible on the row"
    assert w.source == "duration_model", "the row must name the window it used"
    assert w.sum_frames is None


def test_plan_off_is_not_a_fallback(g):
    w = g.resolve_decode_window(None, 33, 300, "words", dur_lo=0.9, dur_hi=1.5,
                                use_plan=False)
    assert (w.min_fr, w.max_fr, w.source) == (33, 300, "words")
    assert w.fallback is False


def test_the_record_carries_the_plan_and_the_run_summarises_it():
    src = (ROOT / "scripts" / "gen_fish_cot_samples.py").read_text(
        encoding="utf-8")
    for key in ('"plan_status": parsed.status', '"plan_ok": parsed.ok',
                '"plan_reason": parsed.reason', '"plan_words"',
                '"plan_sum_frames"', '"plan_window_fallback": win.fallback',
                '"plan_injected"', '"dur_window_source": win.source'):
        assert key in src, f"the row no longer records {key}"
    assert "**plan_extra" in src, "the plan fields are not written to the record"
    assert "PLAN_STATUS_SUMMARY" in src, "the run no longer summarises plan health"
    assert "plan_extra, plan_bins, plan_sched = {}, None, None" in src


def test_think_budget_is_fixed_when_given_and_auto_scales_with_words(g):
    assert g.think_budget(120, "a b c") == 120
    assert g.think_budget(200, " ".join(["w"] * 60)) == 200
    assert g.think_budget(0, "one two") == 120
    assert g.think_budget(0, " ".join(["w"] * 38)) == 6 * 38 + 64
    assert g.think_budget(0, " ".join(["w"] * 107)) == 6 * 107 + 64
    assert g.think_budget(None, "") == 120


def test_plan_flags_are_all_default_off():
    src = (ROOT / "scripts" / "gen_fish_cot_samples.py").read_text(
        encoding="utf-8")
    for flag in ("--plan_window", "--plan_injection"):
        i = src.index(f'"{flag}"')
        assert 'action="store_true"' in src[i:i + 120], f"{flag} is not off by default"
    assert '"--force_plan", default=None' in src
    assert '"--plan_dur_lo", type=float, default=0.8' in src
    assert '"--plan_dur_hi", type=float, default=1.5' in src


def test_plan_words_from_a_token_string(g):
    words = g.plan_words_from_obj("[PW][P:7][D:4][E:3][PW][D:6][E:5]",
                                  where="t")
    assert words == [{"pitch_bin": 7, "dur_bin": 4, "energy_bin": 3},
                     {"pitch_bin": None, "dur_bin": 6, "energy_bin": 5}]


def test_plan_words_from_a_sidecar_object(g):
    obj = {"ok": True, "words": [{"pitch_bin": 1, "dur_bin": 2,
                                  "energy_bin": 3, "dur_frames": 9}]}
    assert g.plan_words_from_obj(obj, where="t") == [
        {"pitch_bin": 1, "dur_bin": 2, "energy_bin": 3}]


def test_the_compact_form_this_script_writes_is_readable_back(g):
    parsed = g.parse_plan_tokens(block((7, 4, 3), (None, 6, 5)), maps(g))
    compact = [[w["pitch_bin"], w["dur_bin"], w["energy_bin"]]
               for w in parsed.words]
    assert compact == [[7, 4, 3], [None, 6, 5]]
    assert g.plan_words_from_obj(compact, where="t") == parsed.words
    assert g.plan_words_from_obj({"plan": compact}, where="t") == parsed.words
    with pytest.raises(SystemExit) as e:
        g.plan_words_from_obj([[1, 2]], where="t")
    assert "pitch, duration, energy" in str(e.value)


def test_a_rejected_sidecar_is_refused(g):
    obj = {"ok": False, "flags": ["low_coverage"], "words": [
        {"pitch_bin": 1, "dur_bin": 2, "energy_bin": 3}]}
    with pytest.raises(SystemExit) as e:
        g.plan_words_from_obj(obj, where="row 4")
    assert "REJECTED" in str(e.value)


def test_out_of_range_bins_are_refused(g):
    with pytest.raises(SystemExit) as e:
        g.plan_words_from_obj([{"pitch_bin": 0, "dur_bin": 99, "energy_bin": 0}],
                              where="row 4")
    assert "dur_bin=99" in str(e.value)
    with pytest.raises(SystemExit):
        g.plan_words_from_obj([{"dur_bin": 1}], where="row 4")


def test_unvoiced_sentinel_normalises_to_none(g):
    from config import PITCH_BINS
    w = g.plan_words_from_obj([{"pitch_bin": PITCH_BINS, "dur_bin": 1,
                                "energy_bin": 1}], where="t")
    assert w[0]["pitch_bin"] is None


def test_forced_plans_bind_by_row_index(g, tmp_path):
    p = tmp_path / "forced.jsonl"
    p.write_text("\n".join(json.dumps(d) for d in [
        {"row": 7, "plan": "[PW][P:1][D:1][E:1]"},
        {"row": 3, "plan": "[PW][P:2][D:2][E:2]"}]) + "\n", encoding="utf-8")
    forced = g.load_forced_plans(str(p))
    assert g.forced_plan_for(forced, 3, 0, str(p))["plan"].endswith("[E:2]")
    with pytest.raises(SystemExit) as e:
        g.forced_plan_for(forced, 5, 1, str(p))
    assert "no entry for val row 5" in str(e.value)


def test_forced_plans_fall_back_to_file_order(g, tmp_path):
    p = tmp_path / "forced.jsonl"
    p.write_text("\n".join(json.dumps({"plan": f"[PW][P:{i}][D:1][E:1]"})
                           for i in range(2)) + "\n", encoding="utf-8")
    forced = g.load_forced_plans(str(p))
    assert g.forced_plan_for(forced, 99, 1, str(p))["plan"].startswith("[PW][P:1]")
    with pytest.raises(SystemExit) as e:
        g.forced_plan_for(forced, 99, 2, str(p))
    assert "holds 2 plans" in str(e.value)


def test_forced_plan_file_refuses_ambiguity(g, tmp_path):
    dup = tmp_path / "dup.jsonl"
    dup.write_text(json.dumps({"row": 1, "plan": "[PW][D:1][E:1]"}) + "\n"
                   + json.dumps({"row": 1, "plan": "[PW][D:2][E:1]"}) + "\n",
                   encoding="utf-8")
    with pytest.raises(SystemExit) as e:
        g.load_forced_plans(str(dup))
    assert "repeats row 1" in str(e.value)

    mixed = tmp_path / "mixed.jsonl"
    mixed.write_text(json.dumps({"row": 1, "plan": "[PW][D:1][E:1]"}) + "\n"
                     + json.dumps({"plan": "[PW][D:2][E:1]"}) + "\n",
                     encoding="utf-8")
    with pytest.raises(SystemExit) as e:
        g.load_forced_plans(str(mixed))
    assert "mixes rows with and without" in str(e.value)


def test_training_still_uses_the_lookahead_convention_this_file_assumes():
    src = (ROOT / "delay_dataset.py").read_text(encoding="utf-8")
    assert "pos = audio_start + t - self.plan_lookahead" in src
    assert "plan_lookahead: int = 1" in src, "the default lookahead moved"
    assert "audio_start = think_hi + 1" in src


def test_decode_schedule_places_frame_zero_on_the_audio_token():
    pytest.importorskip("mamba_ssm", reason="model.py needs mamba_ssm")
    tdc = _load("test_depth_checkpoint")
    s = tdc.build_decode_plan_schedule([2, 3], prefix_len=5, max_frames=5,
                                       mode="pack")
    assert s.shape == (1, 10)
    assert s[0].tolist() == [-1, -1, -1, -1, 0, 0, 1, 1, 1, -1]


def test_decode_schedule_known_answers():
    pytest.importorskip("mamba_ssm", reason="model.py needs mamba_ssm")
    tdc = _load("test_depth_checkpoint")
    s = tdc.build_decode_plan_schedule([2, 2], prefix_len=1, max_frames=6,
                                       mode="pack")
    assert s[0].tolist() == [0, 0, 1, 1, 1, 1, -1]
    s = tdc.build_decode_plan_schedule([2.5, 1.0], prefix_len=1, max_frames=4,
                                       mode="pack")
    assert s[0].tolist() == [0, 0, 0, 1, -1]
    s = tdc.build_decode_plan_schedule([1, 1], prefix_len=1, max_frames=6,
                                       mode="stretch", target_frames=6)
    assert s[0].tolist() == [0, 0, 0, 1, 1, 1, -1]


def test_decode_schedule_refuses_what_it_cannot_do():
    pytest.importorskip("mamba_ssm", reason="model.py needs mamba_ssm")
    tdc = _load("test_depth_checkpoint")
    with pytest.raises(ValueError, match="target_frames"):
        tdc.build_decode_plan_schedule([2, 2], prefix_len=3, max_frames=5,
                                       mode="stretch")
    with pytest.raises(ValueError, match="no words"):
        tdc.build_decode_plan_schedule([], prefix_len=3, max_frames=5,
                                       mode="pack")
    with pytest.raises(ValueError, match="sum to 0"):
        tdc.build_decode_plan_schedule([0, 0], prefix_len=3, max_frames=5,
                                       mode="pack")
    with pytest.raises(ValueError):
        tdc.build_decode_plan_schedule([2], prefix_len=0, max_frames=5,
                                       mode="pack")


def test_decode_schedule_matches_the_training_scheduler_frame_for_frame():
    pytest.importorskip("mamba_ssm", reason="model.py needs mamba_ssm")
    tdc = _load("test_depth_checkpoint")
    from delay_dataset import build_plan_schedule as train_schedule

    durs, T, prefix_len = [3, 1, 4, 2], 17, 6
    ref = train_schedule(durs, T, "stretch")
    s = tdc.build_decode_plan_schedule(durs, prefix_len=prefix_len,
                                       max_frames=T, mode="stretch",
                                       target_frames=T)[0].tolist()
    for t, w in enumerate(ref):
        assert s[prefix_len - 1 + t] == w, f"frame {t} scheduled to the wrong word"
    assert all(v == -1 for v in s[:prefix_len - 1]), "injection reached the prefix"


def test_depth_generate_refuses_half_a_plan():
    pytest.importorskip("mamba_ssm", reason="model.py needs mamba_ssm")
    tdc = _load("test_depth_checkpoint")
    import torch
    reg = types.SimpleNamespace(audio_start_id=1, audio_end_id=2, eos_id=3)
    model = types.SimpleNamespace(backbone=types.SimpleNamespace(
        backbone=types.SimpleNamespace(embedding=None)))
    with pytest.raises(ValueError, match="together"):
        tdc.depth_generate(model, torch.zeros(1, 3, dtype=torch.long), reg,
                           plan_bins=torch.zeros(1, 2, 3, dtype=torch.long))


def test_plan_injection_checks_catch_the_silent_failures():
    pytest.importorskip("mamba_ssm", reason="model.py needs mamba_ssm")
    tdc = _load("test_depth_checkpoint")
    import torch

    bins = torch.zeros(1, 2, 3, dtype=torch.long)
    good = tdc.build_decode_plan_schedule([2, 2], prefix_len=4, max_frames=4,
                                          mode="pack")

    off = types.SimpleNamespace(use_plan_injection=False, training=False)
    with pytest.raises(RuntimeError, match="enable_plan_injection"):
        tdc._check_plan_decode(off, bins, good, 4, 4)

    training = types.SimpleNamespace(use_plan_injection=True, training=True)
    with pytest.raises(RuntimeError, match="eval"):
        tdc._check_plan_decode(training, bins, good, 4, 4)

    ok = types.SimpleNamespace(use_plan_injection=True, training=False)
    tdc._check_plan_decode(ok, bins, good, 4, 4)

    with pytest.raises(ValueError, match="before the <AUDIO> token"):
        tdc._check_plan_decode(ok, bins, good, 6, 2)
    with pytest.raises(ValueError, match="covers"):
        tdc._check_plan_decode(ok, bins, good, 4, 99)
    late = tdc.build_decode_plan_schedule([2, 2], prefix_len=5, max_frames=4,
                                          mode="pack")
    with pytest.raises(ValueError, match="off by at least one"):
        tdc._check_plan_decode(ok, bins, late[:, :9], 4, 4)


def test_the_last_position_always_carries_the_current_frames_word():
    pytest.importorskip("mamba_ssm", reason="model.py needs mamba_ssm")
    import torch
    from model import PlanInjector
    tdc = _load("test_depth_checkpoint")

    prefix_len, max_frames = 6, 8
    durs = [2, 3, 3]
    sched = tdc.build_decode_plan_schedule(durs, prefix_len=prefix_len,
                                           max_frames=max_frames, mode="pack")
    bins = torch.tensor([[[1, 2, 3], [4, 5, 6], [7, 0, 1]]], dtype=torch.long)
    inj = PlanInjector(d_model=8, n_pitch_bins=16, n_duration_bins=16,
                       n_energy_bins=8, d_plan=4, scaffold_dropout=0.0)
    torch.nn.init.normal_(inj.proj.weight, std=0.5)
    inj.eval()

    words = inj.word_vectors(bins)
    expected_frame_word = [0, 0, 1, 1, 1, 2, 2, 2]
    for t in range(max_frames):
        length = prefix_len + t
        out = inj(bins, sched[:, :length])
        w = expected_frame_word[t]
        assert torch.allclose(out[0, -1], inj.proj(words[0, w])), \
            f"frame {t} was injected with the wrong word"
        assert torch.count_nonzero(out[0, :prefix_len - 1]) == 0

_E2E = r'''
import os, sys
os.environ["MVC_NUM_SPEECH_TOKENS"] = "2048"
os.environ["MVC_NUM_CODEC_LEVELS"] = "10"
os.environ["MVC_ENABLE_PLAN_TOKENS"] = "1"
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import torch
from model import MambaCoTModel
import gen_fish_cot_samples as G
import test_depth_checkpoint as tdc

m = MambaCoTModel(model_name="state-spaces/mamba2-130m", device="cuda",
                  dtype=torch.float32, mtp_num_heads=0)
m.enable_speaker_conditioning(speaker_dim=32, num_speakers=16, input_injection=True)
m.enable_depth_module(num_levels=10, codebook_size=2048, d_depth=64, depth_layers=1,
                      depth_feedback="semantic", cross_frame=0,
                      codebook_sizes=G.FISH_CB, depth_arch="gru", depth_cond="add",
                      depth_heads=2)
m.enable_plan_injection(d_plan=16, scaffold_dropout=0.0)
m.eval()
reg = m.token_registry
assert reg.plan_enabled

words = [{"pitch_bin": 7, "dur_bin": 4, "energy_bin": 3},
         {"pitch_bin": None, "dur_bin": 6, "energy_bin": 5},
         {"pitch_bin": 2, "dur_bin": 5, "energy_bin": 1}]
prefix, rtext = G.build_prefix(
    "oracle", m, reg, {"prompt": "hello world", "reasoning": "r",
                       "emotion_label": "joyful"},
    10, 0.7, 50, 1, "cuda", cot_mode="tags+plan",
    plan_ids=G.plan_prefix_ids(reg, words))
parsed = G.parse_plan_tokens(G.plan_region(prefix[0].tolist(), reg),
                             G.plan_id_maps(reg))
assert parsed.ok and parsed.words == words, (parsed.status, parsed.words)
assert rtext == "EMO=joyful PACE=normal PITCH=normal", rtext

win = G.resolve_decode_window(parsed, 20, 400, "words", dur_lo=0.9, dur_hi=1.5,
                              use_plan=True)
assert win.source == "plan" and win.sum_frames > 0

durs = G.plan_dur_frames(parsed.words)
sched = tdc.build_decode_plan_schedule(durs, prefix.shape[1], 10,
                                       mode="stretch", target_frames=12,
                                       device="cuda")
bins = G.plan_bins_tensor(parsed.words, device="cuda")
other = G.plan_bins_tensor([{"pitch_bin": 0, "dur_bin": 12, "energy_bin": 7}] * 3,
                           device="cuda")
# A FRESH injector is zero-init on purpose (exact no-op), so give it weights or
# this test would pass on a channel that is doing nothing.
torch.manual_seed(0)
torch.nn.init.normal_(m.plan_injector.proj.weight, std=0.8)

lts = [0.0] * 10          # argmax everywhere -> deterministic streams
kw = dict(level_temps=lts, allow_stop=False, min_frames=10, speaker_id=1,
          feedback="semantic", cross_frame_decode=False, device="cuda")
with torch.no_grad():
    on = tdc.depth_generate(m, prefix, reg, max_frames=10, plan_bins=bins,
                            plan_schedule=sched, **kw)
    off = tdc.depth_generate(m, prefix, reg, max_frames=10, **kw)
    alt = tdc.depth_generate(m, prefix, reg, max_frames=10, plan_bins=other,
                             plan_schedule=sched, **kw)
assert on[0] != off[0], "injection did not reach the stream"
assert on[0] != alt[0], "the stream is not sensitive to WHICH plan"

sp = []
with torch.no_grad():
    tdc.depth_generate(m, prefix, reg, max_frames=6, stop_prob_out=sp,
                       plan_bins=bins, plan_schedule=sched,
                       **dict(kw, min_frames=6))
assert len(sp) == 6, sp                       # S154 machinery still per-frame
print("PLAN_DECODE_E2E_OK")
'''


def test_plan_decode_end_to_end_on_a_real_model():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA (mamba_ssm has no CPU kernel)")
    import subprocess
    r = subprocess.run([sys.executable, "-c", _E2E], cwd=str(ROOT),
                       capture_output=True, text=True, timeout=900)
    assert "PLAN_DECODE_E2E_OK" in r.stdout, (
        f"rc={r.returncode}\nSTDOUT:\n{r.stdout[-3000:]}\n"
        f"STDERR:\n{r.stderr[-3000:]}")


def test_plan_bins_tensor_uses_the_absent_sentinel_for_unvoiced(g):
    pytest.importorskip("mamba_ssm", reason="model.py needs mamba_ssm")
    from model import PLAN_BIN_ABSENT, PLAN_BIN_CHANNELS
    t = g.plan_bins_tensor([{"pitch_bin": None, "dur_bin": 4, "energy_bin": 3},
                            {"pitch_bin": 9, "dur_bin": 1, "energy_bin": 0}])
    assert tuple(PLAN_BIN_CHANNELS) == ("pitch", "duration", "energy")
    assert t.shape == (1, 2, 3)
    assert t[0].tolist() == [[PLAN_BIN_ABSENT, 4, 3], [9, 1, 0]]


def test_window_without_rho_truncates_a_real_utterance(g):
    parsed = g.ParsedPlan([{"pitch_bin": 2, "dur_bin": 8, "energy_bin": 1}] * 12,
                          g.PLAN_STATUS_OK, "")
    sigma_d = g.plan_sum_frames(parsed.words)
    rho = 0.6
    true_T = sigma_d / rho

    naive = g.resolve_decode_window(parsed, 20, 400, "words", dur_lo=0.9,
                                    dur_hi=1.5, use_plan=True)
    assert naive.max_fr < true_T, ("the un-corrected window caps below the true "
                                   "length — this is the defect")

    fixed = g.resolve_decode_window(parsed, 20, 400, "words", dur_lo=0.9,
                                    dur_hi=1.5, use_plan=True, speech_ratio=rho)
    assert fixed.max_fr > true_T, "the rho-corrected window must contain it"


def test_rho_one_is_exactly_the_old_behaviour(g):
    parsed = g.ParsedPlan([{"pitch_bin": 2, "dur_bin": 10, "energy_bin": 1}] * 10,
                          g.PLAN_STATUS_OK, "")
    a = g.resolve_decode_window(parsed, 20, 400, "words", dur_lo=0.9,
                                dur_hi=1.5, use_plan=True)
    b = g.resolve_decode_window(parsed, 20, 400, "words", dur_lo=0.9,
                                dur_hi=1.5, use_plan=True, speech_ratio=1.0)
    assert (a.min_fr, a.max_fr) == (b.min_fr, b.max_fr)


def test_t_hat_is_recorded_separately_from_sigma_d(g):
    parsed = g.ParsedPlan([{"pitch_bin": 2, "dur_bin": 10, "energy_bin": 1}] * 10,
                          g.PLAN_STATUS_OK, "")
    w = g.resolve_decode_window(parsed, 20, 400, "words", dur_lo=0.9,
                                dur_hi=1.5, use_plan=True, speech_ratio=0.6)
    assert w.sum_frames == pytest.approx(g.plan_sum_frames(parsed.words))
    assert w.t_hat == pytest.approx(w.sum_frames / 0.6)
    assert w.max_fr == int(1.5 * w.t_hat)


def test_plan_and_duration_model_windows_agree_on_the_same_length(g):
    parsed = g.ParsedPlan([{"pitch_bin": 2, "dur_bin": 9, "energy_bin": 1}] * 10,
                          g.PLAN_STATUS_OK, "")
    rho = 0.6
    w = g.resolve_decode_window(parsed, 20, 400, "words", dur_lo=0.9,
                                dur_hi=1.5, use_plan=True, speech_ratio=rho)
    t_hat = g.plan_sum_frames(parsed.words) / rho
    dm_min = max(20, int(0.9 * t_hat))
    dm_max = max(dm_min + 5, int(1.5 * t_hat))
    assert (w.min_fr, w.max_fr) == (dm_min, dm_max)


def test_non_positive_rho_is_rejected(g):
    parsed = g.ParsedPlan([{"pitch_bin": 2, "dur_bin": 8, "energy_bin": 1}] * 4,
                          g.PLAN_STATUS_OK, "")
    for bad in (0.0, -0.5):
        with pytest.raises(ValueError, match="speech_ratio"):
            g.resolve_decode_window(parsed, 20, 400, "words", dur_lo=0.9,
                                    dur_hi=1.5, use_plan=True, speech_ratio=bad)
