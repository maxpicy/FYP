# test_p2cot_seams.py: The plan tier's cross-file contracts.

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PROBE = r'''
import json, sys, tempfile
from pathlib import Path
REPO = Path(sys.argv[1])
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))

out = {}
try:
    import torch
    from tokenizer import load_base_tokenizer, expand_tokenizer, TokenRegistry
    import plan_extract as PX
    import plan_adherence as PA
    import delay_dataset as DD
    import model as M
    import train as TR

    tok = load_base_tokenizer()
    expand_tokenizer(tok)
    reg = TokenRegistry(tok)
except Exception as e:                       # offline / missing cache
    print(json.dumps({"unavailable": f"{type(e).__name__}: {e}"}))
    raise SystemExit(0)

# A plan with a deliberately UNVOICED middle word and both extreme bins.
WORDS = [
    {"word": "alpha", "pitch_bin": 3,  "dur_bin": 4, "energy_bin": 2, "dur_frames": 7},
    {"word": "sss",   "pitch_bin": PX.PITCH_UNVOICED_BIN,
                      "dur_bin": 1, "energy_bin": 0, "dur_frames": 3},
    {"word": "omega", "pitch_bin": 15, "dur_bin": 9, "energy_bin": 7, "dur_frames": 11},
]
T = sum(w["dur_frames"] for w in WORDS)

# ---- S1: extractor string -> tokenizer ids, one id per symbol --------------
plan_str = PX.plan_to_tokens(WORDS, wrap=True)
ids = tok.encode(plan_str, add_special_tokens=False)
n_symbols = 2 + sum(1 + (0 if w["pitch_bin"] == PX.PITCH_UNVOICED_BIN else 1) + 2
                    for w in WORDS)
out["extractor_string"] = plan_str
out["n_ids"] = len(ids)
out["n_symbols"] = n_symbols
out["ids_are_plan_block"] = all(reg.is_plan_token(int(i)) for i in ids)

# ---- S1/S2: dataset-emitted ids -> extractor parser ------------------------
row = {"prompt": "alpha sss omega",
       "speech_tokens": [5] * T,
       "residual_codes": [[7] * T for _ in range(DD.NUM_LEVELS - 1)],
       "speaker_id": 1,
       "plan": {"words": WORDS, "ok": True, "coverage": 1.0}}
p = Path(tempfile.mkdtemp()) / "rows.jsonl"
p.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")

ds = DD.DelayMimiDataset(str(p), tok, reg, max_seq_len=512, stage=2,
                         aligned=True, cot_mode="plan", audio_continue_weight=1.0)
item = ds[0]
seq = item["input_ids"].tolist()
plan_pos = [i for i, t in enumerate(seq) if reg.is_plan_token(t)]
dataset_str = "".join(tok.convert_ids_to_tokens([seq[i] for i in plan_pos]))
out["dataset_string"] = dataset_str
out["dataset_string_matches_extractor"] = (dataset_str == plan_str)
out["parsed_back"] = PX.tokens_to_plan(dataset_str)

# ---- S3: collator slots -> train adapter -> injector bins ------------------
batch = DD.DelayCollator(pad_token_id=0)([ds[0], ds[1]])
out["collator_plan_keys"] = sorted(k for k in batch if k.startswith("plan_"))
bins = TR.plan_bins_from_slots(batch["input_ids"], batch["plan_slots"], reg)
out["bins"] = bins.tolist()
inj = M.PlanInjector(d_model=16, n_pitch_bins=len(reg.pitch_ids),
                     n_duration_bins=len(reg.duration_ids),
                     n_energy_bins=len(reg.energy_ids), d_plan=8,
                     scaffold_dropout=0.0)
out["injector_accepts_batch"] = tuple(
    inj(bins, batch["plan_schedule"]).shape) == (2, len(seq), 16)

# ---- S4: the two schedule builders ----------------------------------------
durs = [w["dur_frames"] for w in WORDS]
a0 = int(item["audio_start"])
ds_sched = item["plan_schedule"].tolist()
m_sched = M.build_plan_schedule([durs], a0, T, seq_len=len(seq))[0].tolist()
# plan_lookahead=1: the dataset places frame t one position EARLIER than
# model.build_plan_schedule does, because the input at p predicts the code at
# p+1. Shift by exactly that one position and the two must be identical.
shifted = [-1] * len(seq)
for t in range(T):
    shifted[a0 + t - 1] = m_sched[a0 + t]
out["schedules_agree_modulo_lookahead"] = (shifted == ds_sched)
out["dataset_schedule_head"] = ds_sched[a0 - 2:a0 + 4]
out["model_schedule_head"] = m_sched[a0 - 2:a0 + 4]

# ---- S2/S5: PAS against the extractor's own output -------------------------
try:
    out["pas_on_quantised_output"] = PA.pas(WORDS, WORDS)["pas"]
except Exception as e:
    out["pas_on_quantised_output"] = f"RAISED {type(e).__name__}: {e}"

out["plan_extract_has_contract"] = {
    n: hasattr(PX, n) for n in ("parse_plan_string", "extract_plan_words")}
try:
    sg = PA.score_generation("g.wav", plan_str, "alpha sss omega",
                             extract_fn=lambda w, t, **k: PX.parse_plan_string(plan_str))
    out["score_generation_self_pas"] = sg["pas"]["pas"]
except Exception as e:
    out["score_generation_self_pas"] = f"RAISED {type(e).__name__}: {e}"

print(json.dumps(out))
'''


@pytest.fixture(scope="module")
def seams():
    env = dict(os.environ)
    env["MVC_ENABLE_PLAN_TOKENS"] = "1"
    env["MVC_NUM_SPEECH_TOKENS"] = "1024"
    env["MVC_NUM_CODEC_LEVELS"] = "10"
    env["HF_HUB_OFFLINE"] = env.get("HF_HUB_OFFLINE", "1")
    proc = subprocess.run([sys.executable, "-c", PROBE, str(REPO_ROOT)],
                          capture_output=True, text=True, env=env, timeout=900)
    if proc.returncode != 0:
        pytest.fail(f"seam probe crashed:\n{proc.stdout}\n{proc.stderr}")
    line = proc.stdout.strip().splitlines()[-1]
    data = json.loads(line)
    if "unavailable" in data:
        pytest.skip(f"real tokenizer unavailable: {data['unavailable']}")
    return data


def test_plan_symbols_cost_exactly_one_token_id_each(seams):
    assert seams["n_ids"] == seams["n_symbols"], seams["extractor_string"]
    assert seams["ids_are_plan_block"], "a plan symbol tokenized outside the plan block"


def test_dataset_emits_exactly_the_string_the_extractor_writes(seams):
    assert seams["dataset_string_matches_extractor"], (
        f"dataset: {seams['dataset_string']!r}\n"
        f"extractor: {seams['extractor_string']!r}")


def test_extractor_parser_recovers_the_dataset_program(seams):
    expect = [{"pitch_bin": 3, "dur_bin": 4, "energy_bin": 2},
              {"pitch_bin": 16, "dur_bin": 1, "energy_bin": 0},
              {"pitch_bin": 15, "dur_bin": 9, "energy_bin": 7}]
    assert seams["parsed_back"] == expect


def test_collator_emits_the_channels_the_train_loop_reads(seams):
    assert seams["collator_plan_keys"] == [
        "plan_durations", "plan_lengths", "plan_schedule", "plan_slots"]


def test_slots_decode_back_to_the_sidecar_bins(seams):
    expect_row = [[3, 4, 2], [-1, 1, 0], [15, 9, 7]]
    assert seams["bins"] == [expect_row, expect_row]


def test_injector_consumes_the_collated_batch_unchanged(seams):
    assert seams["injector_accepts_batch"]


def test_pas_accepts_the_extractors_unvoiced_sentinel(seams):
    assert seams["pas_on_quantised_output"] == 1.0, seams["pas_on_quantised_output"]


def test_training_and_decode_schedules_agree_modulo_the_lookahead(seams):
    assert seams["schedules_agree_modulo_lookahead"], (
        f"dataset {seams['dataset_schedule_head']} vs "
        f"model {seams['model_schedule_head']}")


def test_plan_extract_exposes_the_entry_points_pas_resolves(seams):
    assert seams["plan_extract_has_contract"] == {
        "parse_plan_string": True, "extract_plan_words": True}


def test_score_generation_runs_through_the_real_parser(seams):
    assert seams["score_generation_self_pas"] == 1.0, seams["score_generation_self_pas"]


def test_trailing_zero_duration_word_is_never_scheduled():
    torch = pytest.importorskip("torch")
    from model import build_plan_schedule

    got = build_plan_schedule([[4, 0]], 0, 6)[0].tolist()
    assert got == [0, 0, 0, 0, 0, 0], got

    got = build_plan_schedule([[3, 2, 0, 0]], 0, 8)[0].tolist()
    assert got == [0, 0, 0, 1, 1, 1, 1, 1], got

    got = build_plan_schedule([[0, 0]], 0, 3)[0].tolist()
    assert got == [-1, -1, -1], got


def test_documented_known_answers_still_hold():
    pytest.importorskip("torch")
    from model import build_plan_schedule

    assert build_plan_schedule([[2, 3]], 0, 5)[0].tolist() == [0, 0, 1, 1, 1]
    assert build_plan_schedule([[2, 2]], 0, 6)[0].tolist() == [0, 0, 1, 1, 1, 1]
    assert build_plan_schedule([[3, 0, 2]], 0, 7)[0].tolist() == [0, 0, 0, 2, 2, 2, 2]
    assert build_plan_schedule([[4]], 2, 3, 7)[0].tolist() == [-1, -1, 0, 0, 0, -1, -1]


@pytest.mark.skipif(
    os.environ.get("MVC_ENABLE_PLAN_TOKENS") is not None,
    reason="reads the AMBIENT gate; a process that exported it (a plan arm's "
           "own CI) is not the default environment this asserts about")


def test_gate_off_leaves_no_plan_vocabulary():
    import config
    import importlib
    importlib.reload(config)
    assert config.ENABLE_PLAN_TOKENS is False
    assert config.get_control_tokens() == [
        config.USER_PROMPT_TOKEN, config.THINK_START_TOKEN, config.THINK_END_TOKEN,
        config.AUDIO_START_TOKEN, config.AUDIO_END_TOKEN, config.SPEAKER_TOKEN,
    ] + config.EMOTION_TAGS + config.PROSODY_TAGS


def test_plan_extract_token_surface_matches_config_vocabulary():
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import plan_extract as PX
    assert PX.verify_config_vocab() == "ok"
