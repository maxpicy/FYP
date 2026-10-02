# test_plan_vocab.py: The plan and style vocabulary.

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LEGACY_NON_SPEECH_TOKENS = [
    "<BOS>", "<EOS>", "<PAD>",
    "[USER_PROMPT]",
    "<THINK>", "</THINK>",
    "<AUDIO>", "</AUDIO>",
    "[SPEAKER]",
    "[EMO:neutral]", "[EMO:sarcastic]", "[EMO:angry]", "[EMO:sad]",
    "[EMO:joyful]", "[EMO:fearful]", "[EMO:ironic]", "[EMO:deadpan]",
    "[PACE:slow]", "[PACE:normal]", "[PACE:fast]",
    "[PITCH:low]", "[PITCH:mid]", "[PITCH:high]",
    "[BREAK:short]", "[BREAK:long]",
]
assert len(LEGACY_NON_SPEECH_TOKENS) == 25

PLAN_MARKERS = ("<PLAN>", "</PLAN>", "[PW]", "[P:", "[D:", "[E:", "[STY:")


class FakeTokenizer:
    UNK_ID = 0
    UNK_TOKEN = "<|endoftext|>"

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


def _err(fn, *args):
    try:
        fn(*args)
    except Exception as exc:
        return type(exc).__name__
    return None


def probe() -> dict:
    import torch

    import config
    import tokenizer as tok

    all_new = tok.get_all_new_tokens()
    registry = tok.TokenRegistry(FakeTokenizer(all_new))

    out = {
        "enable_plan": config.ENABLE_PLAN_TOKENS,
        "num_speech": config.NUM_SPEECH_TOKENS,
        "num_style": config.NUM_STYLE_TOKENS,
        "pitch_bins": config.PITCH_BINS,
        "duration_bins": config.DURATION_BINS,
        "energy_bins": config.ENERGY_BINS,
        "n_new": len(all_new),
        "head25": all_new[:25],
        "last_token": all_new[-1],
        "n_plan_marked": sum(1 for t in all_new if t.startswith(PLAN_MARKERS)),
        "n_control": len(config.get_control_tokens()),
        "n_all_special": len(config.get_all_special_tokens()),
        "plan_block": config.get_plan_tokens(),
        "plan_enabled": registry.plan_enabled,
        "plan_start_id": registry.plan_start_id,
        "plan_end_id": registry.plan_end_id,
        "plan_word_sep_id": registry.plan_word_sep_id,
        "pitch_ids": list(registry.pitch_ids),
        "duration_ids": list(registry.duration_ids),
        "energy_ids": list(registry.energy_ids),
        "style_ids": list(registry.style_ids),
        "plan_min": registry.plan_token_id_min,
        "plan_max": registry.plan_token_id_max,
        "speech_min": registry.speech_token_id_min,
        "speech_max": registry.speech_token_id_max,
        "is_plan_of_speech_min": registry.is_plan_token(registry.speech_token_id_min),
        "is_plan_of_think": registry.is_plan_token(registry.think_start_id),
        "mask_sum_speech_only": int(
            registry.build_plan_mask(
                torch.tensor([registry.speech_token_id_min, registry.eos_id])
            ).sum()
        ),
        "err_pitch_bin_to_id": _err(registry.pitch_bin_to_id, 0),
        "err_duration_bin_to_id": _err(registry.duration_bin_to_id, 0),
        "err_energy_bin_to_id": _err(registry.energy_bin_to_id, 0),
        "err_style_index_to_id": _err(registry.style_index_to_id, 0),
        "err_id_to_pitch_bin": _err(registry.id_to_pitch_bin, registry.eos_id),
    }

    if registry.plan_enabled:
        out.update({
            "pitch_roundtrip": [
                registry.id_to_pitch_bin(registry.pitch_bin_to_id(b))
                for b in range(config.PITCH_BINS)
            ],
            "duration_roundtrip": [
                registry.id_to_duration_bin(registry.duration_bin_to_id(b))
                for b in range(config.DURATION_BINS)
            ],
            "energy_roundtrip": [
                registry.id_to_energy_bin(registry.energy_bin_to_id(b))
                for b in range(config.ENERGY_BINS)
            ],
            "style_roundtrip": [
                registry.id_to_style_index(registry.style_index_to_id(b))
                for b in range(config.NUM_STYLE_TOKENS)
            ],
            "is_plan_of_plan_start": registry.is_plan_token(registry.plan_start_id),
            "mask_sum_mixed": int(
                registry.build_plan_mask(
                    torch.tensor([
                        registry.plan_start_id,
                        registry.pitch_bin_to_id(3),
                        registry.speech_token_id_min,
                        registry.eos_id,
                    ])
                ).sum()
            ),
            "err_cross_family": _err(
                registry.id_to_pitch_bin, registry.duration_bin_to_id(0)
            ),
            "err_bin_too_high": _err(registry.pitch_bin_to_id, config.PITCH_BINS),
            "err_bin_negative": _err(registry.pitch_bin_to_id, -1),
            "err_bin_float": _err(registry.pitch_bin_to_id, 1.5),
            "err_bin_str": _err(registry.pitch_bin_to_id, "3"),
            "err_registry_without_plan_tokens": _err(
                tok.TokenRegistry,
                FakeTokenizer([t for t in all_new if not t.startswith(PLAN_MARKERS)]),
            ),
        })

    return out

_PROBE_CACHE: dict = {}


def _run_probe(**env_extra) -> dict:
    key = tuple(sorted(env_extra.items()))
    if key in _PROBE_CACHE:
        return _PROBE_CACHE[key]
    env = dict(os.environ)
    env.update(env_extra)
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve())],
        capture_output=True, text=True, cwd=str(REPO_ROOT), env=env,
    )
    assert proc.returncode == 0, (
        f"probe subprocess failed (env={env_extra}):\n{proc.stdout[-2000:]}\n{proc.stderr[-4000:]}"
    )
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    _PROBE_CACHE[key] = result
    return result


def _import_config(**env_extra) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.update(env_extra)
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-c", "import config; print(config.ENABLE_PLAN_TOKENS)"],
        capture_output=True, text=True, cwd=str(REPO_ROOT), env=env,
    )


@pytest.fixture(scope="module")
def off() -> dict:
    import config
    if config.ENABLE_PLAN_TOKENS:
        pytest.skip("suite is running with MVC_ENABLE_PLAN_TOKENS=1; gate-off cases N/A")
    return probe()


@pytest.fixture(scope="module")
def on_nostyle() -> dict:
    return _run_probe(MVC_ENABLE_PLAN_TOKENS="1", MVC_NUM_STYLE_TOKENS="0",
                      MVC_NUM_SPEECH_TOKENS="64")


@pytest.fixture(scope="module")
def on_style4() -> dict:
    return _run_probe(MVC_ENABLE_PLAN_TOKENS="1", MVC_NUM_STYLE_TOKENS="4",
                      MVC_NUM_SPEECH_TOKENS="64")


class TestGateOff:
    @pytest.mark.skipif(
        os.environ.get("MVC_ENABLE_PLAN_TOKENS") is not None,
        reason="asserts the AMBIENT default; meaningless in a process that "
               "deliberately exported the gate (a plan arm's own CI)")
    def test_default_is_off(self):
        import config
        assert config.ENABLE_PLAN_TOKENS is False
        assert config.NUM_STYLE_TOKENS == 0

    def test_new_token_list_matches_legacy_snapshot(self, off):
        assert off["head25"] == LEGACY_NON_SPEECH_TOKENS
        assert off["n_new"] == 25 + off["num_speech"]
        assert off["last_token"] == f"[SPEECH_{off['num_speech'] - 1}]"

    def test_no_plan_token_leaks_into_the_vocabulary(self, off):
        assert off["n_plan_marked"] == 0

    def test_config_token_lists_unchanged(self, off):
        assert off["n_control"] == 22
        assert off["n_all_special"] == 22 + off["num_speech"]

    def test_plan_block_is_still_enumerable(self, off):
        assert len(off["plan_block"]) == 43
        assert off["plan_block"][:3] == ["<PLAN>", "</PLAN>", "[PW]"]

    def test_registry_attributes_exist_but_are_empty(self, off):
        assert off["plan_enabled"] is False
        assert off["plan_start_id"] is None
        assert off["plan_end_id"] is None
        assert off["plan_word_sep_id"] is None
        assert off["plan_min"] is None and off["plan_max"] is None
        assert off["pitch_ids"] == [] and off["duration_ids"] == []
        assert off["energy_ids"] == [] and off["style_ids"] == []

    def test_predicates_are_false_not_raising(self, off):
        assert off["is_plan_of_speech_min"] is False
        assert off["is_plan_of_think"] is False
        assert off["mask_sum_speech_only"] == 0

    def test_helpers_fail_loudly_rather_than_return_a_wrong_id(self, off):
        assert off["err_pitch_bin_to_id"] == "RuntimeError"
        assert off["err_duration_bin_to_id"] == "RuntimeError"
        assert off["err_energy_bin_to_id"] == "RuntimeError"
        assert off["err_style_index_to_id"] == "RuntimeError"
        assert off["err_id_to_pitch_bin"] == "RuntimeError"

    def test_registry_rejects_a_tokenizer_that_carries_plan_tokens(self, off):
        import config
        import tokenizer as tok
        tokens = tok.get_all_new_tokens() + config.get_plan_tokens()
        with pytest.raises(RuntimeError, match="MVC_ENABLE_PLAN_TOKENS"):
            tok.TokenRegistry(FakeTokenizer(tokens))

    def test_style_tokens_without_the_gate_is_a_hard_error(self):
        proc = _import_config(MVC_NUM_STYLE_TOKENS="4", MVC_ENABLE_PLAN_TOKENS="0")
        assert proc.returncode != 0
        assert "MVC_ENABLE_PLAN_TOKENS" in proc.stderr

    def test_unparseable_gate_value_is_a_hard_error(self):
        proc = _import_config(MVC_ENABLE_PLAN_TOKENS="2")
        assert proc.returncode != 0
        assert "MVC_ENABLE_PLAN_TOKENS" in proc.stderr

    def test_gate_accepts_documented_literals(self):
        for value, expected in (("1", "True"), ("true", "True"), ("on", "True"),
                                ("0", "False"), ("no", "False"), ("OFF", "False")):
            proc = _import_config(MVC_ENABLE_PLAN_TOKENS=value)
            assert proc.returncode == 0, proc.stderr
            assert proc.stdout.strip() == expected, f"{value!r} -> {proc.stdout!r}"


class TestGateOn:
    def test_exact_token_count_added(self, on_nostyle):
        assert on_nostyle["enable_plan"] is True
        assert len(on_nostyle["plan_block"]) == 43
        assert on_nostyle["n_plan_marked"] == 43
        assert on_nostyle["n_new"] == 25 + 43 + on_nostyle["num_speech"]
        assert on_nostyle["n_all_special"] == 22 + 43 + on_nostyle["num_speech"]

    def test_style_tier_adds_exactly_num_style_tokens(self, on_style4):
        assert on_style4["num_style"] == 4
        assert len(on_style4["plan_block"]) == 47
        assert on_style4["n_new"] == 25 + 47 + on_style4["num_speech"]
        assert len(on_style4["style_ids"]) == 4

    def test_block_layout_is_the_documented_order(self, on_style4):
        block = on_style4["plan_block"]
        assert block[:3] == ["<PLAN>", "</PLAN>", "[PW]"]
        assert block[3:19] == [f"[P:{i}]" for i in range(16)]
        assert block[19:35] == [f"[D:{i}]" for i in range(16)]
        assert block[35:43] == [f"[E:{i}]" for i in range(8)]
        assert block[43:] == [f"[STY:{i}]" for i in range(4)]

    def test_word_separator_is_not_a_bare_pipe(self, on_nostyle):
        assert "[PW]" in on_nostyle["plan_block"]
        assert "|" not in on_nostyle["plan_block"]

    def test_registry_resolves_the_whole_block(self, on_style4):
        r = on_style4
        assert r["plan_enabled"] is True
        assert r["plan_start_id"] is not None and r["plan_end_id"] is not None
        assert r["plan_word_sep_id"] is not None
        assert len(r["pitch_ids"]) == 16
        assert len(r["duration_ids"]) == 16
        assert len(r["energy_ids"]) == 8
        ids = (r["pitch_ids"] + r["duration_ids"] + r["energy_ids"] + r["style_ids"]
               + [r["plan_start_id"], r["plan_end_id"], r["plan_word_sep_id"]])
        assert len(set(ids)) == len(ids), "plan token ids are not unique"

    def test_bin_ids_are_ordered_and_consecutive(self, on_nostyle):
        for family in ("pitch_ids", "duration_ids", "energy_ids"):
            ids = on_nostyle[family]
            assert ids == list(range(ids[0], ids[0] + len(ids))), family

    def test_bin_id_roundtrip_every_bin_every_family(self, on_style4):
        assert on_style4["pitch_roundtrip"] == list(range(16))
        assert on_style4["duration_roundtrip"] == list(range(16))
        assert on_style4["energy_roundtrip"] == list(range(8))
        assert on_style4["style_roundtrip"] == list(range(4))

    def test_speech_ids_stay_contiguous(self, on_style4):
        assert on_style4["speech_max"] - on_style4["speech_min"] + 1 == on_style4["num_speech"]

    def test_plan_block_sits_below_the_speech_floor(self, on_style4):
        assert on_style4["plan_max"] < on_style4["speech_min"]
        assert on_style4["plan_max"] - on_style4["plan_min"] + 1 == 47

    def test_predicates_and_mask(self, on_style4):
        assert on_style4["is_plan_of_plan_start"] is True
        assert on_style4["is_plan_of_speech_min"] is False
        assert on_style4["is_plan_of_think"] is False
        assert on_style4["mask_sum_speech_only"] == 0
        assert on_style4["mask_sum_mixed"] == 2

    def test_families_do_not_cross_decode(self, on_nostyle):
        assert on_nostyle["err_cross_family"] == "ValueError"

    def test_out_of_range_and_non_integral_bins_raise(self, on_nostyle):
        assert on_nostyle["err_bin_too_high"] == "ValueError"
        assert on_nostyle["err_bin_negative"] == "ValueError"
        assert on_nostyle["err_bin_float"] == "ValueError"
        assert on_nostyle["err_bin_str"] == "ValueError"

    def test_style_helpers_raise_when_style_tier_absent(self, on_nostyle):
        assert on_nostyle["style_ids"] == []
        assert on_nostyle["err_style_index_to_id"] == "ValueError"

    def test_registry_rejects_a_tokenizer_missing_the_block(self, on_nostyle):
        assert on_nostyle["err_registry_without_plan_tokens"] == "RuntimeError"

if __name__ == "__main__":
    print(json.dumps(probe()))
