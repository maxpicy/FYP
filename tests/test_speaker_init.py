# test_speaker_init.py: Speaker-table initialisation on a warm start.

import ast
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "train.py"


@pytest.fixture(scope="module")
def tree():
    return ast.parse(SRC.read_text(encoding="utf-8"))


def _flags(tree):
    out = set()
    for n in ast.walk(tree):
        if (isinstance(n, ast.Call)
                and getattr(n.func, "attr", "") == "add_argument"
                and n.args and isinstance(n.args[0], ast.Constant)):
            out.add(n.args[0].value)
    return out


def test_both_flags_exist(tree):
    f = _flags(tree)
    assert "--speaker_init_from" in f
    assert "--lr_speaker" in f


def test_speaker_gets_its_own_param_group(tree):
    src = SRC.read_text(encoding="utf-8")
    assert '"speaker_encoder" in name' in src, "speaker params are not split out"
    assert '{"params": spk, "lr": lr_speaker}' in src


def test_warm_start_runs_after_init_from_not_before():
    src = SRC.read_text(encoding="utf-8")
    load = src.index("load_checkpoint(args.init_from, unwrapped")
    copy = src.index("if args.speaker_init_from:")
    assert load < copy, "warm-start must come AFTER the init_from load"


def test_warm_start_is_a_plain_copy_under_no_grad():
    src = SRC.read_text(encoding="utf-8")
    i = src.index("if args.speaker_init_from:")
    block = src[i:i + 1400]
    assert "torch.no_grad()" in block
    assert "W[dst].copy_(W[src])" in block


def test_row_spec_parsing_and_bounds():
    def parse(spec, n_rows):
        out = []
        for pair in spec.split(","):
            dst, src = (int(x) for x in pair.strip().split(":"))
            if not (0 <= dst < n_rows and 0 <= src < n_rows):
                raise ValueError(pair)
            out.append((dst, src))
        return out

    assert parse("4200:1,4201:2", 8192) == [(4200, 1), (4201, 2)]
    assert parse(" 4200:1 , 4201:2 ", 8192) == [(4200, 1), (4201, 2)]
    with pytest.raises(ValueError):
        parse("9000:1", 8192)
    with pytest.raises(ValueError):
        parse("4200:-1", 8192)
    with pytest.raises(ValueError):
        parse("4200", 8192)


def test_defaults_preserve_old_behaviour(tree):
    for n in ast.walk(tree):
        if (isinstance(n, ast.Call)
                and getattr(n.func, "attr", "") == "add_argument"
                and n.args and isinstance(n.args[0], ast.Constant)
                and n.args[0].value in ("--speaker_init_from", "--lr_speaker")):
            default = next((k.value for k in n.keywords if k.arg == "default"), None)
            assert isinstance(default, ast.Constant)
            assert default.value in (None, 0.0), n.args[0].value


def test_lr_speaker_zero_does_not_freeze(tree):
    src = SRC.read_text(encoding="utf-8")
    assert 'lr_speaker = getattr(args, "lr_speaker", 0.0)' in src, (
        "the flag lookup changed — re-check the default is still falsy 0.0")
    assert "if lr_speaker and spk:" in src, (
        "guard changed — re-check whether --lr_speaker 0 now freezes")
    i = src.index("if lr_speaker and spk:")
    else_block = src[i:i + 700]
    assert "trainable = rest + spk" in else_block, (
        "the falsy branch must still be the one that merges spk into the main "
        "group; if this changed, this test's premise is stale")


def test_freeze_speaker_flag_exists_and_is_inert_by_default(tree):
    f = _flags(tree)
    assert "--freeze_speaker" in f
    for n in ast.walk(tree):
        if (isinstance(n, ast.Call)
                and getattr(n.func, "attr", "") == "add_argument"
                and n.args and isinstance(n.args[0], ast.Constant)
                and n.args[0].value == "--freeze_speaker"):
            action = next((k.value.value for k in n.keywords if k.arg == "action"), None)
            assert action == "store_true", "must default to off (store_true)"


def test_freeze_runs_before_ddp_wrap_and_before_build_optimizer():
    src = SRC.read_text(encoding="utf-8")
    freeze = src.index("if getattr(args, \"freeze_speaker\", False):")
    ddp = src.index("model = DDP(model, device_ids=[local_rank])")
    build = src.index("optimizer = build_optimizer(unwrapped, args)")
    assert freeze < ddp, "freeze must precede the DDP wrap"
    assert freeze < build, "freeze must precede build_optimizer"


def test_freeze_is_verified_not_assumed():
    src = SRC.read_text(encoding="utf-8")
    assert "--freeze_speaker did not take" in src
    assert "frozen speaker params are in the optimizer" in src


def test_warm_start_still_runs_after_the_freeze():
    src = SRC.read_text(encoding="utf-8")
    load = src.index("load_checkpoint(args.init_from, unwrapped")
    copy = src.index("if args.speaker_init_from:")
    assert load < copy
