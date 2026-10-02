# test_validate_no_ddp.py: Validation never runs through the DDP wrapper.

import pathlib
import re

TRAIN = pathlib.Path(__file__).resolve().parent.parent / "train.py"


def _validate_body():
    src = TRAIN.read_text(encoding="utf-8")
    start = src.index("def validate(")
    nxt = re.search(r"\n(?:def |# =====)", src[start + 10:])
    return src[start:start + 10 + (nxt.start() if nxt else len(src))]


def test_validate_forwards_through_the_unwrapped_module():
    body = _validate_body()
    assert "outputs = unwrapped(" in body, \
        "validation must forward through the unwrapped module"
    assert "outputs = model(" not in body, (
        "validate() forwards through the DDP wrapper — under a rank-0-only "
        "call this broadcasts buffers to ranks that never arrive and deadlocks "
        "(NCCL BROADCAST watchdog after 600 s)")


def test_validate_still_unwraps_for_ddp():
    body = _validate_body()
    assert "model.module if isinstance(model, DDP)" in body, \
        "the DDP unwrap disappeared; `unwrapped` would then be the wrapper"


def test_call_site_is_still_rank_zero_only():
    src = TRAIN.read_text(encoding="utf-8")
    assert re.search(r"rank == 0 and step % args\.val_every", src), \
        "validation call-site guard changed — re-check the DDP reasoning above"
