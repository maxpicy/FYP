# test_lazy_fork_safety.py: The lazy corpus reader is safe under forked workers.

import json
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

dd = pytest.importorskip("delay_dataset")

pytestmark = pytest.mark.skipif(
    not hasattr(os, "fork"), reason="fork-specific failure mode (POSIX only)")


class _FakeLazy:
    def __init__(self, path, pid_guard):
        self.path = str(path)
        self.pid_guard = pid_guard
        self._fh = None
        self._fh_pid = None
        self.offsets = []
        with open(self.path, "rb") as f:
            pos = f.tell()
            for line in f:
                if line.strip():
                    self.offsets.append(pos)
                pos = f.tell()

    def row(self, idx):
        if self._fh is None or (self.pid_guard and self._fh_pid != os.getpid()):
            self._fh = open(self.path, "rb")
            self._fh_pid = os.getpid()
        self._fh.seek(self.offsets[idx])
        return json.loads(self._fh.readline())


def _corpus(tmp_path, n=400):
    p = tmp_path / "corpus.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for i in range(n):
            f.write(json.dumps({"row": i, "pad": "x" * (7 + (i * 13) % 400)}) + "\n")
    return p


def _read_in_forks(ds, n_children, idxs):
    ds.row(0)
    pipes = [os.pipe() for _ in range(n_children)]
    pids = []
    for c, (r, w) in enumerate(pipes):
        pid = os.fork()
        if pid == 0:
            code = 0
            try:
                os.close(r)
                for k, idx in enumerate(idxs):
                    if k % n_children != c:
                        continue
                    got = ds.row(idx)
                    if got["row"] != idx:
                        code = 2
                        break
            except Exception:
                code = 1
            finally:
                try:
                    os.write(w, bytes([code]))
                    os.close(w)
                except Exception:
                    pass
                os._exit(0)
        os.close(w)
        pids.append(pid)
    codes = []
    for r, _ in pipes:
        b = os.read(r, 1)
        os.close(r)
        codes.append(b[0] if b else 3)
    for pid in pids:
        os.waitpid(pid, 0)
    return codes


def test_offset_index_is_correct(tmp_path):
    p = _corpus(tmp_path, n=50)
    ds = _FakeLazy(p, pid_guard=True)
    assert len(ds.offsets) == 50
    for i in (0, 1, 17, 49):
        assert ds.row(i)["row"] == i


def test_shared_handle_across_forks_corrupts_reads(tmp_path):
    p = _corpus(tmp_path, n=400)
    ds = _FakeLazy(p, pid_guard=False)
    codes = _read_in_forks(ds, n_children=4, idxs=list(range(400)))
    assert any(c != 0 for c in codes), (
        "expected corruption from a shared file descriptor; if this passes, "
        "the reproduction no longer exercises the mechanism and the guard "
        "below is no longer proven to be load-bearing")


def test_pid_guard_makes_forked_reads_correct(tmp_path):
    p = _corpus(tmp_path, n=400)
    ds = _FakeLazy(p, pid_guard=True)
    codes = _read_in_forks(ds, n_children=4, idxs=list(range(400)))
    assert all(c == 0 for c in codes), f"child exit codes {codes}"


def test_dataset_declares_fh_pid():
    src = (REPO / "delay_dataset.py").read_text(encoding="utf-8")
    assert "_fh_pid" in src
    assert "self._fh_pid != os.getpid()" in src, \
        "the reopen condition must compare the OWNING pid, not just None"


def test_getstate_drops_both_handle_and_pid():
    src = (REPO / "delay_dataset.py").read_text(encoding="utf-8")
    i = src.index("def __getstate__")
    body = src[i:i + 400]
    assert 'state["_fh"] = None' in body
    assert 'state["_fh_pid"] = None' in body
