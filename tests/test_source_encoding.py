# test_source_encoding.py: Every source file is valid UTF-8.

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", "node_modules", ".idea",
             ".vscode", "checkpoints", "data", "logs", "out", ".claude",
             ".nscc_helper", "fish_pilot"}
SUFFIXES = {".py", ".pbs", ".sh", ".md", ".json", ".yaml", ".yml", ".toml", ".cfg"}


def iter_source_files():
    import os
    for dirpath, dirnames, filenames in os.walk(REPO):
        dirnames[:] = [d for d in dirnames
                       if d not in SKIP_DIRS
                       and not d.startswith((".", "listen_", "out_"))]
        for fn in filenames:
            p = Path(dirpath) / fn
            if p.suffix.lower() in SUFFIXES:
                yield p


def test_repo_has_source_files_to_check():
    assert sum(1 for _ in iter_source_files()) > 50


def test_every_source_file_is_valid_utf8():
    bad = []
    for p in iter_source_files():
        raw = p.read_bytes()
        try:
            raw.decode("utf-8")
        except UnicodeDecodeError as e:
            rel = p.relative_to(REPO).as_posix()
            bad.append(f"{rel}: byte 0x{raw[e.start]:02x} at offset {e.start}")
    assert not bad, (
        "non-UTF-8 bytes in source (usually a cp1252 punctuation character "
        "written by a patch script that omitted encoding='utf-8'):\n  "
        + "\n  ".join(bad))


@pytest.mark.parametrize("suffix", [".pbs", ".sh"])
def test_shell_scripts_use_lf_endings(suffix):
    bad = [p.relative_to(REPO).as_posix() for p in iter_source_files()
           if p.suffix == suffix and b"\r\n" in p.read_bytes()]
    assert not bad, f"CRLF line endings in {suffix} files: {bad}"
