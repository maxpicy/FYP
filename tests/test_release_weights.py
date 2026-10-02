# test_release_weights.py: Released safetensors load exactly like the training checkpoint.

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

safetensors = pytest.importorskip("safetensors")
from safetensors.torch import load_file, save_file

from train import (checkpoint_speaker_rows, checkpoint_tensor_shapes, load_checkpoint,
                   read_checkpoint)


def _sd():
    g = torch.Generator().manual_seed(0)
    return {"speaker_encoder.embedding.weight": torch.randn(7, 3, generator=g),
            "plan_injector.proj.weight": torch.randn(4, 2, generator=g),
            "lin.weight": torch.randn(3, 3, generator=g), "lin.bias": torch.randn(3, generator=g)}


def test_read_checkpoint_safetensors_matches_pt(tmp_path):
    sd = _sd()
    pt = tmp_path / "c.pt"
    torch.save({"model_state_dict": sd, "step": 40000, "optimizer_state_dict": {}}, pt)
    st = tmp_path / "model.safetensors"
    save_file(sd, str(st), metadata={"step": "40000"})
    a, b = read_checkpoint(pt), read_checkpoint(st)
    assert b["step"] == 40000 and set(b) == {"model_state_dict", "step"}
    assert set(a["model_state_dict"]) == set(b["model_state_dict"])
    for k in sd:
        assert torch.equal(a["model_state_dict"][k], b["model_state_dict"][k])
    assert checkpoint_speaker_rows(pt) == checkpoint_speaker_rows(st) == 7
    assert checkpoint_tensor_shapes(pt) == checkpoint_tensor_shapes(st)
    assert any(k.startswith("plan_injector.") for k in checkpoint_tensor_shapes(st))


def test_load_checkpoint_accepts_safetensors(tmp_path):
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(3, 3)

    src = Tiny()
    st = tmp_path / "model.safetensors"
    save_file({k: v.detach().clone() for k, v in src.state_dict().items()}, str(st), metadata={"step": "7"})
    dst = Tiny()
    assert load_checkpoint(str(st), dst, device="cpu") == 7
    for k, v in src.state_dict().items():
        assert torch.equal(dst.state_dict()[k], v)


def _fake_checkpoint(path, **args):
    t = torch.randn(5, 4)
    sd = _sd()
    sd["head.weight"] = t
    sd["tied.weight"] = t
    targs = {"hybrid_attention_top_k": 0, "backbone": "mamba", "depth_arch": "mamba2", "depth_layers": 4,
             "depth_dim": 1024, "depth_cond": "prefix", "depth_feedback": "all"}
    targs.update(args)
    torch.save({"model_state_dict": sd, "optimizer_state_dict": {"x": torch.zeros(3)}, "step": 40000, "stage": 2,
                "args": targs}, path)
    return sd


def _export(ck, out, arch):
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "export_release_weights.py"), "--checkpoint", str(ck),
                           "--arch", arch, "--name", "t", "--out", str(out)], capture_output=True, text=True)


def test_export_is_bit_exact_and_keeps_aliases(tmp_path):
    ck = tmp_path / "checkpoint_step_40000.pt"
    sd = _fake_checkpoint(ck)
    r = _export(ck, tmp_path / "out", "pure")
    assert r.returncode == 0, r.stdout + r.stderr
    back = load_file(str(tmp_path / "out" / "model.safetensors"))
    assert set(back) == set(sd)
    for k in sd:
        assert torch.equal(back[k], sd[k])
    cfg = json.loads((tmp_path / "out" / "config.json").read_text())
    assert cfg["bit_exact_vs_checkpoint"] and cfg["step"] == 40000 and cfg["speaker_table_rows"] == 7
    assert cfg["tied_aliases"] == {"tied.weight": "head.weight"}
    assert cfg["generator_flags"][-2:] == ["--num_speakers", "7"] and cfg["has_plan_injector"]
    targs = json.loads((tmp_path / "out" / "training_args.json").read_text())
    assert targs["depth_cond"] == "prefix"


@pytest.mark.parametrize("arch,args", [("hyb", {}), ("pure", {"hybrid_attention_top_k": 4}),
                                       ("tfm", {}), ("pure", {"depth_layers": 2})])


def test_export_refuses_a_mislabelled_checkpoint(tmp_path, arch, args):
    ck = tmp_path / "c.pt"
    _fake_checkpoint(ck, **args)
    r = _export(ck, tmp_path / "out", arch)
    assert r.returncode != 0 and "FATAL" in (r.stdout + r.stderr)
