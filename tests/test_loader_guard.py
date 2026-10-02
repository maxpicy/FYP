# test_loader_guard.py: The checkpoint loader refuses silent drops (LoRA, plan tensors, speaker table).

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train import (LoRALinear, _remap_plain_keys_for_lora, checkpoint_speaker_rows,
                   load_checkpoint)


class TinyPlain(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_proj = nn.Linear(4, 8, bias=False)
        self.out_proj = nn.Linear(8, 4, bias=False)
        self.embedding = nn.Embedding(10, 4)


class TinyLora(nn.Module):
    def __init__(self, rank=2):
        super().__init__()
        self.in_proj = LoRALinear(nn.Linear(4, 8, bias=False), rank=rank,
                                  alpha=rank, dropout=0.0)
        self.out_proj = LoRALinear(nn.Linear(8, 4, bias=False), rank=rank,
                                   alpha=rank, dropout=0.0)
        self.embedding = nn.Embedding(10, 4)


def _save_ckpt(tmp_path, sd, name="ckpt.pt", step=123):
    path = tmp_path / name
    torch.save({"step": step, "model_state_dict": sd}, path)
    return str(path)


def test_remap_plain_projection_keys():
    plain_sd = TinyPlain().state_dict()
    lora_sd = TinyLora().state_dict()

    remapped, n = _remap_plain_keys_for_lora(plain_sd, lora_sd)

    assert n == 2, f"expected both projections remapped, got {n}"
    assert "in_proj.original.weight" in remapped
    assert "out_proj.original.weight" in remapped
    assert "in_proj.weight" not in remapped
    assert "embedding.weight" in remapped
    torch.testing.assert_close(remapped["in_proj.original.weight"],
                               plain_sd["in_proj.weight"])


def test_remap_leaves_matching_and_unknown_keys_alone():
    lora_model_sd = TinyLora().state_dict()
    sd = {
        "embedding.weight": torch.zeros(10, 4),
        "totally.unknown.weight": torch.zeros(2, 2),
    }
    remapped, n = _remap_plain_keys_for_lora(sd, lora_model_sd)
    assert n == 0
    assert set(remapped) == set(sd)


def test_fullft_ckpt_loads_exactly_into_lora_model(tmp_path):
    src = TinyPlain()
    with torch.no_grad():
        src.in_proj.weight += 1.0
    path = _save_ckpt(tmp_path, src.state_dict())

    dst = TinyLora()
    step = load_checkpoint(path, dst, device="cpu")

    assert step == 123
    torch.testing.assert_close(dst.in_proj.original.weight, src.in_proj.weight)
    x = torch.randn(3, 4)
    torch.testing.assert_close(dst.in_proj(x), src.in_proj(x))


def test_lora_ckpt_into_plain_model_still_errors(tmp_path):
    path = _save_ckpt(tmp_path, TinyLora().state_dict())
    with pytest.raises(RuntimeError, match="LoRA adapters"):
        load_checkpoint(path, TinyPlain(), device="cpu")


def test_dropped_projection_weight_is_a_hard_error(tmp_path):
    sd = TinyPlain().state_dict()
    sd["mixer.in_proj.weight"] = torch.zeros(8, 4)
    path = _save_ckpt(tmp_path, sd)
    with pytest.raises(RuntimeError, match="silently dropped"):
        load_checkpoint(path, TinyPlain(), device="cpu")


class TinyNeoX(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = nn.Module()
        self.attention.query_key_value = nn.Linear(4, 12, bias=False)
        self.attention.dense = nn.Linear(4, 4, bias=False)
        self.embedding = nn.Embedding(10, 4)


def test_neox_projection_unexpected_is_a_hard_error(tmp_path):
    sd = TinyNeoX().state_dict()
    sd["layers.0.attention.query_key_value.weight"] = torch.zeros(12, 4)
    path = _save_ckpt(tmp_path, sd)
    with pytest.raises(RuntimeError, match="silently dropped"):
        load_checkpoint(path, TinyNeoX(), device="cpu")


def test_missing_projection_is_a_hard_error(tmp_path):
    sd = TinyNeoX().state_dict()
    del sd["attention.query_key_value.weight"]
    path = _save_ckpt(tmp_path, sd)
    with pytest.raises(RuntimeError, match="received NO value"):
        load_checkpoint(path, TinyNeoX(), device="cpu")


def test_hybrid_graft_missing_attention_is_allowed(tmp_path):
    class TinyHybrid(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Module()
            self.backbone.backbone = nn.Module()
            self.backbone.backbone.layers = nn.ModuleDict({
                "0": nn.ModuleDict({"mixer": nn.ModuleDict(
                    {"in_proj": nn.Linear(4, 12, bias=False)})}),
            })
            self.hybrid_replaced_layers = [0]

    model = TinyHybrid()
    sd = {k: v for k, v in model.state_dict().items()
          if not k.endswith("in_proj.weight")}
    path = _save_ckpt(tmp_path, sd)
    load_checkpoint(path, model, device="cpu")


def test_architecture_mismatched_projection_shape_is_a_hard_error(tmp_path):
    sd = TinyPlain().state_dict()
    sd["in_proj.weight"] = torch.zeros(24, 4)
    path = _save_ckpt(tmp_path, sd)
    with pytest.raises(RuntimeError, match="different SHAPE"):
        load_checkpoint(path, TinyPlain(), device="cpu")


def test_shape_mismatched_speaker_table_still_only_warns(tmp_path):
    sd = TinyPlain().state_dict()
    sd["embedding.weight"] = torch.zeros(20, 4)
    path = _save_ckpt(tmp_path, sd, step=5)
    assert load_checkpoint(path, TinyPlain(), device="cpu") == 5


def test_depth_arch_swap_is_allowed(tmp_path):
    class GruDepth(nn.Module):
        def __init__(self):
            super().__init__()
            self.depth_module = nn.Module()
            self.depth_module.gru = nn.Linear(4, 4, bias=False)
            self.depth_module.h_proj = nn.Linear(4, 4, bias=False)

    class TfDepth(nn.Module):
        def __init__(self):
            super().__init__()
            self.depth_module = nn.Module()
            self.depth_module.depth_tf = nn.Module()
            self.depth_module.depth_tf.self_attn = nn.Module()
            self.depth_module.depth_tf.self_attn.out_proj = nn.Linear(4, 4, bias=False)
            self.depth_module.h_proj = nn.Linear(4, 4, bias=False)

    path = _save_ckpt(tmp_path, GruDepth().state_dict(), step=9)
    assert load_checkpoint(path, TfDepth(), device="cpu") == 9


def test_gru_to_mamba2_depth_swap_is_allowed(tmp_path):
    class GruDepth(nn.Module):
        def __init__(self):
            super().__init__()
            self.depth_module = nn.Module()
            self.depth_module.gru = nn.Linear(4, 4, bias=False)

    class SsmDepth(nn.Module):
        def __init__(self):
            super().__init__()
            self.depth_module = nn.Module()
            self.depth_module.depth_ssm = nn.Module()
            self.depth_module.depth_ssm.in_proj = nn.Linear(4, 4, bias=False)
            self.depth_module.depth_ssm.out_proj = nn.Linear(4, 4, bias=False)

    path = _save_ckpt(tmp_path, GruDepth().state_dict(), step=11)
    assert load_checkpoint(path, SsmDepth(), device="cpu") == 11


def test_same_arch_missing_depth_projection_still_raises(tmp_path):
    class TfDepth(nn.Module):
        def __init__(self):
            super().__init__()
            self.depth_module = nn.Module()
            self.depth_module.depth_tf = nn.Module()
            self.depth_module.depth_tf.self_attn = nn.Module()
            self.depth_module.depth_tf.self_attn.out_proj = nn.Linear(4, 4, bias=False)

    sd = TfDepth().state_dict()
    del sd["depth_module.depth_tf.self_attn.out_proj.weight"]
    path = _save_ckpt(tmp_path, sd)
    with pytest.raises(RuntimeError, match="received NO value"):
        load_checkpoint(path, TfDepth(), device="cpu")


def test_matched_load_roundtrip_unaffected(tmp_path):
    src = TinyLora()
    path = _save_ckpt(tmp_path, src.state_dict(), step=7)
    dst = TinyLora()
    step = load_checkpoint(path, dst, device="cpu")
    assert step == 7
    torch.testing.assert_close(dst.in_proj.lora_A, src.in_proj.lora_A)


class TinyDelay(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_proj = nn.Linear(4, 8, bias=False)
        self.delay_heads = nn.ModuleList([nn.Linear(8, 16, bias=False)])


class TinyDepth(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_proj = nn.Linear(4, 8, bias=False)
        self.depth_gru = nn.GRU(8, 8, batch_first=True)


def test_delay_ckpt_into_depth_model_warm_start(tmp_path):
    src = TinyDelay()
    path = _save_ckpt(tmp_path, src.state_dict(), step=100000)
    dst = TinyDepth()
    before = {k: v.clone() for k, v in dst.state_dict().items()
              if k.startswith("depth_gru")}
    load_checkpoint(path, dst, device="cpu")
    torch.testing.assert_close(dst.in_proj.weight, src.in_proj.weight)
    for k, v in before.items():
        torch.testing.assert_close(dst.state_dict()[k], v)


class TinySpeaker(nn.Module):
    def __init__(self, num_speakers):
        super().__init__()
        self.speaker_encoder = nn.Module()
        self.speaker_encoder.embedding = nn.Embedding(num_speakers, 4)
        self.in_proj = nn.Linear(4, 8, bias=False)


def test_speaker_table_mismatch_is_a_hard_error(tmp_path):
    path = _save_ckpt(tmp_path, TinySpeaker(16).state_dict())
    with pytest.raises(RuntimeError, match=r"Speaker table.*--num_speakers 16"):
        load_checkpoint(path, TinySpeaker(8), device="cpu")


def test_speaker_table_reinit_only_when_asked(tmp_path):
    src = TinySpeaker(16)
    with torch.no_grad():
        src.in_proj.weight += 1.0
    path = _save_ckpt(tmp_path, src.state_dict())
    dst = TinySpeaker(8)
    load_checkpoint(path, dst, device="cpu", allow_speaker_table_reinit=True)
    torch.testing.assert_close(dst.in_proj.weight, src.in_proj.weight)
    assert dst.speaker_encoder.embedding.weight.shape[0] == 8


def test_matching_speaker_table_loads_exactly(tmp_path):
    src = TinySpeaker(16)
    path = _save_ckpt(tmp_path, src.state_dict())
    dst = TinySpeaker(16)
    load_checkpoint(path, dst, device="cpu")
    torch.testing.assert_close(dst.speaker_encoder.embedding.weight,
                               src.speaker_encoder.embedding.weight)


def test_checkpoint_speaker_rows_reads_the_table_size(tmp_path):
    assert checkpoint_speaker_rows(
        _save_ckpt(tmp_path, TinySpeaker(16).state_dict())) == 16
    assert checkpoint_speaker_rows(
        _save_ckpt(tmp_path, TinyPlain().state_dict(), name="plain.pt")) is None
    assert checkpoint_speaker_rows(str(tmp_path / "absent.pt")) is None
