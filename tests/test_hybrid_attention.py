# test_hybrid_attention.py: The hybrid top-K attention graft.

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model import AttentionMixer


class TestAttentionMixerCPU:
    def _mixer(self, d=128):
        torch.manual_seed(0)
        return AttentionMixer(d_model=d, head_dim=64, device="cpu",
                              dtype=torch.float32)

    def test_shape(self):
        m = self._mixer()
        x = torch.randn(2, 17, 128)
        y = m(x)
        assert y.shape == (2, 17, 128)
        assert torch.isfinite(y).all()

    def test_causality(self):
        m = self._mixer()
        x = torch.randn(1, 12, 128)
        y1 = m(x)
        x2 = x.clone()
        x2[0, 8:] += 10.0
        y2 = m(x2)
        assert torch.allclose(y1[0, :8], y2[0, :8], atol=1e-5), \
            "future positions leaked into the past"
        assert not torch.allclose(y1[0, 8:], y2[0, 8:], atol=1e-3)

    def test_gradient_flow(self):
        m = self._mixer()
        x = torch.randn(1, 9, 128, requires_grad=True)
        m(x).sum().backward()
        for name, p in m.named_parameters():
            assert p.grad is not None and p.grad.abs().sum() > 0, name

    def test_rope_position_sensitivity(self):
        m = self._mixer()
        tok = torch.randn(1, 1, 128)
        a = torch.cat([tok, torch.randn(1, 3, 128), tok], dim=1)
        y = m(a)
        assert not torch.allclose(y[0, 0], y[0, 4], atol=1e-3)

    def test_uncached_forward_unchanged_and_prefill_matches(self):
        class P:
            seqlen_offset = 0
            key_value_memory_dict = {}
        m = self._mixer()
        m.layer_idx = 0
        x = torch.randn(1, 5, 128)
        y1 = m(x)
        y2 = m(x, inference_params=P())
        assert torch.allclose(y1, y2, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
class TestHybridIntegrationGPU:
    @pytest.fixture(scope="class")
    def hybrid_model(self):
        from model import MambaCoTModel
        model = MambaCoTModel(model_name="state-spaces/mamba2-130m",
                              device="cuda", dtype=torch.bfloat16,
                              mtp_num_heads=0)
        replaced = model.enable_hybrid_attention(top_k=2)
        return model, replaced

    def test_layer_swap(self, hybrid_model):
        model, replaced = hybrid_model
        layers = model.backbone.backbone.layers
        n = len(layers)
        assert replaced == [n - 2, n - 1]
        for i in replaced:
            assert isinstance(layers[i].mixer, AttentionMixer)
        assert not isinstance(layers[0].mixer, AttentionMixer)

    def test_forward_and_grads_both_layer_types(self, hybrid_model):
        model, replaced = hybrid_model
        ids = torch.randint(0, 1000, (1, 32), device="cuda")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(input_ids=ids, labels=ids)
        assert out["loss"] is not None and torch.isfinite(out["loss"])
        out["loss"].backward()
        attn_p = next(model.backbone.backbone.layers[replaced[0]]
                      .mixer.in_proj.parameters())
        mamba_p = model.backbone.backbone.layers[0].mixer.in_proj.weight
        assert attn_p.grad is not None and attn_p.grad.abs().sum() > 0
        assert mamba_p.grad is not None and mamba_p.grad.abs().sum() > 0

    def test_loader_drops_replaced_keys(self, hybrid_model, tmp_path):
        import copy
        from train import load_checkpoint
        from model import MambaCoTModel
        model, replaced = hybrid_model
        pure = MambaCoTModel(model_name="state-spaces/mamba2-130m",
                             device="cuda", dtype=torch.bfloat16,
                             mtp_num_heads=0)
        ck = tmp_path / "pure.pt"
        torch.save({"model_state_dict": pure.state_dict(), "step": 0}, ck)
        before = copy.deepcopy(
            model.backbone.backbone.layers[replaced[0]]
            .mixer.in_proj.weight.detach())
        load_checkpoint(str(ck), model, device="cuda")
        after = model.backbone.backbone.layers[replaced[0]] \
            .mixer.in_proj.weight.detach()
        assert torch.equal(before, after)
        assert torch.equal(
            model.backbone.backbone.layers[0].mixer.in_proj.weight.detach(),
            pure.backbone.backbone.layers[0].mixer.in_proj.weight.detach())
