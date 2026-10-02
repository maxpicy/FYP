# test_cached_decode.py: The cached-state decode matches the recompute path.

import os
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from model import AttentionMixer


class _Params:
    def __init__(self, seqlen_offset=0):
        self.seqlen_offset = seqlen_offset
        self.key_value_memory_dict = {}


def _mixer(d=128):
    torch.manual_seed(0)
    return AttentionMixer(d_model=d, head_dim=64, device="cpu", dtype=torch.float32,
                          layer_idx=3).eval()


def test_uncached_forward_is_unchanged_and_prefill_matches_it():
    m = _mixer()
    x = torch.randn(1, 11, 128)
    with torch.no_grad():
        ref = m(x)
        pre = m(x, inference_params=_Params(0))
    assert torch.allclose(ref, pre, atol=1e-6)


def test_stepwise_decode_equals_full_forward():
    m = _mixer()
    x = torch.randn(1, 13, 128)
    with torch.no_grad():
        ref = m(x)
        p = _Params(0)
        outs = [m(x[:, :5], inference_params=p)]
        p.seqlen_offset = 5
        for i in range(5, 13):
            outs.append(m(x[:, i:i + 1], inference_params=p))
            p.seqlen_offset = i + 1
        step = torch.cat(outs, dim=1)
    assert torch.allclose(ref, step, atol=1e-5), (ref - step).abs().max()
    assert p.key_value_memory_dict[3][0].shape[2] == 13


def test_multi_position_step_onto_cache_is_causal_and_correct():
    m = _mixer()
    x = torch.randn(1, 10, 128)
    with torch.no_grad():
        ref = m(x)
        p = _Params(0)
        a = m(x[:, :4], inference_params=p); p.seqlen_offset = 4
        b = m(x[:, 4:], inference_params=p)
    assert torch.allclose(ref, torch.cat([a, b], dim=1), atol=1e-5)


def test_layer_idx_is_required_for_the_cache():
    m = AttentionMixer(d_model=128, head_dim=64, device="cpu", dtype=torch.float32)
    with pytest.raises(RuntimeError):
        m(torch.randn(1, 3, 128), inference_params=_Params(0))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA (mamba_ssm kernels)")
class TestDepthGenerateIdentityGPU:
    @pytest.fixture(scope="class")
    def tdc(self):
        import test_depth_checkpoint as t
        return t

    def _model(self, hybrid, tdc):
        from model import MambaCoTModel
        torch.manual_seed(0)
        m = MambaCoTModel(model_name="state-spaces/mamba2-130m", device="cuda",
                          dtype=torch.float32, mtp_num_heads=0)
        if hybrid:
            m.enable_hybrid_attention(top_k=2)
        m.enable_depth_module(num_levels=tdc.NUM_LEVELS, codebook_size=64,
                              d_depth=64, depth_layers=1, depth_arch="gru")
        return m.eval()

    def _prefix(self, m):
        torch.manual_seed(1)
        text = torch.randint(5, 1000, (1, 9))
        return torch.cat([text, torch.tensor([[m.token_registry.audio_start_id]])], dim=1).cuda()

    def _run(self, m, tdc, cached, sg=None, n=24):
        torch.manual_seed(2)
        with torch.no_grad():
            return tdc.depth_generate(
                m, self._prefix(m), m.token_registry, max_frames=n, min_frames=n,
                allow_stop=False, feedback="all", cross_frame_decode=False,
                device="cuda", cached=cached, silence_guard=sg)

    @pytest.mark.parametrize("hybrid", [False, True], ids=["pure", "hybrid"])
    def test_cached_equals_recompute_greedy(self, tdc, hybrid):
        m = self._model(hybrid, tdc)
        a = self._run(m, tdc, cached=False)
        b = self._run(m, tdc, cached=True)
        assert a is not None and b is not None
        assert a == b, f"first divergence at frame {next(i for i in range(len(a[0])) if any(a[k][i] != b[k][i] for k in range(len(a))))}"

    def test_rewind_reprefill_matches_recompute(self, tdc):
        m = self._model(False, tdc)
        base = self._run(m, tdc, cached=False)
        pause = sorted(set(base[0][:6]))
        mk = lambda: tdc.SilenceGuard(codes=pause, budget=2, stop_after=99,
                                      retries=1, retry_temp=0.0)
        g0, g1 = mk(), mk()
        a = self._run(m, tdc, cached=False, sg=g0)
        b = self._run(m, tdc, cached=True, sg=g1)
        if g0.stats().get("rewinds", 0) == 0:
            pytest.skip("guard did not fire on this init; rewind path not exercised")
        assert g0.stats() == g1.stats()
        assert a == b

    def test_cached_plan_injection_matches_recompute_teacher_forced(self, tdc):
        m = self._model(False, tdc)
        m.enable_plan_injection()
        m.eval()
        torch.manual_seed(3)
        torch.nn.init.normal_(m.plan_injector.proj.weight, std=0.05)
        torch.nn.init.normal_(m.plan_injector.proj.bias, std=0.05)
        reg = m.token_registry
        prefix = self._prefix(m)
        P, n, NL = prefix.shape[1], 24, tdc.NUM_LEVELS
        sizes = m.plan_injector.bin_sizes
        g = torch.Generator().manual_seed(4)
        bins = torch.stack([torch.randint(0, s, (3,), generator=g) for s in sizes],
                           dim=-1).unsqueeze(0).cuda()
        sched = tdc.build_decode_plan_schedule([6, 9, 9], prefix_len=P, max_frames=n,
                                               mode="pack", device="cuda")
        backbone = m.backbone.backbone
        emb = backbone.embedding
        gen = torch.Generator(device="cuda").manual_seed(7)
        frames = [torch.randint(0, 64, (NL,), generator=gen, device="cuda") for _ in range(n)]

        def ids_codes(t):
            ids = torch.cat([prefix, torch.full((1, t), reg.audio_start_id,
                                                dtype=torch.long, device="cuda")], 1)
            codes = torch.full((1, P + t, NL), -1, dtype=torch.long, device="cuda")
            for i in range(t):
                codes[0, P + i] = frames[i]
            return ids, codes

        def stage(ids, codes, lo, plan):
            emb.set_codes(codes[:, lo:])
            if plan:
                emb.set_plan(m.plan_injector(bins, sched[:, :ids.shape[1]])[:, lo:])

        def run(plan):
            re, ca = [], []
            with torch.no_grad():
                for t in range(n):
                    ids, codes = ids_codes(t)
                    stage(ids, codes, 0, plan)
                    re.append(backbone(ids)[:, -1, :].float().clone())
                ip = tdc._fresh_inference_cache(P + n + 8); n_fed = 0
                for t in range(n):
                    ids, codes = ids_codes(t)
                    lo = n_fed
                    stage(ids, codes, lo, plan)
                    ip.seqlen_offset = n_fed
                    h = backbone(ids[:, lo:], inference_params=ip)
                    n_fed = ids.shape[1]
                    ca.append(h[:, -1, :].float().clone())
                lm = sum(int(m.backbone.lm_head(a).argmax()) == int(m.backbone.lm_head(b).argmax())
                         for a, b in zip(re, ca))
                dp = sum(list(m.depth_module.generate_frame(a)) == list(m.depth_module.generate_frame(b))
                         for a, b in zip(re, ca))
            rel = [float((a - b).norm() / (a.norm() + 1e-9)) for a, b in zip(re, ca)]
            return re, rel, lm, dp

        with_plan, rel, lm, dp = run(True)
        assert max(rel) < 5e-3, f"plan path: hidden rel {max(rel):.2e} (per frame {rel})"
        assert lm == n and dp == n, f"plan path: lm-head {lm}/{n}, depth {dp}/{n}"
        no_plan, rel0, _, _ = run(False)
        moved = max(float((a - b).norm() / (a.norm() + 1e-9)) for a, b in zip(with_plan, no_plan))
        assert moved > 1e-2, f"the randomised injector moved the hidden state by only {moved:.2e}"
        assert max(rel0) < 5e-3
