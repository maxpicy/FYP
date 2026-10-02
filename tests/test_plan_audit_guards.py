# test_plan_audit_guards.py: The plan tier's silent-failure guards.

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

PLAN_LO = 6
N_PLAN = 3
V_OLD = 10
V_NEW = V_OLD + N_PLAN
PAD = 4


def _pad_to(v, mult=PAD):
    return v if v % mult == 0 else v + (mult - v % mult)


class _FakeTok:
    def __init__(self, n):
        self._n = n

    def __len__(self):
        return self._n


class _FakeReg:
    def __init__(self, n_vocab, plan_lo, n_plan):
        self.tokenizer = _FakeTok(n_vocab)
        self.plan_token_id_min = plan_lo
        self.plan_token_id_max = plan_lo + n_plan - 1
        self.plan_token_ids = set(range(plan_lo, plan_lo + n_plan))


class _TinyModel(torch.nn.Module):
    def __init__(self, n_vocab, n_speakers=5, d=4, pad=PAD):
        super().__init__()
        rows = _pad_to(n_vocab, pad)
        self.emb = torch.nn.Embedding(rows, d)
        self.lm_head = torch.nn.Linear(d, rows, bias=False)
        self.speaker_table = torch.nn.Embedding(n_speakers, d)
        self.token_registry = _FakeReg(n_vocab, PLAN_LO, N_PLAN)


def _write_ckpt(tmp_path, model, name="ck.pt"):
    p = tmp_path / name
    torch.save({"model_state_dict": model.state_dict(), "step": 7}, p)
    return str(p)


class TestVocabSplice:
    def test_splice_is_the_hand_computed_answer_with_padding(self):
        import train

        v_old_pad, v_new_pad = _pad_to(V_OLD), _pad_to(V_NEW)
        assert (v_old_pad, v_new_pad) == (12, 16)
        assert v_new_pad - v_old_pad != N_PLAN

        old = torch.arange(v_old_pad * 2, dtype=torch.float32).reshape(v_old_pad, 2)
        fresh = torch.full((v_new_pad, 2), -99.0)
        out = train._splice_plan_vocab_rows(old, fresh, PLAN_LO, N_PLAN, V_OLD)

        assert tuple(out.shape) == (v_new_pad, 2)
        assert torch.equal(out[:PLAN_LO], old[:PLAN_LO])
        assert torch.equal(out[PLAN_LO:PLAN_LO + N_PLAN],
                           torch.full((N_PLAN, 2), -99.0))
        assert out[9].tolist() == old[6].tolist()
        assert out[12].tolist() == old[9].tolist()
        assert torch.equal(out[V_NEW:], torch.full((v_new_pad - V_NEW, 2), -99.0))

    def test_splice_refuses_a_checkpoint_smaller_than_the_old_vocab(self):
        import train

        with pytest.raises(RuntimeError, match="not produced by the vocabulary"):
            train._splice_plan_vocab_rows(torch.zeros(V_OLD - 2, 2),
                                          torch.zeros(_pad_to(V_NEW), 2),
                                          PLAN_LO, N_PLAN, V_OLD)

    def test_splice_refuses_a_model_too_small_to_hold_the_result(self):
        import train

        with pytest.raises(RuntimeError, match="does not fit"):
            train._splice_plan_vocab_rows(torch.zeros(V_OLD, 2),
                                          torch.zeros(V_OLD, 2),
                                          PLAN_LO, N_PLAN, V_OLD)

    def test_vocab_row_keys_finds_padded_embedding_and_head_but_not_speaker(self):
        import train

        m = _TinyModel(V_NEW)
        assert m.emb.weight.shape[0] == _pad_to(V_NEW) > V_NEW
        keys = train._vocab_row_keys(m, m.state_dict())
        assert keys == {"emb.weight", "lm_head.weight"}

    def test_mismatched_vocab_is_a_hard_error_not_a_warning(self, tmp_path):
        import train

        old_model = _TinyModel(V_OLD)
        path = _write_ckpt(tmp_path, old_model)
        new_model = _TinyModel(V_NEW)
        before = new_model.emb.weight.detach().clone()

        with pytest.raises(RuntimeError, match="VOCABULARY-sized"):
            train.load_checkpoint(path, new_model, device="cpu")
        assert torch.equal(new_model.emb.weight.detach(), before)

    def test_splice_flag_loads_the_old_rows_exactly(self, tmp_path):
        import train

        old_model = _TinyModel(V_OLD)
        torch.nn.init.normal_(old_model.emb.weight, std=1.0)
        torch.nn.init.normal_(old_model.lm_head.weight, std=1.0)
        old_emb = old_model.emb.weight.detach().clone()
        old_head = old_model.lm_head.weight.detach().clone()
        path = _write_ckpt(tmp_path, old_model)

        new_model = _TinyModel(V_NEW)
        fresh_plan_rows = new_model.emb.weight.detach()[PLAN_LO:PLAN_LO + N_PLAN].clone()
        step = train.load_checkpoint(path, new_model, device="cpu",
                                     splice_plan_vocab=True)
        assert step == 7

        got = new_model.emb.weight.detach()
        assert torch.equal(got[:PLAN_LO], old_emb[:PLAN_LO])
        assert torch.equal(got[PLAN_LO:PLAN_LO + N_PLAN], fresh_plan_rows)
        assert torch.equal(got[PLAN_LO + N_PLAN:V_NEW], old_emb[PLAN_LO:V_OLD])
        head = new_model.lm_head.weight.detach()
        assert torch.equal(head[:PLAN_LO], old_head[:PLAN_LO])
        assert torch.equal(head[PLAN_LO + N_PLAN:V_NEW], old_head[PLAN_LO:V_OLD])

    def test_matching_vocab_still_loads_with_no_flag(self, tmp_path):
        import train

        src = _TinyModel(V_NEW)
        torch.nn.init.normal_(src.emb.weight, std=1.0)
        path = _write_ckpt(tmp_path, src)
        dst = _TinyModel(V_NEW)
        train.load_checkpoint(path, dst, device="cpu")
        assert torch.equal(dst.emb.weight.detach(), src.emb.weight.detach())

    def test_speaker_table_resize_is_still_a_legitimate_silent_drop(self, tmp_path):
        import train

        src = _TinyModel(V_NEW, n_speakers=5)
        torch.nn.init.normal_(src.emb.weight, std=1.0)
        path = _write_ckpt(tmp_path, src)
        dst = _TinyModel(V_NEW, n_speakers=9)
        train.load_checkpoint(path, dst, device="cpu")
        assert torch.equal(dst.emb.weight.detach(), src.emb.weight.detach())
        assert dst.speaker_table.weight.shape[0] == 9


class TestQuantiserNaN:
    def test_bin_index_known_answers(self):
        import plan_extract as pe

        edges = np.array([1.0, 2.0, 3.0])
        assert pe._bin_index(0.5, edges) == 0
        assert pe._bin_index(1.0, edges) == 1
        assert pe._bin_index(2.5, edges) == 2
        assert pe._bin_index(99.0, edges) == 3

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_raises_instead_of_saturating(self, bad):
        import plan_extract as pe

        with pytest.raises(pe.PlanQuantiseError, match="non-finite"):
            pe._bin_index(bad, np.array([1.0, 2.0, 3.0]))

    def test_nan_energy_no_longer_becomes_the_loudest_bin(self):
        import plan_extract as pe

        dur_s = 0.1
        w = [{"word": "a", "dur_s": dur_s,
              "dur_frames": max(1, int(round(dur_s * pe.FRAME_RATE_HZ))),
              "f0_mean_hz": 200.0, "energy_rms": float("nan")}]
        with pytest.raises(pe.PlanQuantiseError):
            pe.quantise_plan(w)

    def test_a_finite_row_still_quantises(self):
        import plan_extract as pe

        dur_s = 0.4
        w = [{"word": "a", "dur_s": dur_s,
              "dur_frames": max(1, int(round(dur_s * pe.FRAME_RATE_HZ))),
              "f0_mean_hz": 200.0, "energy_rms": 0.05}]
        q = pe.quantise_plan(w)
        assert 0 <= q[0]["energy_bin"] < pe.N_ENERGY_BINS
        assert 0 <= q[0]["dur_bin"] < pe.N_DUR_BINS
        assert 0 <= q[0]["pitch_bin"] < pe.N_PITCH_BINS
        w2 = [dict(w[0], energy_rms=0.0)]
        assert pe.quantise_plan(w2)[0]["energy_bin"] == 0


class TestInjectorBinRange:
    def _inj(self):
        from model import PlanInjector
        return PlanInjector(d_model=8, n_pitch_bins=16, n_duration_bins=16,
                            n_energy_bins=8, d_plan=4, scaffold_dropout=0.0)

    def test_pitch_sentinel_is_accepted_as_absent(self):
        inj = self._inj()
        sentinel = inj.word_vectors(torch.tensor([[[16, 4, 2]]]))
        absent = inj.word_vectors(torch.tensor([[[-1, 4, 2]]]))
        assert torch.equal(sentinel, absent)

    @pytest.mark.parametrize("bins,axis", [([3, 16, 2], "duration"),
                                           ([3, 4, 8], "energy")])
    def test_out_of_range_on_the_other_channels_raises(self, bins, axis):
        inj = self._inj()
        with pytest.raises(ValueError, match=axis):
            inj.word_vectors(torch.tensor([[bins]]))

    def test_valid_top_bins_still_pass(self):
        inj = self._inj()
        v = inj.word_vectors(torch.tensor([[[15, 15, 7]]]))
        assert v.shape == (1, 1, 4)
        assert bool(v.abs().sum() > 0)


class TestCellWavOrdering:
    def _make(self, tmp_path, n):
        d = tmp_path / "cellA"
        d.mkdir()
        for i in range(n):
            (d / f"sample_{i:02d}.wav").write_bytes(b"")
        cells = {"cellA": {100 + i: {} for i in range(n)}}
        return cells, tmp_path

    def test_order_is_numeric_beyond_99(self, tmp_path):
        pfp = pytest.importorskip("plan_flip_probe")
        cells, wavdir = self._make(tmp_path, 105)
        got = pfp.cell_wavs(cells, "cellA", wavdir)
        assert got[100].name == "sample_00.wav"
        assert got[111].name == "sample_11.wav"
        assert got[200].name == "sample_100.wav"
        assert got[204].name == "sample_104.wav"

    def test_a_gap_in_the_decode_run_raises(self, tmp_path):
        pfp = pytest.importorskip("plan_flip_probe")
        cells, wavdir = self._make(tmp_path, 5)
        (wavdir / "cellA" / "sample_02.wav").unlink()
        cells["cellA"].pop(102)
        with pytest.raises(SystemExit, match="complete 0"):
            pfp.cell_wavs(cells, "cellA", wavdir)


class TestIntegrityReadsRealFields:
    def _rec(self, n_frames=40, **extra):
        r = {"speech_tokens": list(range(n_frames)), "dur_window": [20, 80]}
        r.update(extra)
        return r

    def test_status_name_matches_the_generator(self):
        pfp = pytest.importorskip("plan_flip_probe")
        gen = pytest.importorskip("gen_fish_cot_samples")
        assert pfp.PLAN_STATUS_OK_NAME == gen.PLAN_STATUS_OK

    def test_malformed_rate_is_counted_from_plan_status(self):
        pfp = pytest.importorskip("plan_flip_probe")
        cells = {"self": {
            0: self._rec(plan_status="ok", plan_ok=True),
            1: self._rec(plan_status="unterminated", plan_ok=False,
                         plan_window_fallback=True),
            2: self._rec(plan_status="malformed", plan_ok=False,
                         plan_window_fallback=True),
        }}
        rep = pfp._integrity(cells, None, "plan_used")["self"]
        assert rep["n_plans_parsed"] == 3
        assert rep["n_plans_malformed"] == 2
        assert rep["n_plan_window_fallback"] == 2

    def test_a_plan_free_cell_reports_no_plans_and_does_not_raise(self):
        pfp = pytest.importorskip("plan_flip_probe")
        cells = {"base": {0: self._rec(), 1: self._rec()}}
        rep = pfp._integrity(cells, None, "plan_used")["base"]
        assert rep["n_plans_parsed"] == 0
        assert rep["n_plans_malformed"] == 0
        assert rep["n_plan_window_fallback"] == 0
