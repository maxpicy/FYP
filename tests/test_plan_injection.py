# test_plan_injection.py: Duration-scheduled plan injection.

import pytest
import torch
import torch.nn as nn

from model import (
    PLAN_BIN_ABSENT,
    PLAN_BIN_CHANNELS,
    DelayEmbedding,
    PlanInjector,
    build_plan_schedule,
)

CUDA = torch.cuda.is_available()

D_MODEL = 8
VOCAB = 32


def _delay_embedding(d_model=D_MODEL, vocab=VOCAB, num_levels=2, codebook=4):
    torch.manual_seed(0)
    base = nn.Embedding(vocab, d_model)
    nn.init.normal_(base.weight, std=1.0)
    return DelayEmbedding(base, num_levels=num_levels, codebook_size=codebook)


def _injector(scaffold_dropout=0.0, d_model=D_MODEL, seed=0):
    torch.manual_seed(seed)
    return PlanInjector(d_model=d_model, n_pitch_bins=16, n_duration_bins=16,
                        n_energy_bins=8, d_plan=6,
                        scaffold_dropout=scaffold_dropout)


def _train_injector(inj):
    with torch.no_grad():
        torch.manual_seed(1)
        nn.init.normal_(inj.proj.weight, std=0.5)
        nn.init.normal_(inj.proj.bias, std=0.5)
    return inj


def test_set_plan_adds_per_position_and_speaker_still_broadcasts():
    emb = _delay_embedding()
    ids = torch.arange(5).unsqueeze(0)
    base = emb(ids)

    plan = torch.zeros(1, 5, D_MODEL)
    plan[0, 2] = 1.0
    emb.set_plan(plan)
    with_plan = emb(ids)
    assert torch.equal(with_plan[0, 2], base[0, 2] + 1.0)
    for p in (0, 1, 3, 4):
        assert torch.equal(with_plan[0, p], base[0, p]), \
            "set_plan must be per-position, not broadcast"

    spk = torch.full((1, D_MODEL), 2.0)
    emb.set_speaker(spk)
    with_spk = emb(ids)
    for p in range(5):
        assert torch.equal(with_spk[0, p], base[0, p] + 2.0), \
            "set_speaker must still broadcast to every position"


def test_plan_and_speaker_compose():
    emb = _delay_embedding()
    ids = torch.arange(4).unsqueeze(0)
    base = emb(ids)

    plan = torch.zeros(1, 4, D_MODEL)
    plan[0, 1] = 1.0
    plan[0, 3] = 4.0
    spk = torch.full((1, D_MODEL), 2.0)
    emb.set_plan(plan)
    emb.set_speaker(spk)
    both = emb(ids)

    expected = base + plan + 2.0
    assert torch.equal(both, expected), \
        "staging a speaker AND a plan must add both channels"


def test_staged_plan_consumed_exactly_once():
    emb = _delay_embedding()
    ids = torch.arange(3).unsqueeze(0)
    base = emb(ids)

    plan = torch.full((1, 3, D_MODEL), 7.0)
    emb.set_plan(plan)
    first = emb(ids)
    assert torch.equal(first, base + 7.0)

    second = emb(ids)
    assert torch.equal(second, base), \
        "a stale plan must NOT be re-applied on the next forward"
    assert emb._plan is None


def test_codes_and_plan_are_independent_slots():
    emb = _delay_embedding()
    ids = torch.arange(3).unsqueeze(0)
    base = emb(ids)
    with torch.no_grad():
        emb.level_embeds[0].weight.fill_(0.5)

    codes = torch.full((1, 3, 2), -1, dtype=torch.long)
    codes[0, 1] = torch.tensor([0, 0])
    plan = torch.zeros(1, 3, D_MODEL)
    plan[0, 2] = 3.0
    emb.set_codes(codes)
    emb.set_plan(plan)
    out = emb(ids)

    assert torch.equal(out[0, 0], base[0, 0])
    assert torch.equal(out[0, 1], base[0, 1] + 0.5)
    assert torch.equal(out[0, 2], base[0, 2] + 3.0)


def _sched(durs, audio_start, n_frames, seq_len=None):
    return build_plan_schedule(durs, audio_start, n_frames,
                               seq_len=seq_len).tolist()


def test_schedule_basic_two_words():
    assert _sched([[2, 3]], 0, 5) == [[0, 0, 1, 1, 1]]


def test_schedule_audio_start_offset():
    got = _sched([[2, 3]], 3, 5, seq_len=10)
    assert got == [[-1, -1, -1, 0, 0, 1, 1, 1, -1, -1]]


def test_schedule_single_word_plan():
    assert _sched([[4]], 0, 4) == [[0, 0, 0, 0]]


def test_schedule_last_word_absorbs_the_remainder():
    assert _sched([[2, 2]], 0, 6) == [[0, 0, 1, 1, 1, 1]]


def test_schedule_plan_longer_than_audio_truncates_cleanly():
    assert _sched([[5, 3, 2]], 0, 3) == [[0, 0, 0]]


def test_schedule_zero_duration_word_is_skipped():
    assert _sched([[3, 0, 2]], 0, 7) == [[0, 0, 0, 2, 2, 2, 2]]


def test_schedule_float_durations_round_half_up():
    assert _sched([[1.4, 2.6]], 0, 4) == [[0, 1, 1, 1]]


def test_schedule_batch_with_padding_and_ragged_rows():
    durs = [[2, 1, PLAN_BIN_ABSENT],
            [1, 1, 1]]
    got = _sched(durs, [1, 0], [3, 3], seq_len=5)
    assert got == [[-1, 0, 0, 1, -1],
                   [0, 1, 2, -1, -1]]


def test_schedule_accepts_tensor_input():
    durs = torch.tensor([[2, 3], [4, -1]])
    got = build_plan_schedule(durs, 0, 5, seq_len=5).tolist()
    assert got == [[0, 0, 1, 1, 1],
                   [0, 0, 0, 0, 0]]


def test_schedule_accepts_numpy_and_scalar_types():
    np = pytest.importorskip("numpy")
    durs = np.array([[2, 3]], dtype=np.int64)
    got = build_plan_schedule(durs, np.int64(1), np.int64(5), seq_len=6).tolist()
    assert got == [[-1, 0, 0, 1, 1, 1]]
    got_f = build_plan_schedule(np.array([[1.4, 2.6]]), 0, 4, seq_len=4).tolist()
    assert got_f == [[0, 1, 1, 1]]


def test_schedule_empty_plan_row_is_all_minus_one():
    got = _sched([[], [2]], 0, 2, seq_len=2)
    assert got == [[-1, -1], [0, 0]]


def test_schedule_no_frames_is_all_minus_one():
    assert _sched([[3]], 2, 0, seq_len=4) == [[-1, -1, -1, -1]]


def test_schedule_default_seq_len_is_the_audio_end():
    assert _sched([[2]], 2, 2) == [[-1, -1, 0, 0]]


def test_schedule_rejects_audio_block_past_seq_len():
    with pytest.raises(ValueError, match="past seq_len"):
        build_plan_schedule([[4]], 3, 4, seq_len=5)


def test_schedule_rejects_interior_padding():
    with pytest.raises(ValueError, match="interior position"):
        build_plan_schedule([[2, -1, 3]], 0, 5)


def test_schedule_rejects_stray_negative_duration():
    with pytest.raises(ValueError, match="pad sentinel"):
        build_plan_schedule([[2, -3]], 0, 5)


def test_schedule_rejects_negative_audio_start():
    with pytest.raises(ValueError, match="audio_start"):
        build_plan_schedule([[2]], -1, 2, seq_len=4)


def test_schedule_rejects_flat_duration_sequence():
    with pytest.raises(ValueError, match="PER-ROW"):
        build_plan_schedule([2, 3], 0, 5)


def test_schedule_batch_length_mismatch_is_loud():
    with pytest.raises(ValueError, match="audio_start"):
        build_plan_schedule([[2], [2]], [0, 0, 0], 2)


def test_schedule_total_frames_match_durations():
    durs = [3, 1, 4, 2, 5]
    sched = build_plan_schedule([durs], 7, sum(durs), seq_len=7 + sum(durs))[0]
    for w, d in enumerate(durs):
        assert int((sched == w).sum()) == d, f"word {w} got the wrong span"
    assert int((sched == -1).sum()) == 7


def _bins(rows):
    return torch.tensor(rows, dtype=torch.long)


def test_zero_init_injection_is_an_exact_noop():
    inj = _injector()
    plan = _bins([[[7, 4, 3], [9, 2, 4]]])
    sched = build_plan_schedule([[2, 2]], 1, 4, seq_len=6)
    out = inj(plan, sched)
    assert torch.equal(out, torch.zeros_like(out)), \
        "fresh PlanInjector must emit exactly zero (zero-init proj)"

    emb = _delay_embedding()
    ids = torch.arange(6).unsqueeze(0)
    base = emb(ids)
    emb.set_plan(out)
    assert torch.equal(emb(ids), base), \
        "enabling injection must not perturb the forward before training"


def test_non_audio_positions_stay_zero_after_training():
    inj = _train_injector(_injector())
    plan = _bins([[[7, 4, 3], [9, 2, 4]]])
    sched = build_plan_schedule([[2, 2]], 2, 4, seq_len=8)
    out = inj(plan, sched)
    assert torch.equal(out[0, :2], torch.zeros(2, D_MODEL))
    assert torch.equal(out[0, 6:], torch.zeros(2, D_MODEL))
    assert out[0, 2:6].abs().sum() > 0


def test_absent_and_unvoiced_pitch_share_the_absent_row():
    inj = _train_injector(_injector())
    sched = build_plan_schedule([[2]], 0, 2, seq_len=2)
    a = inj(_bins([[[PLAN_BIN_ABSENT, 4, 3]]]), sched)
    b = inj(_bins([[[16, 4, 3]]]), sched)
    assert torch.equal(a, b)
    c = inj(_bins([[[0, 4, 3]]]), sched)
    assert not torch.equal(a, c), "bin 0 must differ from 'no pitch'"


def test_out_of_range_bin_is_loud():
    inj = _injector()
    sched = build_plan_schedule([[1]], 0, 1, seq_len=1)
    with pytest.raises(ValueError, match="pitch bin 17"):
        inj(_bins([[[17, 0, 0]]]), sched)
    with pytest.raises(ValueError, match="energy bin 9"):
        inj(_bins([[[0, 0, 9]]]), sched)
    with pytest.raises(ValueError, match="duration bin -2"):
        inj(_bins([[[0, -2, 0]]]), sched)


def test_schedule_referencing_a_missing_word_is_loud():
    inj = _injector()
    plan = _bins([[[1, 1, 1]]])
    sched = torch.tensor([[0, 1]])
    with pytest.raises(ValueError, match="references word 1"):
        inj(plan, sched)


def test_empty_plan_tensor_is_loud():
    inj = _injector()
    with pytest.raises(ValueError, match="zero words"):
        inj(torch.zeros(1, 0, 3, dtype=torch.long), torch.full((1, 2), -1))


def test_bin_channel_order_is_pitch_duration_energy():
    assert PLAN_BIN_CHANNELS == ("pitch", "duration", "energy")
    inj = _train_injector(_injector())
    sched = build_plan_schedule([[1]], 0, 1, seq_len=1)
    straight = inj(_bins([[[1, 2, 3]]]), sched)
    swapped = inj(_bins([[[2, 1, 3]]]), sched)
    assert not torch.equal(straight, swapped)


def test_gradient_reaches_the_injector():
    inj = _injector()
    plan = _bins([[[7, 4, 3], [9, 2, 4]]])
    sched = build_plan_schedule([[2, 2]], 0, 4, seq_len=4)

    inj(plan, sched).sum().backward()
    assert inj.proj.weight.grad is not None
    assert inj.proj.weight.grad.abs().sum() > 0
    assert inj.proj.bias.grad.abs().sum() > 0

    opt = torch.optim.SGD(inj.parameters(), lr=0.1)
    opt.step()
    opt.zero_grad()

    inj(plan, sched).sum().backward()
    grads = [e.weight.grad.abs().sum().item() for e in inj.embeds]
    assert all(g > 0 for g in grads), f"dead bin embedding(s): {grads}"
    for e in inj.embeds:
        assert torch.equal(e.weight.grad[0], torch.zeros(inj.d_plan))


def test_scaffold_dropout_drops_whole_utterances_in_train_mode():
    inj = _train_injector(_injector(scaffold_dropout=1.0))
    plan = _bins([[[7, 4, 3], [9, 2, 4]]])
    sched = build_plan_schedule([[2, 2]], 0, 4, seq_len=4)

    inj.train()
    out = inj(plan, sched)
    assert torch.equal(out, torch.zeros_like(out)), \
        "p=1.0 in train mode must drop the injection entirely"

    inj.eval()
    out_eval = inj(plan, sched)
    assert out_eval.abs().sum() > 0, \
        "scaffold dropout must be a no-op in eval mode (it is a training " \
        "scaffold, not an inference condition)"


def test_scaffold_dropout_is_per_utterance_not_per_position():
    torch.manual_seed(3)
    inj = _train_injector(_injector(scaffold_dropout=0.5))
    inj.train()
    plan = _bins([[[7, 4, 3]]] * 16)
    sched = build_plan_schedule([[4]] * 16, 0, 4, seq_len=4)
    out = inj(plan, sched)
    per_row_alive = (out.abs().sum(dim=(1, 2)) > 0)
    for b in range(16):
        row = out[b]
        alive = (row.abs().sum(dim=-1) > 0)
        assert bool(alive.all()) == bool(per_row_alive[b]), \
            "dropout must apply to the whole utterance, not per position"
    assert 0 < int(per_row_alive.sum()) < 16, \
        "p=0.5 over 16 rows should drop some and keep some (seeded)"


def test_scaffold_dropout_keeps_params_in_the_graph():
    inj = _train_injector(_injector(scaffold_dropout=1.0))
    inj.train()
    plan = _bins([[[7, 4, 3]]])
    sched = build_plan_schedule([[2]], 0, 2, seq_len=2)
    out = inj(plan, sched)
    assert out.requires_grad
    out.sum().backward()
    assert inj.proj.weight.grad is not None


def test_scaffold_dropout_rejects_non_probabilities():
    with pytest.raises(ValueError, match="scaffold_dropout"):
        _injector(scaffold_dropout=1.5)
    with pytest.raises(ValueError, match="scaffold_dropout"):
        PlanInjector(d_model=4, n_pitch_bins=2, n_duration_bins=2,
                     n_energy_bins=2, scaffold_dropout=-0.1)


def test_different_plan_changes_the_injected_stream():
    inj = _train_injector(_injector())
    sched = build_plan_schedule([[2, 2]], 0, 4, seq_len=4)
    a = inj(_bins([[[7, 4, 3], [9, 2, 4]]]), sched)
    b = inj(_bins([[[11, 4, 3], [9, 2, 4]]]), sched)
    assert not torch.equal(a, b), \
        "if flipping a plan bin leaves the input stream unchanged, the plan " \
        "is decorative by construction"
    assert not torch.equal(a[0, :2], b[0, :2])
    assert torch.equal(a[0, 2:], b[0, 2:])


def test_different_durations_reschedule_the_stream():
    inj = _train_injector(_injector())
    plan = _bins([[[7, 4, 3], [9, 2, 4]]])
    a = inj(plan, build_plan_schedule([[2, 2]], 0, 4, seq_len=4))
    b = inj(plan, build_plan_schedule([[3, 1]], 0, 4, seq_len=4))
    assert torch.equal(a[0, :2], b[0, :2])
    assert not torch.equal(a[0, 2], b[0, 2])
    assert torch.equal(a[0, 3], b[0, 3])


def test_injected_stream_is_constant_within_a_word():
    inj = _train_injector(_injector())
    plan = _bins([[[7, 4, 3], [9, 2, 4]]])
    sched = build_plan_schedule([[3, 2]], 0, 5, seq_len=5)
    out = inj(plan, sched)[0]
    vec = inj.word_vectors(plan)
    gathered = vec[0][sched[0].clamp(min=0)]
    assert torch.equal(gathered[0], gathered[1])
    assert torch.equal(gathered[3], gathered[4])
    assert not torch.equal(gathered[2], gathered[3])
    for a, b in ((0, 1), (1, 2), (3, 4)):
        assert torch.allclose(out[a], out[b], atol=1e-6)
    assert (out[2] - out[3]).abs().max() > 1e-3


@pytest.fixture(scope="module")
def depth_model():
    from model import MambaCoTModel
    if not CUDA:
        pytest.skip("needs CUDA")
    m = MambaCoTModel(model_name="state-spaces/mamba2-130m", device="cuda",
                      dtype=torch.bfloat16, mtp_num_heads=0)
    m.enable_depth_module(num_levels=8, codebook_size=2048)
    return m


def _depth_batch(model, seq_len=24, audio_start=8, n_frames=8):
    V = len(model.tokenizer)
    ids = torch.randint(0, min(V, 1000), (1, seq_len), device="cuda")
    labels = ids.clone()
    codes = torch.full((1, seq_len, 8), -1, dtype=torch.long, device="cuda")
    codes[0, audio_start:audio_start + n_frames] = torch.randint(
        0, 2048, (n_frames, 8), device="cuda")
    plan = torch.tensor([[[7, 4, 3], [9, 2, 4]]], device="cuda")
    sched = build_plan_schedule([[4, 4]], audio_start, n_frames,
                                seq_len=seq_len, device="cuda")
    return ids, labels, codes, plan, sched


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_model_zero_init_plan_injection_is_exact(depth_model):
    m = depth_model
    ids, labels, codes, plan, sched = _depth_batch(m)
    with torch.no_grad():
        base = m.forward_depth(ids, codes, labels=labels)
    m.enable_plan_injection(scaffold_dropout=0.0)
    with torch.no_grad():
        inj = m.forward_depth(ids, codes, labels=labels,
                              plan_bins=plan, plan_schedule=sched)
    assert torch.equal(base["logits"], inj["logits"]), \
        "fresh plan injection must be an exact no-op (zero-init projection)"
    assert torch.equal(base["loss"], inj["loss"])


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_model_plan_state_dict_only_grows_when_enabled(depth_model):
    from model import MambaCoTModel
    keys = [k for k in depth_model.state_dict() if k.startswith("plan_injector")]
    assert keys, "enabled model must expose plan_injector.* parameters"
    assert not any(k.endswith(("in_proj.weight", "out_proj.weight")) for k in keys)

    fresh = MambaCoTModel(model_name="state-spaces/mamba2-130m", device="cuda",
                          dtype=torch.bfloat16, mtp_num_heads=0)
    fresh.enable_depth_module(num_levels=8, codebook_size=2048)
    assert not [k for k in fresh.state_dict() if k.startswith("plan_injector")], \
        "a model without enable_plan_injection() must add no parameters"


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_model_grad_flows_to_the_injector(depth_model):
    m = depth_model
    if not m.use_plan_injection:
        m.enable_plan_injection(scaffold_dropout=0.0)
    ids, labels, codes, plan, sched = _depth_batch(m)
    m.zero_grad(set_to_none=True)
    out = m.forward_depth(ids, codes, labels=labels, plan_bins=plan,
                          plan_schedule=sched)
    out["loss"].backward()
    assert m.plan_injector.proj.weight.grad is not None
    assert m.plan_injector.proj.weight.grad.abs().sum() > 0
    m.zero_grad(set_to_none=True)


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_model_half_a_plan_is_loud(depth_model):
    m = depth_model
    if not m.use_plan_injection:
        m.enable_plan_injection(scaffold_dropout=0.0)
    ids, labels, codes, plan, sched = _depth_batch(m)
    with pytest.raises(ValueError, match="must be supplied together"):
        m.forward_depth(ids, codes, labels=labels, plan_bins=plan)
    with pytest.raises(ValueError, match="must be supplied together"):
        m.forward_depth(ids, codes, labels=labels, plan_schedule=sched)


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_plan_injection_requires_the_delay_embedding_channel():
    from model import MambaCoTModel
    m = MambaCoTModel(model_name="state-spaces/mamba2-130m", device="cuda",
                      dtype=torch.bfloat16, mtp_num_heads=0)
    with pytest.raises(RuntimeError, match="DelayEmbedding"):
        m.enable_plan_injection()
