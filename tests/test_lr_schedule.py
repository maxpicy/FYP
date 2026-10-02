# test_lr_schedule.py: The learning-rate schedule anneals under gradient accumulation.

import math
import types

import torch

from train import build_scheduler

PEAK = 3e-4
LR_MIN = 1e-5


def _make_args(*, max_steps=1000, warmup_steps=100, grad_accum=4):
    return types.SimpleNamespace(
        lr=PEAK,
        lr_min=LR_MIN,
        max_steps=max_steps,
        warmup_steps=warmup_steps,
        grad_accum=grad_accum,
        stage=1,
    )


def _run_full_schedule(args):
    param = torch.nn.Parameter(torch.zeros(1))
    param.grad = torch.zeros_like(param)
    opt = torch.optim.SGD([param], lr=args.lr)
    sched = build_scheduler(opt, args)

    total_updates = max(1, args.max_steps // args.grad_accum)
    lrs = [opt.param_groups[0]["lr"]]
    for _ in range(total_updates):
        opt.step()
        sched.step()
        lrs.append(opt.param_groups[0]["lr"])
    return lrs


def test_starts_at_zero():
    lrs = _run_full_schedule(_make_args())
    assert lrs[0] == 0.0


def test_peaks_at_warmup_end():
    args = _make_args()
    warmup_updates = args.warmup_steps // args.grad_accum
    lrs = _run_full_schedule(args)
    assert math.isclose(lrs[warmup_updates], PEAK, rel_tol=1e-9)
    assert lrs[warmup_updates - 1] < PEAK


def test_anneals_to_lr_min_at_horizon():
    args = _make_args()
    lrs = _run_full_schedule(args)
    final_lr = lrs[-1]
    assert math.isclose(final_lr, LR_MIN, rel_tol=1e-6)
    assert final_lr < 0.05 * PEAK


def test_monotonic_decay_after_warmup():
    args = _make_args()
    warmup_updates = args.warmup_steps // args.grad_accum
    lrs = _run_full_schedule(args)
    decay = lrs[warmup_updates:]
    assert all(a >= b - 1e-12 for a, b in zip(decay, decay[1:]))


def test_grad_accum_one_is_consistent():
    args = _make_args(max_steps=500, warmup_steps=50, grad_accum=1)
    lrs = _run_full_schedule(args)
    assert lrs[0] == 0.0
    assert math.isclose(lrs[-1], LR_MIN, rel_tol=1e-6)


def test_production_scale_horizon_hits_lr_min():
    args = _make_args(max_steps=200_000, warmup_steps=4_000, grad_accum=4)
    param = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([param], lr=args.lr)
    sched = build_scheduler(opt, args)

    total_updates = args.max_steps // args.grad_accum
    warmup_updates = args.warmup_steps // args.grad_accum
    lr_lambda = sched.lr_lambdas[0]

    assert math.isclose(lr_lambda(0), 0.0, abs_tol=1e-12)
    assert math.isclose(lr_lambda(warmup_updates), 1.0, rel_tol=1e-9)
    assert math.isclose(args.lr * lr_lambda(total_updates), LR_MIN, rel_tol=1e-6)
