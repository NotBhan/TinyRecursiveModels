"""Tests for the non-invasive recurrence verification primitives."""

from __future__ import annotations

import torch

from evaluators.recurrence_verification import (
    logit_difference,
    pair_stats,
    run_independent_ensemble,
    run_recurrence,
    state_stats,
)
from models.recursive_reasoning.trm_lm_adaptive import DualStateConfig, DualStateModel


def tiny_model():
    cfg = DualStateConfig(
        vocab_size=128,
        seq_len=32,
        hidden_size=64,
        num_heads=4,
        n_low_layers=1,
        n_high_layers=1,
        n_low_cycles=1,
        n_high_cycles=1,
        min_high_cycles=1,
        max_high_cycles=4,
    )
    model = DualStateModel(cfg)
    model.eval()
    return model


def fixed_input():
    torch.manual_seed(0)
    ids = torch.randint(1, 127, (2, 12))
    mask = torch.ones_like(ids)
    return ids, mask


def test_normal_recurrence_matches_forward():
    model = tiny_model()
    ids, mask = fixed_input()
    for k in [1, 2, 4]:
        ref = model(ids, mask, mode="fixed", n_high_cycles=k).logits
        manual = run_recurrence(
            model, ids, mask, n_high_cycles=k, reset_mode="none",
            last_token_only=False,
        )["final_logits"]
        assert torch.allclose(ref, manual, atol=0.0, rtol=0.0)


def test_broken_both_equals_depth_one():
    model = tiny_model()
    ids, mask = fixed_input()
    a1 = run_recurrence(
        model, ids, mask, n_high_cycles=1, reset_mode="none", last_token_only=True
    )["final_logits"]
    b4 = run_recurrence(
        model, ids, mask, n_high_cycles=4, reset_mode="both", last_token_only=True
    )["final_logits"]
    assert torch.equal(a1, b4)


def test_single_pathway_resets_change_the_output():
    model = tiny_model()
    ids, mask = fixed_input()
    a4 = run_recurrence(
        model, ids, mask, n_high_cycles=4, reset_mode="none", last_token_only=True
    )["final_logits"]
    high = run_recurrence(
        model, ids, mask, n_high_cycles=4, reset_mode="high", last_token_only=True
    )["final_logits"]
    low = run_recurrence(
        model, ids, mask, n_high_cycles=4, reset_mode="low", last_token_only=True
    )["final_logits"]
    assert not torch.allclose(a4, high)
    assert not torch.allclose(a4, low)


def test_zero_perturbation_is_identity():
    model = tiny_model()
    ids, mask = fixed_input()
    clean = run_recurrence(
        model, ids, mask, n_high_cycles=4, reset_mode="none", last_token_only=True,
        record_states=True,
    )
    zero = run_recurrence(
        model, ids, mask, n_high_cycles=4, reset_mode="none", last_token_only=True,
        record_states=True, perturb_after_depth=2, perturb_target="both",
        perturb_scale=0.0, perturb_seed=0,
    )
    assert torch.equal(clean["final_logits"], zero["final_logits"])
    assert pair_stats(clean["z_H"], zero["z_H"])["rel_change_max"] == 0.0


def test_perturbation_is_monotonic_in_scale():
    model = tiny_model()
    ids, mask = fixed_input()
    clean = run_recurrence(
        model, ids, mask, n_high_cycles=4, reset_mode="none", last_token_only=True
    )["final_logits"]
    values = []
    for scale in [0.0, 0.02, 0.05, 0.1]:
        out = run_recurrence(
            model, ids, mask, n_high_cycles=4, reset_mode="none", last_token_only=True,
            perturb_after_depth=1, perturb_target="z_H", perturb_scale=scale,
            perturb_seed=0,
        )["final_logits"]
        values.append(logit_difference(clean, out)["max_abs"])
    assert values == sorted(values)
    assert values[0] == 0.0
    assert values[-1] > 0.0


def test_independent_ensemble_without_noise_equals_one_step():
    model = tiny_model()
    ids, mask = fixed_input()
    one = run_recurrence(
        model, ids, mask, n_high_cycles=1, reset_mode="none", last_token_only=True
    )["final_logits"]
    ensemble = run_independent_ensemble(
        model, ids, mask, n_passes=4, noise_scale=0.0, last_token_only=True
    )["final_logits"]
    assert torch.allclose(one, ensemble, atol=0.0, rtol=0.0)


def test_state_and_logit_statistics_are_finite():
    model = tiny_model()
    ids, mask = fixed_input()
    out = run_recurrence(
        model, ids, mask, n_high_cycles=2, reset_mode="none", last_token_only=True
    )
    stats = state_stats(out["z_H"])
    assert all(torch.isfinite(torch.tensor(v)) for v in stats.values())
    diff = logit_difference(out["final_logits"], out["final_logits"])
    assert diff["max_abs"] == 0.0
    assert diff["top1_change_rate"] == 0.0
