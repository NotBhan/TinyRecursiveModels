"""Tests for the optional gated/residual stability update.

The default ``update_mode='overwrite'`` must remain identical to the historical
model. The stability paths are only active from ``stable_update_from_depth``.
"""

from __future__ import annotations

import torch

from models.recursive_reasoning.trm_lm_adaptive import DualStateConfig, DualStateModel


def make_cfg(**overrides):
    base = dict(
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
    base.update(overrides)
    return DualStateConfig(**base)


def fixed_input():
    torch.manual_seed(0)
    ids = torch.randint(1, 127, (2, 10))
    return ids, torch.ones_like(ids)


def test_overwrite_has_no_stability_parameters_and_strict_loads():
    cfg = make_cfg()
    model = DualStateModel(cfg)
    assert model.gate_H is None and model.gate_L is None
    assert model.alpha_H is None and model.alpha_L is None
    clone = DualStateModel(cfg)
    clone.load_state_dict(model.state_dict(), strict=True)


def test_depth_one_is_unchanged_by_gate():
    base = DualStateModel(make_cfg())
    gated = DualStateModel(make_cfg(update_mode="gated", stable_update_from_depth=2))
    missing, unexpected = gated.load_state_dict(base.state_dict(), strict=False)
    assert all(k.startswith(("gate_", "alpha_")) for k in missing)
    assert unexpected == []
    ids, mask = fixed_input()
    base_depth1 = base(ids, mask, mode="fixed", n_high_cycles=1).logits
    gated_depth1 = gated(ids, mask, mode="fixed", n_high_cycles=1).logits
    torch.testing.assert_close(base_depth1, gated_depth1, atol=0.0, rtol=0.0)


def test_gate_changes_depth_four_at_neutral_initialization():
    base = DualStateModel(make_cfg())
    gated = DualStateModel(make_cfg(update_mode="gated", stable_update_from_depth=2))
    gated.load_state_dict(base.state_dict(), strict=False)
    ids, mask = fixed_input()
    base4 = base(ids, mask, mode="fixed", n_high_cycles=4).logits
    gated4 = gated(ids, mask, mode="fixed", n_high_cycles=4).logits
    assert not torch.allclose(base4, gated4)


def test_gate_can_preserve_and_can_refine():
    base = DualStateModel(make_cfg())
    ids, mask = fixed_input()
    depth1 = base(ids, mask, mode="fixed", n_high_cycles=1).logits
    depth4 = base(ids, mask, mode="fixed", n_high_cycles=4).logits

    preserve = DualStateModel(make_cfg(update_mode="gated", stable_update_from_depth=2))
    preserve.load_state_dict(base.state_dict(), strict=False)
    with torch.no_grad():
        preserve.gate_H.bias.fill_(-20.0)
        preserve.gate_L.bias.fill_(-20.0)
    preserved4 = preserve(ids, mask, mode="fixed", n_high_cycles=4).logits
    torch.testing.assert_close(preserved4, depth1, atol=1e-5, rtol=1e-4)

    refine = DualStateModel(make_cfg(update_mode="gated", stable_update_from_depth=2))
    refine.load_state_dict(base.state_dict(), strict=False)
    with torch.no_grad():
        refine.gate_H.bias.fill_(20.0)
        refine.gate_L.bias.fill_(20.0)
    refined4 = refine(ids, mask, mode="fixed", n_high_cycles=4).logits
    torch.testing.assert_close(refined4, depth4, atol=1e-4, rtol=1e-3)


def test_residual_gate_is_finite_and_parameterized():
    model = DualStateModel(
        make_cfg(
            update_mode="residual",
            stable_update_from_depth=2,
            residual_alpha_init=0.25,
        )
    )
    ids, mask = fixed_input()
    out = model(ids, mask, mode="fixed", n_high_cycles=4)
    assert torch.isfinite(out.logits).all()
    assert model.alpha_H is not None and model.alpha_L is not None
    breakdown = model.parameter_breakdown()
    assert breakdown["update_stability"] == 2
    # The alpha is initialized to sigmoid(logit(0.25)) = 0.25.
    assert abs(float(torch.sigmoid(model.alpha_H)) - 0.25) < 1e-5
