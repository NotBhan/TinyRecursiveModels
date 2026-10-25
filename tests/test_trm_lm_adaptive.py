from __future__ import annotations

import copy
import os

import pytest
import torch
from torch import nn

from models.recursive_reasoning.trm_lm_adaptive import (
    STOP_BUDGET_EXHAUSTED,
    STOP_FIXED_DEPTH,
    STOP_LEARNED_HALT,
    STOP_MAX_REACHED,
    STOP_MINIMUM_REACHED,
    STOP_PADDING,
    DualStateConfig,
    DualStateModel,
    per_position_cross_entropy,
    per_sequence_cross_entropy,
)


class ConstantController(nn.Module):
    """Deterministic per-position mock controller."""

    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value
        self.calls = 0

    def forward(
        self, z_H, z_L, attention_mask, depth, max_depth,
        remaining_budget_fraction=None,
    ):
        self.calls += 1
        return torch.full(
            z_H.shape[:2], self.value, device=z_H.device, dtype=z_H.dtype
        )


class VectorController(nn.Module):
    """Mock controller returning one constant per sequence, per position."""

    def __init__(self, values) -> None:
        super().__init__()
        self.values = list(values)

    def forward(
        self, z_H, z_L, attention_mask, depth, max_depth,
        remaining_budget_fraction=None,
    ):
        values = torch.tensor(
            self.values, device=z_H.device, dtype=z_H.dtype
        )
        return values.view(-1, 1).expand(-1, z_H.shape[1])


class PositionValuesController(nn.Module):
    """Causal mock controller returning one constant per sequence position."""

    def __init__(self, values) -> None:
        super().__init__()
        self.values = list(values)

    def forward(
        self, z_H, z_L, attention_mask, depth, max_depth,
        remaining_budget_fraction=None,
    ):
        values = torch.tensor(
            self.values, device=z_H.device, dtype=z_H.dtype
        )
        return values.view(1, -1).expand(z_H.shape[0], -1)


@pytest.fixture()
def small_config() -> DualStateConfig:
    return DualStateConfig(
        vocab_size=64,
        seq_len=16,
        hidden_size=32,
        num_heads=4,
        n_low_layers=1,
        n_high_layers=1,
        n_low_cycles=1,
        n_high_cycles=2,
        min_high_cycles=1,
        max_high_cycles=3,
        controller_hidden_size=16,
        tie_embeddings=True,
        pad_token_id=0,
    )


@pytest.fixture()
def small_model(small_config: DualStateConfig) -> DualStateModel:
    torch.manual_seed(0)
    return DualStateModel(small_config)


@pytest.fixture()
def batch():
    generator = torch.Generator().manual_seed(7)
    inputs = torch.randint(1, 64, (2, 16), generator=generator)
    labels = torch.full((2, 16), -100, dtype=torch.long)
    labels[:, :-1] = inputs[:, 1:]
    mask = torch.ones((2, 16), dtype=torch.long)
    return {"input_ids": inputs, "labels": labels, "attention_mask": mask}


@pytest.fixture()
def padded_batch(batch):
    batch = {k: v.clone() for k, v in batch.items()}
    batch["input_ids"][:, 12:] = 0
    batch["attention_mask"][:, 12:] = 0
    batch["labels"][:, 11:] = -100
    return batch


# ---------------------------------------------------------------------------
# Configuration and architecture
# ---------------------------------------------------------------------------

def test_config_validation() -> None:
    with pytest.raises(ValueError):
        DualStateConfig(
            vocab_size=64, seq_len=16, hidden_size=32, num_heads=4,
            min_high_cycles=4, n_high_cycles=2, max_high_cycles=3,
        )
    with pytest.raises(ValueError):
        DualStateConfig(
            vocab_size=64, seq_len=16, hidden_size=32, num_heads=4,
            halt_threshold=1.5,
        )
    with pytest.raises(ValueError):
        DualStateConfig(
            vocab_size=64, seq_len=16, hidden_size=32, num_heads=4,
            gradient_strategy="invalid",
        )
    with pytest.raises(ValueError):
        DualStateConfig(
            vocab_size=64, seq_len=16, hidden_size=32, num_heads=4,
            n_low_cycles=1, min_high_cycles=1,
            max_total_refinement_steps=1,
        )


def test_shapes_states_and_counts(small_model, batch) -> None:
    out = small_model(
        batch["input_ids"], batch["attention_mask"],
        mode="fixed", n_high_cycles=2, n_low_cycles=1,
    )
    assert out.logits.shape == (2, 16, 64)
    assert out.z_H.shape == out.z_L.shape == (2, 16, 32)
    assert out.high_cycles.tolist() == [2, 2]
    assert out.low_cycles.tolist() == [2, 2]
    assert out.transitions.tolist() == [4, 4]
    assert out.stopping_reason == ["fixed_depth", "fixed_depth"]
    assert out.high_cycles_per_position.tolist() == [[2] * 16, [2] * 16]
    assert out.low_cycles_per_position.tolist() == [[2] * 16, [2] * 16]
    assert out.transitions_per_position.tolist() == [[4] * 16, [4] * 16]
    assert bool((out.stop_reason_codes == STOP_FIXED_DEPTH).all())
    assert out.block_calls == 4  # 2 cycles x (1 low + 1 high block)
    assert out.controller_calls == 0
    assert out.active_counts_per_step == [32, 32]


def test_shared_parameter_reuse_and_constant_count(small_config) -> None:
    import dataclasses
    config_a = dataclasses.replace(small_config, max_high_cycles=2)
    config_b = dataclasses.replace(small_config, max_high_cycles=4)
    model_a = DualStateModel(config_a)
    model_b = DualStateModel(config_b)
    assert model_a.num_parameters == model_b.num_parameters
    assert len(model_a.F_L) == 1
    assert len(model_a.F_H) == 1

    calls = {"L": 0, "H": 0}
    hook_l = model_a.F_L[0].register_forward_hook(
        lambda module, inputs, output: calls.__setitem__("L", calls["L"] + 1)
    )
    hook_h = model_a.F_H[0].register_forward_hook(
        lambda module, inputs, output: calls.__setitem__("H", calls["H"] + 1)
    )
    try:
        with torch.no_grad():
            model_a(
                torch.randint(1, 64, (1, 16)), torch.ones(1, 16),
                mode="fixed", n_high_cycles=2, n_low_cycles=1,
            )
        assert calls == {"L": 2, "H": 2}
        calls["L"] = calls["H"] = 0
        with torch.no_grad():
            model_a(
                torch.randint(1, 64, (1, 16)), torch.ones(1, 16),
                mode="fixed", n_high_cycles=2, n_low_cycles=2,
            )
        assert calls == {"L": 4, "H": 2}
    finally:
        hook_l.remove()
        hook_h.remove()


def test_share_high_low_blocks(small_config) -> None:
    import dataclasses
    config = dataclasses.replace(small_config, share_high_low_blocks=True)
    model = DualStateModel(config)
    assert model.F_H is None
    out = model(
        torch.randint(1, 64, (1, 16)), torch.ones(1, 16),
        mode="fixed", n_high_cycles=2,
    )
    assert out.logits.shape == (1, 16, 64)
    assert model.parameter_breakdown()["F_H"] == 0


def test_recursion_depth_changes_output(small_model, batch) -> None:
    small_model.eval()
    with torch.no_grad():
        out1 = small_model(
            batch["input_ids"], batch["attention_mask"],
            mode="fixed", n_high_cycles=1, n_low_cycles=1,
        ).logits
        out2 = small_model(
            batch["input_ids"], batch["attention_mask"],
            mode="fixed", n_high_cycles=2, n_low_cycles=1,
        ).logits
        out3 = small_model(
            batch["input_ids"], batch["attention_mask"],
            mode="fixed", n_high_cycles=2, n_low_cycles=2,
        ).logits
    assert not torch.allclose(out1, out2, atol=1e-6)
    assert not torch.allclose(out2, out3, atol=1e-6)


def test_both_latent_projections_affect_logits(small_model, batch) -> None:
    small_model.eval()
    state = copy.deepcopy(small_model.state_dict())
    with torch.no_grad():
        baseline = small_model(
            batch["input_ids"], batch["attention_mask"],
            mode="fixed", n_high_cycles=2,
        ).logits
        model_l = DualStateModel(small_model.config)
        model_l.load_state_dict(state)
        model_l.P_L.weight.zero_()
        no_l = model_l(
            batch["input_ids"], batch["attention_mask"],
            mode="fixed", n_high_cycles=2,
        ).logits
        model_h = DualStateModel(small_model.config)
        model_h.load_state_dict(state)
        model_h.P_H.weight.zero_()
        no_h = model_h(
            batch["input_ids"], batch["attention_mask"],
            mode="fixed", n_high_cycles=2,
        ).logits
    assert not torch.allclose(baseline, no_l, atol=1e-6)
    assert not torch.allclose(baseline, no_h, atol=1e-6)


def test_min_max_and_budget_validation(small_model, batch) -> None:
    with pytest.raises(ValueError):
        small_model(
            batch["input_ids"], batch["attention_mask"],
            mode="fixed", n_high_cycles=0,
        )
    with pytest.raises(ValueError):
        small_model(
            batch["input_ids"], batch["attention_mask"],
            mode="fixed", n_high_cycles=99,
        )
    with pytest.raises(ValueError):
        small_model(
            batch["input_ids"], batch["attention_mask"],
            mode="adaptive", max_high_cycles=0,
        )
    with pytest.raises(ValueError):
        small_model(
            batch["input_ids"], batch["attention_mask"],
            mode="budgeted", max_total_transitions=None,
        )
    with pytest.raises(ValueError):
        small_model(
            batch["input_ids"], batch["attention_mask"],
            mode="budgeted", max_total_transitions=1,
        )


def test_parameter_breakdown_sums_to_total(small_model) -> None:
    breakdown = small_model.parameter_breakdown()
    components = sum(
        value for key, value in breakdown.items() if key != "total_dedup"
    )
    assert components == breakdown["total_dedup"] == small_model.num_parameters


# ---------------------------------------------------------------------------
# Causal and padding behavior
# ---------------------------------------------------------------------------

def test_causal_future_token_invariance(small_model, batch) -> None:
    small_model.eval()
    with torch.no_grad():
        base = small_model(
            batch["input_ids"], batch["attention_mask"],
            mode="fixed", n_high_cycles=2,
        ).logits
        perturbed = batch["input_ids"].clone()
        perturbed[:, -1] = (perturbed[:, -1] + 1) % 64
        changed = small_model(
            perturbed, batch["attention_mask"],
            mode="fixed", n_high_cycles=2,
        ).logits
    assert torch.allclose(base[:, :-1], changed[:, :-1], atol=1e-6)
    assert not torch.allclose(base[:, -1], changed[:, -1], atol=1e-6)


def test_padding_invariance(small_model, padded_batch) -> None:
    small_model.eval()
    with torch.no_grad():
        base = small_model(
            padded_batch["input_ids"], padded_batch["attention_mask"],
            mode="fixed", n_high_cycles=2,
        ).logits
        changed = padded_batch["input_ids"].clone()
        changed[:, 12:] = 1
        altered = small_model(
            changed, padded_batch["attention_mask"],
            mode="fixed", n_high_cycles=2,
        ).logits
    assert torch.allclose(base[:, :12], altered[:, :12], atol=1e-6)


def test_mixed_lengths_are_finite(small_model, batch) -> None:
    mask = batch["attention_mask"].clone()
    mask[0, 10:] = 0
    with torch.no_grad():
        out = small_model(
            batch["input_ids"], mask, mode="fixed", n_high_cycles=2
        )
    assert torch.isfinite(out.logits).all()


def test_controller_padding_invariance(small_model, padded_batch) -> None:
    small_model.eval()
    valid = padded_batch["attention_mask"].bool()
    with torch.no_grad():
        out_a = small_model.forward_depths(
            padded_batch["input_ids"], padded_batch["attention_mask"],
            depths=[1, 2], max_depth=3, evaluate_controller=True,
        )
        changed = padded_batch["input_ids"].clone()
        changed[:, 12:] = 1
        out_b = small_model.forward_depths(
            changed, padded_batch["attention_mask"],
            depths=[1, 2], max_depth=3, evaluate_controller=True,
        )
    for depth in (1, 2):
        assert torch.allclose(
            out_a.controller_logits_by_depth[depth][valid],
            out_b.controller_logits_by_depth[depth][valid],
            atol=1e-6,
        )


# ---------------------------------------------------------------------------
# Gradients and gradient strategies
# ---------------------------------------------------------------------------

def test_full_bptt_gradients(small_model, batch) -> None:
    small_model.zero_grad(set_to_none=True)
    out = small_model.forward_depths(
        batch["input_ids"], batch["attention_mask"],
        depths=[1, 2, 3], max_depth=3,
    )
    loss = torch.nn.functional.cross_entropy(
        out.logits_by_depth[3].reshape(-1, 64),
        batch["labels"].reshape(-1), ignore_index=-100,
    )
    loss.backward()
    for name in ["P_H", "P_L"]:
        weight = getattr(small_model, name).weight.grad
        assert weight is not None and torch.isfinite(weight).all()
        assert weight.abs().sum() > 0
    for module in (small_model.F_L, small_model.F_H):
        for parameter in module.parameters():
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
            assert parameter.grad.abs().sum() > 0
    assert small_model.embed_tokens.weight.grad is not None
    assert small_model.lm_head.weight.grad is not None


def test_truncated_gradient(small_config, batch) -> None:
    import dataclasses
    config = dataclasses.replace(small_config, gradient_strategy="truncated")
    model = DualStateModel(config)
    model.zero_grad(set_to_none=True)
    out = model.forward_depths(
        batch["input_ids"], batch["attention_mask"],
        depths=[1, 2, 3], max_depth=3,
    )
    loss = torch.nn.functional.cross_entropy(
        out.logits_by_depth[3].reshape(-1, 64),
        batch["labels"].reshape(-1), ignore_index=-100,
    )
    loss.backward()
    for name in ["P_H", "P_L"]:
        grad = getattr(model, name).weight.grad
        assert grad is None or grad.abs().sum() == 0
    for module in (model.F_L, model.F_H):
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in module.parameters()
        )


def test_last_high_cycle_gradient(small_config, batch) -> None:
    import dataclasses
    config = dataclasses.replace(
        small_config, gradient_strategy="last_high_cycle"
    )
    model = DualStateModel(config)
    model.zero_grad(set_to_none=True)
    out = model.forward_depths(
        batch["input_ids"], batch["attention_mask"],
        depths=[1, 2, 3], max_depth=3,
    )
    loss = torch.nn.functional.cross_entropy(
        out.logits_by_depth[3].reshape(-1, 64),
        batch["labels"].reshape(-1), ignore_index=-100,
    )
    loss.backward()
    for name in ["P_H", "P_L"]:
        grad = getattr(model, name).weight.grad
        assert grad is None or grad.abs().sum() == 0
    for module in (model.F_L, model.F_H):
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in module.parameters()
        )


def test_controller_loss_gradients(small_model, batch) -> None:
    small_model.zero_grad(set_to_none=True)
    out = small_model.forward_depths(
        batch["input_ids"], batch["attention_mask"],
        depths=[1, 2, 3], max_depth=3,
    )
    logits = out.controller_logits_by_depth[1]
    target = torch.ones_like(logits)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
    loss.backward()
    controller_grads = [
        p.grad for p in small_model.controller.parameters()
        if p.grad is not None
    ]
    assert controller_grads
    assert all(torch.isfinite(g).all() for g in controller_grads)
    assert any(g.abs().sum() > 0 for g in controller_grads)
    for name in ["P_H", "P_L"]:
        grad = getattr(small_model, name).weight.grad
        assert grad is None or grad.abs().sum() == 0


def test_per_sequence_cross_entropy() -> None:
    torch.manual_seed(0)
    logits = torch.randn(2, 4, 8)
    labels = torch.tensor([
        [1, 2, 3, -100],
        [4, -100, -100, -100],
    ])
    per_seq = per_sequence_cross_entropy(logits, labels)
    manual_0 = torch.nn.functional.cross_entropy(
        logits[0, :3].reshape(-1, 8), labels[0, :3]
    )
    manual_1 = torch.nn.functional.cross_entropy(
        logits[1, :1].reshape(-1, 8), labels[1, :1]
    )
    assert torch.allclose(per_seq[0], manual_0, atol=1e-6)
    assert torch.allclose(per_seq[1], manual_1, atol=1e-6)


# ---------------------------------------------------------------------------
# Halting and inference modes
# ---------------------------------------------------------------------------

def test_controller_threshold_uses_probability(small_model, batch) -> None:
    """The controller emits logits; inference must threshold sigmoid(logit)."""
    small_model.eval()
    small_model.controller = ConstantController(0.0)
    with torch.no_grad():
        out = small_model(
            batch["input_ids"], batch["attention_mask"],
            mode="adaptive", max_high_cycles=3, halt_threshold=0.5,
        )
    assert out.high_cycles.tolist() == [3, 3]
    assert out.halt_probs[0].shape == (2, 16)
    assert torch.allclose(
        out.halt_probs[0], torch.full((2, 16), 0.5), atol=1e-6
    )

    small_model.controller = ConstantController(-0.1)
    with torch.no_grad():
        out = small_model(
            batch["input_ids"], batch["attention_mask"],
            mode="adaptive", max_high_cycles=3, halt_threshold=0.5,
        )
    assert out.high_cycles.tolist() == [1, 1]
    assert out.stopping_reason == ["learned_halt", "learned_halt"]


def test_adaptive_teacher_forcing_is_causal(small_model, batch) -> None:
    """Future tokens must not change earlier logits or decisions."""
    small_model.eval()
    ids = batch["input_ids"]
    mask = batch["attention_mask"]
    with torch.no_grad():
        base = small_model(
            ids, mask, mode="adaptive", max_high_cycles=3
        )
        base_depths = small_model.forward_depths(
            ids, mask, depths=[1, 2], max_depth=3,
            evaluate_controller=True,
        )
        for position in (8, 15):
            for token in range(1, 64, 5):
                perturbed = ids.clone()
                perturbed[:, position] = token
                out = small_model(
                    perturbed, mask, mode="adaptive", max_high_cycles=3
                )
                assert torch.allclose(
                    base.logits[:, :position],
                    out.logits[:, :position],
                    atol=1e-6,
                ), (position, token)
                assert torch.equal(
                    base.high_cycles_per_position[:, :position],
                    out.high_cycles_per_position[:, :position],
                ), (position, token)
                assert torch.equal(
                    base.stop_reason_codes[:, :position],
                    out.stop_reason_codes[:, :position],
                ), (position, token)
                pert_depths = small_model.forward_depths(
                    perturbed, mask, depths=[1, 2], max_depth=3,
                    evaluate_controller=True,
                )
                for depth in (1, 2):
                    assert torch.allclose(
                        base_depths.controller_logits_by_depth[depth][
                            :, :position
                        ],
                        pert_depths.controller_logits_by_depth[depth][
                            :, :position
                        ],
                        atol=1e-6,
                    ), (position, token, depth)


def test_adaptive_teacher_forcing_mixed_lengths(small_model, batch) -> None:
    """Causality holds near padding and in mixed-length batches."""
    small_model.eval()
    ids = batch["input_ids"]
    mask = batch["attention_mask"].clone()
    mask[0, 10:] = 0
    with torch.no_grad():
        base = small_model(ids, mask, mode="adaptive", max_high_cycles=3)
        for position in (5, 9):
            for token in range(1, 64, 9):
                perturbed = ids.clone()
                perturbed[:, position] = token
                out = small_model(
                    perturbed, mask, mode="adaptive", max_high_cycles=3
                )
                assert torch.allclose(
                    base.logits[:, :position],
                    out.logits[:, :position],
                    atol=1e-6,
                ), (position, token)
                assert torch.equal(
                    base.high_cycles_per_position[:, :position],
                    out.high_cycles_per_position[:, :position],
                ), (position, token)


def test_positionwise_depths_and_transitions(small_model, batch) -> None:
    small_model.eval()
    values = [10.0, 10.0] + [-10.0] * 14
    small_model.controller = PositionValuesController(values)
    with torch.no_grad():
        out = small_model(
            batch["input_ids"], batch["attention_mask"],
            mode="adaptive", max_high_cycles=3,
        )
    expected_high = [3, 3] + [1] * 14
    assert out.high_cycles_per_position.tolist() == [
        expected_high, expected_high
    ]
    assert out.transitions_per_position.tolist() == [
        [h * 2 for h in expected_high],
        [h * 2 for h in expected_high],
    ]
    expected_codes = [STOP_MAX_REACHED, STOP_MAX_REACHED] + [
        STOP_LEARNED_HALT
    ] * 14
    assert out.stop_reason_codes[0].tolist() == expected_codes
    assert out.active_counts_per_step == [32, 4, 4]
    assert out.block_calls == 6
    assert out.controller_calls == 2
    assert out.high_cycles.tolist() == [3, 3]
    assert out.transitions.tolist() == [6, 6]
    assert out.stopping_reason == ["mixed", "mixed"]
    assert len(out.halt_probs) == 2
    assert out.halt_probs[0].shape == (2, 16)


def test_state_retention_after_early_halt(small_model, batch) -> None:
    """Later updates must not overwrite an earlier halted position."""
    small_model.eval()
    values = [-10.0] * 4 + [10.0] * 12
    small_model.controller = PositionValuesController(values)
    ids = batch["input_ids"]
    mask = batch["attention_mask"]
    with torch.no_grad():
        base = small_model(ids, mask, mode="adaptive", max_high_cycles=3)
        for position in (4, 8, 15):
            for token in range(1, 64, 11):
                perturbed = ids.clone()
                perturbed[:, position] = token
                out = small_model(
                    perturbed, mask, mode="adaptive", max_high_cycles=3
                )
                assert torch.allclose(
                    base.logits[:, :4], out.logits[:, :4], atol=1e-6
                ), (position, token)
                assert torch.allclose(
                    base.z_H[:, :4], out.z_H[:, :4], atol=1e-6
                ), (position, token)
                assert torch.allclose(
                    base.z_L[:, :4], out.z_L[:, :4], atol=1e-6
                ), (position, token)
                assert torch.equal(
                    base.high_cycles_per_position[:, :4],
                    out.high_cycles_per_position[:, :4],
                ), (position, token)
                assert torch.equal(
                    base.stop_reason_codes[:, :4],
                    out.stop_reason_codes[:, :4],
                ), (position, token)


def test_fixed_depth_equivalence_adaptive_forced_continue(
    small_model, batch
) -> None:
    small_model.eval()
    ids = batch["input_ids"]
    mask = batch["attention_mask"]
    with torch.no_grad():
        fixed = small_model(ids, mask, mode="fixed", n_high_cycles=3).logits
        adaptive = small_model(
            ids, mask, mode="adaptive", max_high_cycles=3,
            halt_threshold=0.0,
        ).logits
        fast = small_model(ids, mask, mode="fast").logits
        fixed_one = small_model(
            ids, mask, mode="fixed", n_high_cycles=1
        ).logits
    assert torch.allclose(fixed, adaptive, atol=1e-6)
    assert torch.allclose(fast, fixed_one, atol=1e-6)


def test_padding_positions_are_inactive(small_model, padded_batch) -> None:
    small_model.eval()
    valid = padded_batch["attention_mask"].bool()
    with torch.no_grad():
        base = small_model(
            padded_batch["input_ids"], padded_batch["attention_mask"],
            mode="adaptive", max_high_cycles=3,
        )
        changed_ids = padded_batch["input_ids"].clone()
        changed_ids[:, 12:] = 1
        out = small_model(
            changed_ids, padded_batch["attention_mask"],
            mode="adaptive", max_high_cycles=3,
        )
    assert bool((base.high_cycles_per_position[~valid] == 0).all())
    assert bool((base.transitions_per_position[~valid] == 0).all())
    assert bool((base.stop_reason_codes[~valid] == STOP_PADDING).all())
    assert bool((base.stop_reason_codes[valid] != STOP_PADDING).all())
    assert base.active_counts_per_step[0] == int(valid.sum())
    assert torch.allclose(base.logits[valid], out.logits[valid], atol=1e-6)
    assert torch.equal(
        base.high_cycles_per_position[valid],
        out.high_cycles_per_position[valid],
    )
    assert torch.equal(
        base.stop_reason_codes[valid], out.stop_reason_codes[valid]
    )


def test_block_call_counter_matches_hooks(small_model, batch) -> None:
    small_model.eval()
    calls = {"L": 0, "H": 0}
    hook_l = small_model.F_L[0].register_forward_hook(
        lambda *args: calls.__setitem__("L", calls["L"] + 1)
    )
    hook_h = small_model.F_H[0].register_forward_hook(
        lambda *args: calls.__setitem__("H", calls["H"] + 1)
    )
    try:
        small_model.controller = PositionValuesController(
            [-10.0] * 4 + [10.0] * 12
        )
        with torch.no_grad():
            out = small_model(
                batch["input_ids"], batch["attention_mask"],
                mode="adaptive", max_high_cycles=3,
            )
    finally:
        hook_l.remove()
        hook_h.remove()
    assert out.block_calls == calls["L"] + calls["H"]
    assert calls["L"] == calls["H"] == len(out.active_counts_per_step)
    assert out.block_calls == len(out.active_counts_per_step) * 2
    assert out.controller_calls == len(out.halt_probs)


def test_per_position_cross_entropy() -> None:
    torch.manual_seed(0)
    logits = torch.randn(2, 4, 8)
    labels = torch.tensor([
        [1, 2, 3, -100],
        [4, -100, -100, -100],
    ])
    per_position = per_position_cross_entropy(logits, labels)
    manual = torch.nn.functional.cross_entropy(
        logits.float().reshape(-1, 8),
        labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).view(2, 4)
    manual = manual * (labels != -100)
    assert torch.allclose(per_position, manual, atol=1e-6)
    assert float(per_position[0, 3]) == 0.0
    assert float(per_position[1, 1]) == 0.0


def test_adaptive_immediate_halt(small_model, batch) -> None:
    small_model.eval()
    small_model.controller = ConstantController(-10.0)
    with torch.no_grad():
        out = small_model(
            batch["input_ids"], batch["attention_mask"],
            mode="adaptive", max_high_cycles=3, n_low_cycles=1,
        )
    assert out.high_cycles.tolist() == [1, 1]
    assert out.transitions.tolist() == [2, 2]
    assert out.stopping_reason == ["learned_halt", "learned_halt"]
    assert len(out.halt_probs) == 1
    assert out.active_counts_per_step == [32]
    assert out.block_calls == 2
    assert out.controller_calls == 1
    assert bool((out.stop_reason_codes == STOP_LEARNED_HALT).all())
    assert out.high_cycles_per_position.tolist() == [[1] * 16, [1] * 16]


def test_adaptive_continue_to_max(small_model, batch) -> None:
    small_model.eval()
    small_model.controller = ConstantController(10.0)
    with torch.no_grad():
        out = small_model(
            batch["input_ids"], batch["attention_mask"],
            mode="adaptive", max_high_cycles=3, n_low_cycles=1,
        )
    assert out.high_cycles.tolist() == [3, 3]
    assert out.transitions.tolist() == [6, 6]
    assert out.stopping_reason == ["max_reached", "max_reached"]
    assert len(out.halt_probs) == 2
    assert out.active_counts_per_step == [32, 32, 32]
    assert out.block_calls == 6
    assert out.controller_calls == 2
    assert bool((out.stop_reason_codes == STOP_MAX_REACHED).all())


def test_adaptive_mixed_batch_per_sequence_halting(small_model, batch) -> None:
    small_model.eval()
    small_model.controller = VectorController([-10.0, 10.0])
    with torch.no_grad():
        out = small_model(
            batch["input_ids"], batch["attention_mask"],
            mode="adaptive", max_high_cycles=3, n_low_cycles=1,
        )
    assert out.high_cycles.tolist() == [1, 3]
    assert out.transitions.tolist() == [2, 6]
    assert out.stopping_reason == ["learned_halt", "max_reached"]
    assert out.active_counts_per_step == [32, 16, 16]
    assert out.block_calls == 6
    assert out.controller_calls == 2
    assert out.stop_reason_codes[0].tolist() == [STOP_LEARNED_HALT] * 16
    assert out.stop_reason_codes[1].tolist() == [STOP_MAX_REACHED] * 16


def test_budgeted_stops_at_budget(small_model, batch) -> None:
    small_model.eval()
    small_model.controller = ConstantController(10.0)
    with torch.no_grad():
        out = small_model(
            batch["input_ids"], batch["attention_mask"],
            mode="budgeted", max_high_cycles=3, n_low_cycles=1,
            max_total_transitions=4,
        )
    assert out.high_cycles.tolist() == [2, 2]
    assert out.transitions.tolist() == [4, 4]
    assert out.stopping_reason == ["budget_exhausted", "budget_exhausted"]
    assert out.active_counts_per_step == [32, 32]
    assert out.block_calls == 4
    assert out.controller_calls == 2
    assert bool((out.stop_reason_codes == STOP_BUDGET_EXHAUSTED).all())
    assert out.high_cycles_per_position.tolist() == [[2] * 16, [2] * 16]


def test_fixed_mode_ignores_controller(small_model, batch) -> None:
    small_model.eval()
    small_model.controller = ConstantController(-10.0)
    with torch.no_grad():
        out = small_model(
            batch["input_ids"], batch["attention_mask"],
            mode="fixed", n_high_cycles=3, n_low_cycles=1,
        )
    assert out.high_cycles.tolist() == [3, 3]
    assert out.transitions.tolist() == [6, 6]
    assert out.stopping_reason == ["fixed_depth", "fixed_depth"]
    assert out.halt_probs == []
    assert out.controller_calls == 0
    assert out.active_counts_per_step == [32, 32, 32]
    assert out.block_calls == 6
    assert bool((out.stop_reason_codes == STOP_FIXED_DEPTH).all())


def test_fast_mode_minimum(small_model, batch) -> None:
    small_model.eval()
    with torch.no_grad():
        out = small_model(
            batch["input_ids"], batch["attention_mask"], mode="fast"
        )
    assert out.high_cycles.tolist() == [1, 1]
    assert out.transitions.tolist() == [2, 2]
    assert out.stopping_reason == ["minimum_reached", "minimum_reached"]
    assert out.controller_calls == 0
    assert out.active_counts_per_step == [32]
    assert out.block_calls == 2
    assert bool((out.stop_reason_codes == STOP_MINIMUM_REACHED).all())


def test_stopping_decisions_are_deterministic(small_model, batch) -> None:
    small_model.eval()
    small_model.controller = ConstantController(10.0)
    with torch.no_grad():
        a = small_model(
            batch["input_ids"], batch["attention_mask"],
            mode="adaptive", max_high_cycles=3,
        )
        b = small_model(
            batch["input_ids"], batch["attention_mask"],
            mode="adaptive", max_high_cycles=3,
        )
    assert torch.allclose(a.logits, b.logits)
    assert a.high_cycles.tolist() == b.high_cycles.tolist()
    assert a.stopping_reason == b.stopping_reason
    assert torch.equal(
        a.high_cycles_per_position, b.high_cycles_per_position
    )
    assert torch.equal(a.stop_reason_codes, b.stop_reason_codes)


def test_generation_modes(small_model) -> None:
    small_model.eval()
    small_model.controller = ConstantController(10.0)
    prompt = torch.tensor([[2, 10, 11, 12]])
    specs = {
        "fast": {},
        "fixed": {"n_high_cycles": 3},
        "adaptive": {"max_high_cycles": 3},
        "budgeted": {"max_high_cycles": 3, "max_total_transitions": 4},
    }
    for mode, kwargs in specs.items():
        out = small_model.generate(
            prompt, max_new_tokens=3, eos_token_id=None,
            mode=mode, n_low_cycles=1, **kwargs,
        )
        assert out.tokens.shape == (1, 7)
        assert out.new_token_count == 3
        assert len(out.transitions_per_step) == 3
        assert len(out.position_transitions_per_step) == 3
        assert len(out.stopping_reasons) == 3
        assert len(out.position_high_cycles_per_step) == 3
        assert len(out.block_calls_per_step) == 3
        assert len(out.controller_calls_per_step) == 3
        assert out.total_transitions == sum(out.transitions_per_step)
        for position_counts in out.position_high_cycles_per_step:
            assert len(position_counts) == 1
            assert len(position_counts[0]) >= 1


# ---------------------------------------------------------------------------
# Checkpoint and GPU smoke
# ---------------------------------------------------------------------------

def test_checkpoint_roundtrip(small_model, batch, tmp_path) -> None:
    small_model.eval()
    with torch.no_grad():
        expected = small_model(
            batch["input_ids"], batch["attention_mask"],
            mode="fixed", n_high_cycles=2,
        ).logits
    path = tmp_path / "dual_state.pt"
    torch.save(small_model.state_dict(), path)
    fresh = DualStateModel(small_model.config)
    fresh.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    fresh.eval()
    with torch.no_grad():
        actual = fresh(
            batch["input_ids"], batch["attention_mask"],
            mode="fixed", n_high_cycles=2,
        ).logits
    assert torch.allclose(expected, actual, atol=1e-6)

    with torch.no_grad():
        expected_adaptive = small_model(
            batch["input_ids"], batch["attention_mask"],
            mode="adaptive", max_high_cycles=3,
        )
        actual_adaptive = fresh(
            batch["input_ids"], batch["attention_mask"],
            mode="adaptive", max_high_cycles=3,
        )
    assert torch.allclose(
        expected_adaptive.logits, actual_adaptive.logits, atol=1e-6
    )
    assert torch.equal(
        expected_adaptive.high_cycles_per_position,
        actual_adaptive.high_cycles_per_position,
    )
    assert torch.equal(
        expected_adaptive.stop_reason_codes,
        actual_adaptive.stop_reason_codes,
    )


def _gpu_device() -> torch.device:
    if os.environ.get("TRM_LM_ADAPTIVE_TEST_GPU") != "1":
        pytest.skip("set TRM_LM_ADAPTIVE_TEST_GPU=1 for the opt-in GPU test")
    if not torch.cuda.is_available():
        pytest.skip("torch.cuda.is_available() is False")
    properties = torch.cuda.get_device_properties(0)
    arch = str(getattr(properties, "gcnArchName", ""))
    if (
        "gfx1032" in arch
        and "gfx1032" not in torch.cuda.get_arch_list()
        and "HSA_OVERRIDE_GFX_VERSION" not in os.environ
    ):
        pytest.skip("gfx1032 requires HSA_OVERRIDE_GFX_VERSION=10.3.0")
    return torch.device("cuda:0")


@pytest.mark.skipif(
    os.environ.get("TRM_LM_ADAPTIVE_TEST_GPU") != "1",
    reason="opt-in GPU test",
)
def test_gpu_forward_backward(small_config, batch) -> None:
    device = _gpu_device()
    torch.cuda.reset_peak_memory_stats(device)
    model = DualStateModel(small_config).to(device=device, dtype=torch.bfloat16)
    model.rotary_emb.float()
    batch_dev = {k: v.to(device) for k, v in batch.items()}
    model.zero_grad(set_to_none=True)
    out = model.forward_depths(
        batch_dev["input_ids"], batch_dev["attention_mask"],
        depths=[1, 2], max_depth=2,
    )
    loss = torch.nn.functional.cross_entropy(
        out.logits_by_depth[2].float().reshape(-1, 64),
        batch_dev["labels"].reshape(-1), ignore_index=-100,
    )
    controller_target = torch.ones_like(out.controller_logits_by_depth[1])
    controller_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        out.controller_logits_by_depth[1], controller_target
    )
    (loss + 0.1 * controller_loss).backward()
    torch.cuda.synchronize()
    assert torch.isfinite(loss)
    assert torch.isfinite(controller_loss)
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in model.parameters() if p.requires_grad
    )
    print(
        f"\n[gpu] dual_state loss={float(loss.detach()):.4f} "
        f"peak_alloc={torch.cuda.max_memory_allocated(device) / 1e9:.3f} GB",
        flush=True,
    )


@pytest.mark.skipif(
    os.environ.get("TRM_LM_ADAPTIVE_TEST_GPU") != "1",
    reason="opt-in GPU test",
)
def test_gpu_causal_invariance(small_config, batch) -> None:
    """bf16 GPU check: future tokens must not change earlier outputs."""
    device = _gpu_device()
    torch.manual_seed(0)
    model = DualStateModel(small_config).to(
        device=device, dtype=torch.bfloat16
    )
    model.rotary_emb.float()
    model.eval()
    ids = batch["input_ids"].to(device)
    mask = batch["attention_mask"].to(device)
    with torch.no_grad():
        base = model(ids, mask, mode="adaptive", max_high_cycles=3)
        for position in (8, 15):
            for token in range(1, 64, 17):
                perturbed = ids.clone()
                perturbed[:, position] = token
                out = model(
                    perturbed, mask, mode="adaptive", max_high_cycles=3
                )
                assert torch.allclose(
                    base.logits[:, :position].float(),
                    out.logits[:, :position].float(),
                    atol=2e-2,
                ), (position, token)
                assert torch.equal(
                    base.high_cycles_per_position[:, :position],
                    out.high_cycles_per_position[:, :position],
                ), (position, token)
                assert torch.equal(
                    base.stop_reason_codes[:, :position],
                    out.stop_reason_codes[:, :position],
                ), (position, token)
