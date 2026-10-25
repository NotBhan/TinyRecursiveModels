"""Dual-state recursive causal language model with adaptive halting.

This is a NEW research module. It does not modify the frozen Samsung TRM
implementation in ``models/recursive_reasoning/trm.py`` or the existing
single-state ``trm_lm`` pilot implementation.

State-transition convention
---------------------------
Inputs are token IDs ``X`` of shape ``[B, T]``; embeddings are ``E`` of shape
``[B, T, d]``.  Causal/padding masking is handled by the shared
``TRM_LMBlock``.  The two latent states are initialized with separate learned
projections:

    z_H = P_H(E)
    z_L = P_L(E)

For every high-level cycle:

    repeat n_low_cycles:
        z_L = Norm(F_L(z_L + z_H + E))
    z_H = Norm(F_H(z_H + z_L + E))

``F_L`` and ``F_H`` are shared across recursive applications.  By default they
are separate stacks; ``share_high_low_blocks=True`` makes ``F_H`` reuse
``F_L`` (closer to the reference TRM's single shared reasoning stack).  The
final logits are produced from ``z_H`` and therefore depend on both states.

Halting is per position.  Each position carries its own active flag and its own
controller score.  Once a position halts, both ``z_H`` and ``z_L`` at that
position are frozen and are not updated by later recursive transitions.  The
controller reads only the local position's states (plus depth/budget scalars),
so position ``i`` depends only on tokens at positions ``<= i``.

This is a TRM-inspired causal language-model adaptation, not a claim of
source-level equivalence with the original puzzle model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from models.common import trunc_normal_init_
from models.layers import RotaryEmbedding, rms_norm
from models.recursive_reasoning.trm_lm import TRM_LMBlock, TRM_LMConfig

IGNORE_INDEX = -100
GRADIENT_STRATEGIES = ("full", "truncated", "last_high_cycle")
INFERENCE_MODES = ("fast", "fixed", "adaptive", "budgeted")

STOP_PADDING = 0
STOP_LEARNED_HALT = 1
STOP_MAX_REACHED = 2
STOP_BUDGET_EXHAUSTED = 3
STOP_MINIMUM_REACHED = 4
STOP_FIXED_DEPTH = 5
STOP_REASON_NAMES = {
    STOP_PADDING: "padding",
    STOP_LEARNED_HALT: "learned_halt",
    STOP_MAX_REACHED: "max_reached",
    STOP_BUDGET_EXHAUSTED: "budget_exhausted",
    STOP_MINIMUM_REACHED: "minimum_reached",
    STOP_FIXED_DEPTH: "fixed_depth",
}


def _summarize_position_reasons(
    codes: torch.Tensor, valid: torch.Tensor
) -> str:
    """Collapse per-position stop codes to one sequence label."""
    selected = codes[valid]
    if selected.numel() == 0:
        return STOP_REASON_NAMES[STOP_PADDING]
    unique = torch.unique(selected).tolist()
    if len(unique) == 1:
        return STOP_REASON_NAMES[int(unique[0])]
    return "mixed"


@dataclass
class DualStateConfig(TRM_LMConfig):
    """Typed configuration for the dual-state adaptive model.

    Inherited fields ``num_layers`` and ``n_refine`` are unused by this model;
    they are retained so the existing ``TRM_LMBlock`` configuration interface is
    compatible.  Recursion is controlled by ``n_low_cycles`` / high-cycle
    fields.
    """

    n_low_layers: int = 1
    n_high_layers: int = 1
    n_low_cycles: int = 1
    n_high_cycles: int = 2
    min_high_cycles: int = 1
    max_high_cycles: int = 4
    controller_hidden_size: int = 64
    controller_dropout: float = 0.0
    halt_threshold: float = 0.5
    controller_detach_features: bool = True
    share_high_low_blocks: bool = False
    gradient_strategy: str = "full"
    max_total_refinement_steps: Optional[int] = None
    # Stability options. "overwrite" is the original behavior and is the
    # default, so all historical checkpoints load and run unchanged.
    update_mode: str = "overwrite"  # overwrite | residual | gated
    stable_update_from_depth: int = 1
    gate_init_bias: float = 0.0
    residual_alpha_init: float = 0.5

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.n_low_layers <= 0 or self.n_high_layers <= 0:
            raise ValueError("n_low_layers and n_high_layers must be positive")
        if self.update_mode not in {"overwrite", "residual", "gated"}:
            raise ValueError("update_mode must be overwrite, residual, or gated")
        if self.stable_update_from_depth < 1:
            raise ValueError("stable_update_from_depth must be >= 1")
        if self.update_mode in {"residual", "gated"}:
            # A stable update that starts only after the maximum supported depth
            # would silently disable the intervention; reject that configuration.
            if self.stable_update_from_depth > self.max_high_cycles:
                raise ValueError(
                    "stable_update_from_depth must be <= max_high_cycles"
                )
        if self.n_low_cycles <= 0:
            raise ValueError("n_low_cycles must be positive")
        if not (1 <= self.min_high_cycles <= self.n_high_cycles <= self.max_high_cycles):
            raise ValueError(
                "require 1 <= min_high_cycles <= n_high_cycles <= max_high_cycles"
            )
        if self.controller_hidden_size <= 0:
            raise ValueError("controller_hidden_size must be positive")
        if not (0.0 <= self.halt_threshold <= 1.0):
            raise ValueError("halt_threshold must be in [0, 1]")
        if self.controller_dropout < 0.0 or self.controller_dropout >= 1.0:
            raise ValueError("controller_dropout must be in [0, 1)")
        if self.gradient_strategy not in GRADIENT_STRATEGIES:
            raise ValueError(
                f"gradient_strategy must be one of {GRADIENT_STRATEGIES}"
            )
        if self.max_total_refinement_steps is not None:
            min_transitions = self.min_high_cycles * (self.n_low_cycles + 1)
            if self.max_total_refinement_steps < min_transitions:
                raise ValueError(
                    "max_total_refinement_steps is smaller than the minimum "
                    f"required transitions ({min_transitions})"
                )

    def config_dict(self) -> dict:
        return asdict(self)


@dataclass
class DualStateOutput:
    logits: torch.Tensor
    z_H: torch.Tensor
    z_L: torch.Tensor
    high_cycles: torch.Tensor              # [B] max high cycles over valid positions
    low_cycles: torch.Tensor               # [B] max low updates over valid positions
    transitions: torch.Tensor              # [B] max transitions over valid positions
    high_cycles_per_position: torch.Tensor  # [B, T]
    low_cycles_per_position: torch.Tensor   # [B, T]
    transitions_per_position: torch.Tensor  # [B, T]
    stop_reason_codes: torch.Tensor         # [B, T] STOP_* codes
    active_counts_per_step: List[int]       # active valid positions per executed cycle
    block_calls: int                        # actual recursive block invocations
    controller_calls: int                   # actual controller invocations
    halt_probs: List[torch.Tensor]          # [B, T] continuation probability
    stopping_reason: List[str]              # one summary string per sequence
    mode: str


@dataclass
class DepthOutput:
    logits_by_depth: Dict[int, torch.Tensor]
    state_by_depth: Dict[int, Tuple[torch.Tensor, torch.Tensor]]
    controller_logits_by_depth: Dict[int, torch.Tensor]
    final_z_H: torch.Tensor
    final_z_L: torch.Tensor
    max_depth: int
    transitions_per_high_cycle: int
    block_calls: int
    controller_calls: int


@dataclass
class GenerationOutput:
    tokens: torch.Tensor
    transitions_per_step: List[int]
    position_transitions_per_step: List[int]
    total_transitions: int
    stopping_reasons: List[List[str]]
    high_cycles_per_step: List[List[int]]
    position_high_cycles_per_step: List[List[List[int]]]
    block_calls_per_step: List[int]
    controller_calls_per_step: List[int]
    prompt_length: int
    new_token_count: int
    mode: str
    transitions_last_forward: int
    token_entropies: List[float] = field(default_factory=list)
    token_margins: List[float] = field(default_factory=list)


class ChannelGate(nn.Module):
    """Per-channel gate between the previous state and the new candidate.

    ``g = sigmoid(w_prev * prev + w_cand * candidate + bias)``. The parameters
    are shared across positions; the gate value is per position and per channel.
    A bias of 0 starts at 0.5 (half preservation, half update). A large negative
    bias starts near preservation; a large positive bias starts near the
    original overwrite behavior. The gate is only used when
    ``update_mode == "gated"`` and the current cycle is at or beyond
    ``stable_update_from_depth``.
    """

    def __init__(self, hidden_size: int, init_bias: float = 0.0) -> None:
        super().__init__()
        self.w_prev = nn.Parameter(torch.zeros(hidden_size))
        self.w_cand = nn.Parameter(torch.zeros(hidden_size))
        self.bias = nn.Parameter(torch.full((hidden_size,), float(init_bias)))

    def forward(self, prev: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(
            self.w_prev * prev + self.w_cand * candidate + self.bias
        )


class DualStateController(nn.Module):
    """Per-position learned continuation controller.

    The controller reads only the local position's ``z_H`` and ``z_L`` plus two
    scalar features (normalized depth and remaining budget).  It performs no
    pooling over sequence positions, so position ``i`` cannot see position
    ``j > i``.  It never receives token labels or deeper latent states.
    ``forward`` returns a logit per position; callers apply sigmoid to obtain
    P(continue).
    """

    def __init__(self, config: DualStateConfig) -> None:
        super().__init__()
        self.config = config
        input_dim = 2 * config.hidden_size + 2
        self.net = nn.Sequential(
            nn.Linear(input_dim, config.controller_hidden_size),
            nn.SiLU(),
            nn.Dropout(config.controller_dropout),
            nn.Linear(config.controller_hidden_size, 1),
        )
        for module in self.net:
            if isinstance(module, nn.Linear):
                trunc_normal_init_(
                    module.weight, std=config.initializer_range
                )
                nn.init.zeros_(module.bias)

    def forward(
        self,
        z_H: torch.Tensor,
        z_L: torch.Tensor,
        attention_mask: torch.Tensor,
        depth: int,
        max_depth: int,
        remaining_budget_fraction: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if depth < 0 or max_depth <= 0 or depth > max_depth:
            raise ValueError("invalid controller depth")
        if z_H.dim() != 3 or z_L.shape != z_H.shape:
            raise ValueError("z_H and z_L must both be [batch, seq_len, hidden]")
        batch_size, seq_len, _ = z_H.shape
        if attention_mask.shape != (batch_size, seq_len):
            raise ValueError("attention_mask must match z_H's first two dimensions")

        if self.config.controller_detach_features:
            z_H = z_H.detach()
            z_L = z_L.detach()

        depth_feature = torch.full(
            (batch_size, seq_len, 1), float(depth) / float(max_depth),
            dtype=z_H.dtype, device=z_H.device,
        )
        if remaining_budget_fraction is None:
            remaining = torch.full(
                (batch_size, seq_len, 1),
                float(max_depth - depth) / float(max_depth),
                dtype=z_H.dtype, device=z_H.device,
            )
        else:
            remaining = remaining_budget_fraction.to(
                dtype=z_H.dtype, device=z_H.device
            )
            if remaining.dim() == 1:
                remaining = remaining.view(batch_size, 1, 1).expand(
                    -1, seq_len, -1
                )
            elif remaining.dim() == 2:
                remaining = remaining.unsqueeze(-1)
            else:
                raise ValueError(
                    "remaining_budget_fraction must be [B] or [B, T]"
                )
        features = torch.cat(
            [z_H, z_L, depth_feature, remaining], dim=-1
        )
        return self.net(features).squeeze(-1)



class DualStateModel(nn.Module):
    """Dual-state recursive causal LM with explicit inference modes."""

    def __init__(self, config: DualStateConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        trunc_normal_init_(self.embed_tokens.weight, std=config.initializer_range)

        # Separate learned initial-state projections; documented extra params.
        self.P_H = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.P_L = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        trunc_normal_init_(
            self.P_H.weight, std=1.0 / (config.hidden_size ** 0.5)
        )
        trunc_normal_init_(
            self.P_L.weight, std=1.0 / (config.hidden_size ** 0.5)
        )

        self.F_L = nn.ModuleList(
            [TRM_LMBlock(config) for _ in range(config.n_low_layers)]
        )
        if config.share_high_low_blocks:
            self.F_H = None
        else:
            self.F_H = nn.ModuleList(
                [TRM_LMBlock(config) for _ in range(config.n_high_layers)]
            )

        self.lm_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False
        )
        if config.tie_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        else:
            trunc_normal_init_(
                self.lm_head.weight, std=config.initializer_range
            )

        self.controller = DualStateController(config)
        if config.update_mode == "gated":
            self.gate_H = ChannelGate(config.hidden_size, config.gate_init_bias)
            self.gate_L = ChannelGate(config.hidden_size, config.gate_init_bias)
            self.alpha_H = None
            self.alpha_L = None
        elif config.update_mode == "residual":
            a0 = min(max(float(config.residual_alpha_init), 1e-4), 1.0 - 1e-4)
            self.alpha_H = nn.Parameter(torch.logit(torch.tensor(a0)))
            self.alpha_L = nn.Parameter(torch.logit(torch.tensor(a0)))
            self.gate_H = None
            self.gate_L = None
        else:
            self.gate_H = None
            self.gate_L = None
            self.alpha_H = None
            self.alpha_L = None
        self.rotary_emb = RotaryEmbedding(
            dim=config.hidden_size // config.num_heads,
            max_position_embeddings=config.seq_len,
            base=config.rope_theta,
        )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @property
    def transitions_per_high_cycle(self) -> int:
        return self.config.n_low_cycles + 1

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def config_dict(self) -> dict:
        return asdict(self.config)

    def parameter_breakdown(self) -> Dict[str, int]:
        breakdown = {
            "embedding": int(self.embed_tokens.weight.numel()),
            "projections": int(
                sum(p.numel() for p in self.P_H.parameters())
                + sum(p.numel() for p in self.P_L.parameters())
            ),
            "F_L": int(sum(p.numel() for p in self.F_L.parameters())),
            "F_H": (
                0
                if self.F_H is None
                else int(sum(p.numel() for p in self.F_H.parameters()))
            ),
            "controller": int(sum(p.numel() for p in self.controller.parameters())),
            "update_stability": int(
                (sum(p.numel() for p in self.gate_H.parameters()) if self.gate_H is not None else 0)
                + (sum(p.numel() for p in self.gate_L.parameters()) if self.gate_L is not None else 0)
                + (self.alpha_H.numel() if self.alpha_H is not None else 0)
                + (self.alpha_L.numel() if self.alpha_L is not None else 0)
            ),
            "lm_head_untied": (
                0
                if self.config.tie_embeddings
                else int(self.lm_head.weight.numel())
            ),
        }
        breakdown["total_dedup"] = self.num_parameters
        return breakdown

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------
    def _validate_inputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must have shape [batch, seq_len]")
        batch_size, seq_len = input_ids.shape
        if seq_len > self.config.seq_len:
            raise ValueError(
                f"sequence length {seq_len} exceeds configured seq_len "
                f"{self.config.seq_len}"
            )
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_len), dtype=torch.bool, device=input_ids.device
            )
        else:
            if attention_mask.shape != (batch_size, seq_len):
                raise ValueError("attention_mask must match input_ids shape")
            attention_mask = attention_mask.to(
                device=input_ids.device, dtype=torch.bool
            )
        return attention_mask

    def _init_states(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        embeddings = self.embed_tokens(input_ids)
        z_H = self.P_H(embeddings)
        z_L = self.P_L(embeddings)
        return embeddings, z_H, z_L

    def _run_low(
        self,
        z_L: torch.Tensor,
        z_H: torch.Tensor,
        embeddings: torch.Tensor,
        cos_sin,
        attention_mask: torch.Tensor,
        active: Optional[torch.Tensor] = None,
        counter: Optional[List[int]] = None,
        use_stable_update: bool = False,
    ) -> torch.Tensor:
        hidden = z_L + z_H + embeddings
        if active is not None:
            active_view = active.unsqueeze(-1)
        for block in self.F_L:
            candidate = block(hidden, cos_sin, attention_mask)
            if active is not None:
                hidden = torch.where(active_view, candidate, hidden)
            else:
                hidden = candidate
            if counter is not None:
                counter[0] += 1
        candidate = rms_norm(hidden, variance_epsilon=self.config.rms_norm_eps)
        if use_stable_update and self.config.update_mode == "residual":
            alpha = torch.sigmoid(self.alpha_L)
            candidate = rms_norm(
                (1.0 - alpha) * z_L + alpha * candidate,
                variance_epsilon=self.config.rms_norm_eps,
            )
        elif use_stable_update and self.config.update_mode == "gated":
            gate = self.gate_L(z_L, candidate)
            candidate = rms_norm(
                gate * candidate + (1.0 - gate) * z_L,
                variance_epsilon=self.config.rms_norm_eps,
            )
        new_z_L = candidate
        if active is not None:
            new_z_L = torch.where(active_view, new_z_L, z_L)
        return new_z_L

    def _run_high(
        self,
        z_H: torch.Tensor,
        z_L: torch.Tensor,
        embeddings: torch.Tensor,
        cos_sin,
        attention_mask: torch.Tensor,
        active: Optional[torch.Tensor] = None,
        counter: Optional[List[int]] = None,
        use_stable_update: bool = False,
    ) -> torch.Tensor:
        hidden = z_H + z_L + embeddings
        if active is not None:
            active_view = active.unsqueeze(-1)
        blocks = self.F_L if self.F_H is None else self.F_H
        for block in blocks:
            candidate = block(hidden, cos_sin, attention_mask)
            if active is not None:
                hidden = torch.where(active_view, candidate, hidden)
            else:
                hidden = candidate
            if counter is not None:
                counter[0] += 1
        candidate = rms_norm(hidden, variance_epsilon=self.config.rms_norm_eps)
        if use_stable_update and self.config.update_mode == "residual":
            alpha = torch.sigmoid(self.alpha_H)
            candidate = rms_norm(
                (1.0 - alpha) * z_H + alpha * candidate,
                variance_epsilon=self.config.rms_norm_eps,
            )
        elif use_stable_update and self.config.update_mode == "gated":
            gate = self.gate_H(z_H, candidate)
            candidate = rms_norm(
                gate * candidate + (1.0 - gate) * z_H,
                variance_epsilon=self.config.rms_norm_eps,
            )
        new_z_H = candidate
        if active is not None:
            new_z_H = torch.where(active_view, new_z_H, z_H)
        return new_z_H

    def _high_cycle(
        self,
        z_H: torch.Tensor,
        z_L: torch.Tensor,
        embeddings: torch.Tensor,
        cos_sin,
        attention_mask: torch.Tensor,
        n_low_cycles: Optional[int] = None,
        active: Optional[torch.Tensor] = None,
        counter: Optional[List[int]] = None,
        use_stable_update: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        low_cycles = (
            self.config.n_low_cycles
            if n_low_cycles is None
            else int(n_low_cycles)
        )
        if low_cycles <= 0:
            raise ValueError("n_low_cycles must be positive")
        for _ in range(low_cycles):
            z_L = self._run_low(
                z_L, z_H, embeddings, cos_sin, attention_mask,
                active=active, counter=counter,
                use_stable_update=use_stable_update,
            )
        z_H = self._run_high(
            z_H, z_L, embeddings, cos_sin, attention_mask,
            active=active, counter=counter,
            use_stable_update=use_stable_update,
        )
        return z_H, z_L

    def _logits(
        self, z_H: torch.Tensor, last_token_only: bool = False
    ) -> torch.Tensor:
        z = rms_norm(z_H, variance_epsilon=self.config.rms_norm_eps)
        if last_token_only:
            z = z[:, -1:, :]
        return self.lm_head(z)

    # ------------------------------------------------------------------
    # Fixed-depth multi-depth forward (training)
    # ------------------------------------------------------------------
    def forward_depths(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        depths: Optional[List[int]] = None,
        max_depth: Optional[int] = None,
        evaluate_controller: bool = True,
        controller_max_depth: Optional[int] = None,
    ) -> DepthOutput:
        """Run a fixed maximum depth and return logits/states at requested depths.

        This is the training path.  ``gradient_strategy`` controls which high
        cycles receive gradients.  It does not perform halting.
        """
        attention_mask = self._validate_inputs(input_ids, attention_mask)
        if max_depth is None:
            max_depth = (
                max(depths)
                if depths is not None
                else self.config.max_high_cycles
            )
        if max_depth < 1 or max_depth > self.config.max_high_cycles:
            raise ValueError(
                f"max_depth must be in [1, {self.config.max_high_cycles}]"
            )
        if depths is None:
            depths = list(range(1, max_depth + 1))
        requested = sorted(set(int(d) for d in depths))
        if not requested or requested[0] < 1 or requested[-1] > max_depth:
            raise ValueError("depths must be non-empty and within max_depth")
        if controller_max_depth is None:
            controller_max_depth = max_depth
        if not (1 <= controller_max_depth <= self.config.max_high_cycles):
            raise ValueError("controller_max_depth is out of range")

        embeddings, z_H, z_L = self._init_states(input_ids, attention_mask)
        cos_sin = self.rotary_emb()
        logits_by_depth: Dict[int, torch.Tensor] = {}
        state_by_depth: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        controller_logits: Dict[int, torch.Tensor] = {}
        block_calls = [0]
        controller_calls = 0

        for depth in range(1, max_depth + 1):
            use_grad = not (
                self.config.gradient_strategy == "last_high_cycle"
                and depth < max_depth
            )
            use_stable = depth >= self.config.stable_update_from_depth
            if use_grad:
                z_H, z_L = self._high_cycle(
                    z_H, z_L, embeddings, cos_sin, attention_mask,
                    counter=block_calls, use_stable_update=use_stable,
                )
            else:
                with torch.no_grad():
                    z_H, z_L = self._high_cycle(
                        z_H, z_L, embeddings, cos_sin, attention_mask,
                        counter=block_calls, use_stable_update=use_stable,
                    )
            logits = self._logits(z_H)
            if depth in requested:
                logits_by_depth[depth] = logits
                state_by_depth[depth] = (z_H, z_L)
            if evaluate_controller and depth < max_depth:
                controller_logits[depth] = self.controller(
                    z_H,
                    z_L,
                    attention_mask,
                    depth=depth,
                    max_depth=controller_max_depth,
                    remaining_budget_fraction=None,
                )
                controller_calls += 1
            if (
                self.config.gradient_strategy == "truncated"
                and depth < max_depth
            ):
                z_H = z_H.detach()
                z_L = z_L.detach()

        return DepthOutput(
            logits_by_depth=logits_by_depth,
            state_by_depth=state_by_depth,
            controller_logits_by_depth=controller_logits,
            final_z_H=z_H,
            final_z_L=z_L,
            max_depth=max_depth,
            transitions_per_high_cycle=self.transitions_per_high_cycle,
            block_calls=block_calls[0],
            controller_calls=controller_calls,
        )

    # ------------------------------------------------------------------
    # Inference modes
    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        mode: str = "fixed",
        n_high_cycles: Optional[int] = None,
        n_low_cycles: Optional[int] = None,
        max_high_cycles: Optional[int] = None,
        max_total_transitions: Optional[int] = None,
        halt_threshold: Optional[float] = None,
        last_token_only: bool = False,
    ) -> DualStateOutput:
        if mode not in INFERENCE_MODES:
            raise ValueError(f"mode must be one of {INFERENCE_MODES}")
        attention_mask = self._validate_inputs(input_ids, attention_mask)
        low_cycles = (
            self.config.n_low_cycles if n_low_cycles is None else int(n_low_cycles)
        )
        if low_cycles <= 0:
            raise ValueError("n_low_cycles must be positive")
        per_high = low_cycles + 1
        batch_size = input_ids.shape[0]
        device = input_ids.device
        threshold = (
            self.config.halt_threshold
            if halt_threshold is None
            else float(halt_threshold)
        )
        if not (0.0 <= threshold <= 1.0):
            raise ValueError("halt_threshold must be in [0, 1]")

        if mode in ("fast", "fixed"):
            if mode == "fast":
                high_cycles = self.config.min_high_cycles
                reason_code = STOP_MINIMUM_REACHED
            else:
                high_cycles = (
                    self.config.n_high_cycles
                    if n_high_cycles is None
                    else int(n_high_cycles)
                )
                if not (
                    self.config.min_high_cycles
                    <= high_cycles
                    <= self.config.max_high_cycles
                ):
                    raise ValueError(
                        "fixed n_high_cycles must be within "
                        "[min_high_cycles, max_high_cycles]"
                    )
                reason_code = STOP_FIXED_DEPTH
            embeddings, z_H, z_L = self._init_states(
                input_ids, attention_mask
            )
            cos_sin = self.rotary_emb()
            counter = [0]
            for cycle in range(1, high_cycles + 1):
                z_H, z_L = self._high_cycle(
                    z_H, z_L, embeddings, cos_sin, attention_mask,
                    n_low_cycles=low_cycles, counter=counter,
                    use_stable_update=(
                        cycle >= self.config.stable_update_from_depth
                    ),
                )
            high_cycles_seq = torch.full(
                (batch_size,), high_cycles, dtype=torch.int64, device=device
            )
            high_cycles_pos = attention_mask.to(torch.int64) * high_cycles
            stop_codes = torch.full(
                input_ids.shape, STOP_PADDING, dtype=torch.int64, device=device
            )
            stop_codes = torch.where(
                attention_mask,
                torch.full_like(stop_codes, reason_code),
                stop_codes,
            )
            return DualStateOutput(
                logits=self._logits(z_H, last_token_only=last_token_only),
                z_H=z_H,
                z_L=z_L,
                high_cycles=high_cycles_seq,
                low_cycles=high_cycles_seq * low_cycles,
                transitions=high_cycles_seq * per_high,
                high_cycles_per_position=high_cycles_pos,
                low_cycles_per_position=high_cycles_pos * low_cycles,
                transitions_per_position=high_cycles_pos * per_high,
                stop_reason_codes=stop_codes,
                active_counts_per_step=[int(attention_mask.sum())] * high_cycles,
                block_calls=counter[0],
                controller_calls=0,
                halt_probs=[],
                stopping_reason=[STOP_REASON_NAMES[reason_code]] * batch_size,
                mode=mode,
            )

        # Adaptive / budgeted: per-position halting with state retention.
        min_high = self.config.min_high_cycles
        max_high = (
            self.config.max_high_cycles
            if max_high_cycles is None
            else int(max_high_cycles)
        )
        if not (min_high <= max_high <= self.config.max_high_cycles):
            raise ValueError(
                "max_high_cycles must be within "
                "[min_high_cycles, config.max_high_cycles]"
            )
        budget = (
            self.config.max_total_refinement_steps
            if max_total_transitions is None
            else int(max_total_transitions)
        )
        if mode == "budgeted" and budget is None:
            raise ValueError("budgeted mode requires max_total_transitions")
        if budget is not None and budget < min_high * per_high:
            raise ValueError(
                "max_total_transitions is smaller than the minimum required "
                f"transitions ({min_high * per_high})"
            )

        embeddings, z_H, z_L = self._init_states(input_ids, attention_mask)
        cos_sin = self.rotary_emb()
        valid = attention_mask
        active = valid.clone()
        high_cycles_pos = torch.zeros_like(input_ids, dtype=torch.int64)
        transitions_pos = torch.zeros_like(input_ids, dtype=torch.int64)
        stop_codes = torch.full(
            input_ids.shape, STOP_PADDING, dtype=torch.int64, device=device
        )
        counter = [0]
        controller_calls = 0
        active_counts: List[int] = []
        halt_probs: List[torch.Tensor] = []

        for cycle in range(1, min_high + 1):
            active_counts.append(int(active.sum()))
            z_H, z_L = self._high_cycle(
                z_H, z_L, embeddings, cos_sin, attention_mask,
                n_low_cycles=low_cycles, active=active, counter=counter,
                use_stable_update=(
                    cycle >= self.config.stable_update_from_depth
                ),
            )
            high_cycles_pos = high_cycles_pos + active.to(torch.int64)
            transitions_pos = (
                transitions_pos + active.to(torch.int64) * per_high
            )

        if min_high >= max_high:
            stop_codes = torch.where(
                active,
                torch.full_like(stop_codes, STOP_MAX_REACHED),
                stop_codes,
            )
            active = torch.zeros_like(active)
        else:
            for depth in range(min_high, max_high):
                if not bool(active.any()):
                    break
                if budget is not None:
                    remaining = (budget - transitions_pos).clamp_min(0).to(
                        z_H.dtype
                    )
                    remaining = remaining / float(budget)
                else:
                    remaining = None
                q_logits = self.controller(
                    z_H, z_L, attention_mask,
                    depth=depth, max_depth=max_high,
                    remaining_budget_fraction=remaining,
                )
                q = torch.sigmoid(q_logits)
                halt_probs.append(q.detach().cpu())
                controller_calls += 1
                want = q >= threshold
                if budget is not None:
                    can_continue = (transitions_pos + per_high) <= budget
                else:
                    can_continue = torch.ones_like(active)
                learned_halt = active & ~want
                budget_halt = active & want & ~can_continue
                stop_codes = torch.where(
                    learned_halt,
                    torch.full_like(stop_codes, STOP_LEARNED_HALT),
                    stop_codes,
                )
                stop_codes = torch.where(
                    budget_halt,
                    torch.full_like(stop_codes, STOP_BUDGET_EXHAUSTED),
                    stop_codes,
                )
                active_next = active & want & can_continue
                if not bool(active_next.any()):
                    active = active_next
                    break
                active_counts.append(int(active_next.sum()))
                z_H, z_L = self._high_cycle(
                    z_H, z_L, embeddings, cos_sin, attention_mask,
                    n_low_cycles=low_cycles, active=active_next,
                    counter=counter,
                    use_stable_update=(
                        (depth + 1) >= self.config.stable_update_from_depth
                    ),
                )
                high_cycles_pos = (
                    high_cycles_pos + active_next.to(torch.int64)
                )
                transitions_pos = (
                    transitions_pos
                    + active_next.to(torch.int64) * per_high
                )
                reached_max = active_next & (high_cycles_pos >= max_high)
                stop_codes = torch.where(
                    reached_max,
                    torch.full_like(stop_codes, STOP_MAX_REACHED),
                    stop_codes,
                )
                active = active_next & ~reached_max

        unlabeled = valid & (stop_codes == STOP_PADDING)
        stop_codes = torch.where(
            unlabeled,
            torch.full_like(stop_codes, STOP_MAX_REACHED),
            stop_codes,
        )
        high_cycles_seq = high_cycles_pos.max(dim=1).values
        reasons = [
            _summarize_position_reasons(stop_codes[i], valid[i])
            for i in range(batch_size)
        ]
        return DualStateOutput(
            logits=self._logits(z_H, last_token_only=last_token_only),
            z_H=z_H,
            z_L=z_L,
            high_cycles=high_cycles_seq,
            low_cycles=high_cycles_seq * low_cycles,
            transitions=high_cycles_seq * per_high,
            high_cycles_per_position=high_cycles_pos,
            low_cycles_per_position=high_cycles_pos * low_cycles,
            transitions_per_position=transitions_pos,
            stop_reason_codes=stop_codes,
            active_counts_per_step=active_counts,
            block_calls=counter[0],
            controller_calls=controller_calls,
            halt_probs=halt_probs,
            stopping_reason=reasons,
            mode=mode,
        )

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate_with_diagnostics(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 8,
        eos_token_id: Optional[int] = 1,
        mode: str = "fixed",
        compute_confidence: bool = True,
        last_token_only: bool = True,
        **forward_kwargs,
    ) -> GenerationOutput:
        self.eval()
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if input_ids.dim() != 2:
            raise ValueError("input_ids must have shape [batch, seq_len]")
        prompt_length = input_ids.shape[1]
        generated = input_ids
        transitions_per_step: List[int] = []
        position_transitions_per_step: List[int] = []
        high_cycles_per_step: List[List[int]] = []
        position_high_cycles_per_step: List[List[List[int]]] = []
        block_calls_per_step: List[int] = []
        controller_calls_per_step: List[int] = []
        stopping_reasons: List[List[str]] = []
        transitions_last = 0
        token_entropies: List[float] = []
        token_margins: List[float] = []
        for _ in range(max_new_tokens):
            if generated.shape[1] >= self.config.seq_len:
                break
            attention_mask = torch.ones_like(generated, dtype=torch.bool)
            output = self.forward(
                generated,
                attention_mask,
                mode=mode,
                last_token_only=last_token_only,
                **forward_kwargs,
            )
            last_logits = output.logits[:, -1, :]
            next_token = last_logits.argmax(dim=-1, keepdim=True)
            if compute_confidence:
                probs = torch.softmax(last_logits.float(), dim=-1)
                entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(-1)
                top2 = probs.topk(2, dim=-1).values
                margin = top2[..., 0] - top2[..., 1]
                token_entropies.append(float(entropy.mean().item()))
                token_margins.append(float(margin.mean().item()))
            generated = torch.cat([generated, next_token], dim=1)
            transitions_last = int(output.transitions.sum().item())
            transitions_per_step.append(transitions_last)
            position_transitions_per_step.append(
                int(output.transitions_per_position.sum().item())
            )
            high_cycles_per_step.append(
                output.high_cycles.detach().cpu().tolist()
            )
            position_high_cycles_per_step.append(
                output.high_cycles_per_position.detach().cpu().tolist()
            )
            block_calls_per_step.append(int(output.block_calls))
            controller_calls_per_step.append(int(output.controller_calls))
            stopping_reasons.append(list(output.stopping_reason))
            if (
                eos_token_id is not None
                and bool((next_token == eos_token_id).all())
            ):
                break
        return GenerationOutput(
            tokens=generated,
            transitions_per_step=transitions_per_step,
            position_transitions_per_step=position_transitions_per_step,
            total_transitions=int(sum(transitions_per_step)),
            stopping_reasons=stopping_reasons,
            high_cycles_per_step=high_cycles_per_step,
            position_high_cycles_per_step=position_high_cycles_per_step,
            block_calls_per_step=block_calls_per_step,
            controller_calls_per_step=controller_calls_per_step,
            prompt_length=prompt_length,
            new_token_count=generated.shape[1] - prompt_length,
            mode=mode,
            transitions_last_forward=transitions_last,
            token_entropies=token_entropies,
            token_margins=token_margins,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 8,
        eos_token_id: Optional[int] = 1,
        mode: str = "fixed",
        **forward_kwargs,
    ) -> GenerationOutput:
        return self.generate_with_diagnostics(
            input_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
            mode=mode,
            **forward_kwargs,
        )


def per_sequence_cross_entropy(
    logits: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    """Token-averaged CE per sequence, ignoring labels == -100.

    Returns a ``[B]`` tensor.  Retained for aggregate reporting; the halting
    controller is supervised per position by :func:`per_position_cross_entropy`.
    The controller itself never sees labels; targets are training-time
    supervision only.
    """
    if logits.dim() != 3:
        raise ValueError("logits must be [B, T, V]")
    if labels.shape != logits.shape[:2]:
        raise ValueError("labels must match logits' first two dimensions")
    batch_size, seq_len, vocab = logits.shape
    losses = F.cross_entropy(
        logits.float().reshape(-1, vocab),
        labels.reshape(-1),
        ignore_index=IGNORE_INDEX,
        reduction="none",
    ).view(batch_size, seq_len)
    valid = labels != IGNORE_INDEX
    counts = valid.sum(dim=1).clamp_min(1)
    return (losses * valid).sum(dim=1) / counts


def per_position_cross_entropy(
    logits: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    """Per-position next-token CE, ignoring labels == -100.

    Returns a ``[B, T]`` tensor with zero at ignored positions.  Position ``i``
    depends only on tokens ``<= i`` because the LM path is causal.  Used to
    build per-position marginal-utility targets for the halting controller.
    """
    if logits.dim() != 3:
        raise ValueError("logits must be [B, T, V]")
    if labels.shape != logits.shape[:2]:
        raise ValueError("labels must match logits' first two dimensions")
    batch_size, seq_len, vocab = logits.shape
    losses = F.cross_entropy(
        logits.float().reshape(-1, vocab),
        labels.reshape(-1),
        ignore_index=IGNORE_INDEX,
        reduction="none",
    ).view(batch_size, seq_len)
    valid = labels != IGNORE_INDEX
    return losses * valid

