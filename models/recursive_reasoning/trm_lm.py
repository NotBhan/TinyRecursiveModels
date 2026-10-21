"""TRM-LM: a causal language-model research variant of Tiny Recursive Models.

This module is intentionally separate from the reference Samsung TRM
implementation in ``models/recursive_reasoning/trm.py``. It does not modify or
replace that implementation.

Design mapping from the reference TRM
-------------------------------------
The reference TRM maintains two latent states (``z_H`` and ``z_L``) and applies
shared reasoning layers for ``H_cycles`` / ``L_cycles``. That two-state
hierarchy was designed for structured puzzle canvases and is collapsed here to a
single latent ``z`` for language modeling. The defining TRM property retained is
*repeated refinement through shared parameters with input injection*: the same
block stack is applied ``n_refine`` times, and the token embeddings are injected
at every pass (``z <- Blocks(z + x)``). Every pass is differentiable. There is
no ACT halting, no q-head, and no puzzle embedding in this variant.

The attention path is causal *and* padding-aware. It uses an explicit boolean
mask with ``scaled_dot_product_attention`` so that neither future tokens nor
padding tokens can influence a position's prediction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from models.common import trunc_normal_init_
from models.layers import (
    CastedLinear,
    RotaryEmbedding,
    SwiGLU,
    apply_rotary_pos_emb,
    rms_norm,
)

IGNORE_INDEX = -100


@dataclass
class TRM_LMConfig:
    """Configuration for the causal TRM-LM variant.

    ``puzzle_emb_ndim`` and ``puzzle_emb_len`` are accepted for explicit
    compatibility with the reference configuration, but the LM path requires
    both to be zero and has no puzzle-embedding parameters.
    """

    vocab_size: int
    seq_len: int = 256
    hidden_size: int = 512
    num_layers: int = 2
    num_heads: int = 8
    expansion: float = 4.0
    n_refine: int = 1
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5
    tie_embeddings: bool = True
    pad_token_id: int = 0
    initializer_range: float = 0.02
    puzzle_emb_ndim: int = 0
    puzzle_emb_len: int = 0

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.seq_len <= 0:
            raise ValueError("seq_len must be positive")
        if self.hidden_size <= 0 or self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.n_refine <= 0:
            raise ValueError("n_refine must be >= 1")
        if self.puzzle_emb_ndim != 0 or self.puzzle_emb_len != 0:
            raise ValueError(
                "TRM-LM has no puzzle embeddings; set puzzle_emb_ndim=0 and "
                "puzzle_emb_len=0"
            )
        if not (0 <= self.pad_token_id < self.vocab_size):
            raise ValueError("pad_token_id must be inside the vocabulary")

    def config_dict(self) -> dict:
        return asdict(self)


class TRM_LMCausalSelfAttention(nn.Module):
    """Multi-head self-attention with explicit causal + padding masking."""

    def __init__(self, config: TRM_LMConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.qkv_proj = CastedLinear(
            config.hidden_size, 3 * config.hidden_size, bias=False
        )
        self.o_proj = CastedLinear(
            config.hidden_size, config.hidden_size, bias=False
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos_sin: Optional[Tuple[torch.Tensor, torch.Tensor]],
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split(self.hidden_size, dim=-1)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)

        if cos_sin is not None:
            cos, sin = cos_sin
            q, k = apply_rotary_pos_emb(q, k, cos[:seq_len], sin[:seq_len])

        # [B, H, L, D] for torch's SDPA.
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Boolean SDPA mask: True means the key participates.  A key at
        # position j is visible to query i iff j <= i and the key is not a
        # padding position.  Padding queries are deliberately allowed to attend
        # to real keys; their outputs are ignored by the loss.
        causal = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=hidden_states.device)
        )
        key_is_real = attention_mask.to(dtype=torch.bool, device=hidden_states.device)
        mask = causal.unsqueeze(0) & key_is_real.unsqueeze(1)  # [B, L, L]
        mask = mask.unsqueeze(1)  # [B, 1, L, L]

        attn_output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, is_causal=False, dropout_p=0.0
        )
        attn_output = attn_output.transpose(1, 2).reshape(
            batch_size, seq_len, self.hidden_size
        )
        return self.o_proj(attn_output)


class TRM_LMBlock(nn.Module):
    """A post-norm transformer block matching the reference TRM block order."""

    def __init__(self, config: TRM_LMConfig) -> None:
        super().__init__()
        self.self_attn = TRM_LMCausalSelfAttention(config)
        self.mlp = SwiGLU(hidden_size=config.hidden_size, expansion=config.expansion)
        self.norm_eps = config.rms_norm_eps

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos_sin: Optional[Tuple[torch.Tensor, torch.Tensor]],
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = rms_norm(
            hidden_states
            + self.self_attn(hidden_states, cos_sin, attention_mask),
            variance_epsilon=self.norm_eps,
        )
        hidden_states = rms_norm(
            hidden_states + self.mlp(hidden_states),
            variance_epsilon=self.norm_eps,
        )
        return hidden_states


class TRM_LM(nn.Module):
    """Causal language model with shared recursive refinement passes."""

    def __init__(self, config: TRM_LMConfig) -> None:
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        trunc_normal_init_(self.embed_tokens.weight, std=config.initializer_range)

        self.blocks = nn.ModuleList(
            [TRM_LMBlock(config) for _ in range(config.num_layers)]
        )

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_embeddings:
            # Standard tied input/output embedding for a small from-scratch LM.
            self.lm_head.weight = self.embed_tokens.weight
        else:
            trunc_normal_init_(self.lm_head.weight, std=config.initializer_range)

        self.rotary_emb = RotaryEmbedding(
            dim=config.hidden_size // config.num_heads,
            max_position_embeddings=config.seq_len,
            base=config.rope_theta,
        )

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def config_dict(self) -> dict:
        return asdict(self.config)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        n_refine: Optional[int] = None,
    ) -> torch.Tensor:
        """Return vocabulary logits of shape ``[batch, seq_len, vocab_size]``.

        ``attention_mask`` is ``[batch, seq_len]`` with 1 for real tokens and 0
        for padding.  ``n_refine`` overrides the configured number of shared
        refinement passes for this call (useful for the pass-count tests).
        """
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

        passes = self.config.n_refine if n_refine is None else int(n_refine)
        if passes < 1:
            raise ValueError("n_refine must be >= 1")

        input_embeddings = self.embed_tokens(input_ids)
        latent = input_embeddings
        cos_sin = self.rotary_emb()

        # All refinement passes remain in the autograd graph.  There is no
        # torch.no_grad() and no detached carry here.
        for _ in range(passes):
            hidden_states = latent + input_embeddings
            for block in self.blocks:
                hidden_states = block(hidden_states, cos_sin, attention_mask)
            latent = hidden_states

        latent = rms_norm(latent, variance_epsilon=self.config.rms_norm_eps)
        return self.lm_head(latent)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 8,
        eos_token_id: Optional[int] = 1,
    ) -> torch.Tensor:
        """Greedy autoregressive generation from a prompt."""
        self.eval()
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if input_ids.dim() != 2:
            raise ValueError("input_ids must have shape [batch, seq_len]")

        generated = input_ids
        for _ in range(max_new_tokens):
            if generated.shape[1] >= self.config.seq_len:
                break
            attention_mask = torch.ones_like(generated, dtype=torch.bool)
            logits = self.forward(generated, attention_mask=attention_mask)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if eos_token_id is not None and bool((next_token == eos_token_id).all()):
                break
        return generated
