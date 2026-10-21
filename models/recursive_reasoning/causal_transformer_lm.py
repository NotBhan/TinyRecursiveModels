"""Conventional decoder-only causal transformer baseline for the TRM-LM pilot.

This is a from-scratch, randomly initialized causal transformer.  It reuses the
same block implementation, RoPE, RMSNorm, SwiGLU, tied embedding, and explicit
causal/padding mask as ``trm_lm`` so that the comparison isolates the recursive
shared-block computation rather than unrelated architectural details.

Differences from ``trm_lm``:
- the ``num_layers`` blocks are distinct layers, not a shared stack applied
  repeatedly;
- there is no input injection and no recursive refinement;
- ``n_refine`` is fixed at 1.

The parameter count is identical to ``TRM_LM`` with the same configuration.
"""

from __future__ import annotations

from typing import Optional

import torch

from models.layers import rms_norm
from models.recursive_reasoning.trm_lm import TRM_LM, TRM_LMConfig


class CausalTransformerLM(TRM_LM):
    """Standard single-pass decoder-only causal transformer."""

    def __init__(self, config: TRM_LMConfig) -> None:
        if config.n_refine != 1:
            raise ValueError(
                "CausalTransformerLM is single-pass; set n_refine=1"
            )
        super().__init__(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        n_refine: Optional[int] = None,
    ) -> torch.Tensor:
        """Return ``[batch, seq_len, vocab_size]`` logits in one pass."""
        if n_refine not in (None, 1):
            raise ValueError(
                "CausalTransformerLM is single-pass; n_refine must be 1"
            )
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

        hidden_states = self.embed_tokens(input_ids)
        cos_sin = self.rotary_emb()
        for block in self.blocks:
            hidden_states = block(hidden_states, cos_sin, attention_mask)
        hidden_states = rms_norm(
            hidden_states, variance_epsilon=self.config.rms_norm_eps
        )
        return self.lm_head(hidden_states)
