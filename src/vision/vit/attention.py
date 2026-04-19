"""
Transparent Multi-Head Self-Attention with optional QK-Normalization
=====================================================================
A fully explicit attention implementation that replaces PyTorch's ``nn.MultiheadAttention``
black box.  We need full control over Q/K/V for:

  • Attention weight visualization (already useful today).
  • MAE masking — selectively zero-ing out key/value positions (Phase 1 step 3).
  • Cross-attention for VLM projection (Phase 3).
  • NaViT variable-length packing with custom masks (Phase 1 step 2).

QK-Normalization
----------------
Dividing attention logits only by sqrt(d_k) can produce very large values when
depth increases, causing entropy collapse (all weight concentrates on one token).
QK-norm applies L2-normalization to Q and K *before* the dot product, bounding
the inner products to [-1, 1] before applying the learnable/fixed temperature.
This stabilizes training at depth without needing architectural changes to the
residual stream.

Reference: Beyer et al. 2022 (arXiv:2205.01580); Henry et al. 2020 (arXiv:2002.07028)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MultiheadSelfAttention(nn.Module):
    """
    Multi-head self-attention with an explicit Q/K/V formulation.

    Args:
        emb_dim:    Total embedding dimension (must be divisible by num_heads).
        num_heads:  Number of attention heads.
        qk_norm:    If True, apply L2 normalization to Q and K before the dot
                    product (QK-normalization).  Default: True.
        dropout:    Attention dropout probability (applied to the attention
                    weights *before* the weighted sum).  Default: 0.0.

    Inputs:
        x:    Tensor of shape ``(B, N, D)`` — batch, sequence length, embedding dim.
        mask: Optional boolean mask of shape ``(B, N)`` or ``(B, 1, 1, N)``.
              True values are *ignored* (masked out).

    Returns:
        (output, attn_weights):
            output      — Tensor ``(B, N, D)``
            attn_weights — Tensor ``(B, num_heads, N, N)`` — useful for visualization.
    """

    def __init__(
        self,
        emb_dim: int,
        num_heads: int,
        qk_norm: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert emb_dim % num_heads == 0, \
            f"emb_dim ({emb_dim}) must be divisible by num_heads ({num_heads})"

        self.emb_dim   = emb_dim
        self.num_heads = num_heads
        self.head_dim  = emb_dim // num_heads
        self.qk_norm   = qk_norm
        self.scale     = math.sqrt(self.head_dim)  # fixed temperature

        # Separate Q, K, V projections (more transparent than a combined in_proj)
        self.q_proj = nn.Linear(emb_dim, emb_dim, bias=True)
        self.k_proj = nn.Linear(emb_dim, emb_dim, bias=True)
        self.v_proj = nn.Linear(emb_dim, emb_dim, bias=True)
        self.out_proj = nn.Linear(emb_dim, emb_dim, bias=True)

        self.attn_drop = nn.Dropout(dropout)

    # ──────────────────────────────────────────────────────────────────────────

    def _split_heads(self, t: Tensor) -> Tensor:
        """(B, N, D) → (B, num_heads, N, head_dim)"""
        B, N, _ = t.shape
        return t.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, t: Tensor) -> Tensor:
        """(B, num_heads, N, head_dim) → (B, N, D)"""
        B, H, N, _ = t.shape
        return t.transpose(1, 2).reshape(B, N, self.emb_dim)

    # ──────────────────────────────────────────────────────────────────────────

    def forward(
        self,
        x: Tensor,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        B, N, D = x.shape

        # ── Project Q, K, V ──────────────────────────────────────────────────
        q = self._split_heads(self.q_proj(x))  # (B, H, N, head_dim)
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        # ── QK Normalization (optional) ───────────────────────────────────────
        if self.qk_norm:
            q = F.normalize(q, dim=-1)
            k = F.normalize(k, dim=-1)

        # ── Scaled dot-product attention ──────────────────────────────────────
        # attn_logits: (B, H, N, N)
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        # Apply mask (True = ignore)
        if mask is not None:
            # Expand mask to (B, 1, 1, N) if needed
            if mask.dim() == 2:
                mask = mask[:, None, None, :]   # (B, 1, 1, N)
            attn_logits = attn_logits.masked_fill(mask, float("-inf"))

        attn_weights = attn_logits.softmax(dim=-1)          # (B, H, N, N)
        attn_weights = self.attn_drop(attn_weights)

        # Weighted sum of values
        out = torch.matmul(attn_weights, v)                  # (B, H, N, head_dim)
        out = self._merge_heads(out)                         # (B, N, D)
        out = self.out_proj(out)

        return out, attn_weights
