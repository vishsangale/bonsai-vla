"""
2D Sinusoidal Positional Encoding
==================================
Fixed (non-learnable) positional encoding for Vision Transformers.

Each patch at grid position (row, col) gets a deterministic encoding built from
sinusoidal bands over its row index and column index independently.  The two
half-dimensional embeddings are then concatenated.

Why this instead of learned 1D embeddings
------------------------------------------
- No parameters to train → cleaner ablations.
- Resolution-agnostic: interpolating to a new (H', W') grid requires only a
  simple bilinear interpolation of the 2D grid, not re-training.  This is the
  property that makes NaViT and variable-resolution VLAs tractable.
- Empirically matches or beats learned embeddings on standard benchmarks
  (Beyer et al., 2022 — arXiv:2205.01580).

Reference
---------
"Better plain ViT baselines for ImageNet-1k", Beyer et al., 2022
https://arxiv.org/abs/2205.01580
"""

import math
import torch
import torch.nn.functional as F
from torch import Tensor


def sinusoidal_2d_pos_encoding(
    grid_h: int,
    grid_w: int,
    emb_dim: int,
    temperature: float = 10_000.0,
    device: torch.device | None = None,
) -> Tensor:
    """
    Build a fixed 2D sinusoidal positional encoding tensor.

    Args:
        grid_h:      Number of patch rows  (img_size // patch_size).
        grid_w:      Number of patch cols  (img_size // patch_size).
        emb_dim:     Total embedding dimension (must be even).
        temperature: Base frequency for the sinusoidal bands (default 10 000).
        device:      Target device (None → CPU).

    Returns:
        Tensor of shape ``(grid_h * grid_w, emb_dim)``.

    The first ``emb_dim // 2`` dimensions encode the *row* position;
    the last  ``emb_dim // 2`` dimensions encode the *col* position.
    Each half uses alternating sin/cos at geometrically-spaced frequencies.
    """
    assert emb_dim % 2 == 0, f"emb_dim must be even, got {emb_dim}"
    half = emb_dim // 2  # each axis gets half the dims

    # ── frequency bands ──────────────────────────────────────────────────────
    # omega_i = 1 / temperature^(2i / half)   for i in [0, half/2)
    # Result shape: (half // 2,)
    omega = 1.0 / (temperature ** (torch.arange(0, half, 2, dtype=torch.float32, device=device) / half))

    # ── position grids ───────────────────────────────────────────────────────
    rows = torch.arange(grid_h, dtype=torch.float32, device=device)  # (H,)
    cols = torch.arange(grid_w, dtype=torch.float32, device=device)  # (W,)

    # Outer product: each position × each frequency band → sin and cos
    # row_enc shape: (H, half)
    row_angles = torch.outer(rows, omega)            # (H, half//2)
    row_enc    = torch.cat([row_angles.sin(), row_angles.cos()], dim=-1)  # (H, half)

    col_angles = torch.outer(cols, omega)            # (W, half//2)
    col_enc    = torch.cat([col_angles.sin(), col_angles.cos()], dim=-1)  # (W, half)

    # ── broadcast onto (H, W) grid then flatten ──────────────────────────────
    # row_enc[:, None, :] → (H, 1, half) broadcasts with col_enc[None, :, :] → (1, W, half)
    enc = torch.cat(
        [
            row_enc[:, None, :].expand(grid_h, grid_w, half),  # (H, W, half)
            col_enc[None, :, :].expand(grid_h, grid_w, half),  # (H, W, half)
        ],
        dim=-1,
    )  # (H, W, emb_dim)

    return enc.reshape(grid_h * grid_w, emb_dim)  # (N_patches, emb_dim)


def interpolate_pos_encoding(
    enc: Tensor,
    src_grid_h: int,
    src_grid_w: int,
    tgt_grid_h: int,
    tgt_grid_w: int,
) -> Tensor:
    """
    Bilinearly interpolate a flat positional encoding to a new resolution.

    Useful when fine-tuning a model trained at one resolution on images of a
    different resolution (e.g., 32×32 → 224×224).

    Args:
        enc:         Source encoding, shape ``(src_h*src_w, D)``.
        src_grid_h:  Source grid height.
        src_grid_w:  Source grid width.
        tgt_grid_h:  Target grid height.
        tgt_grid_w:  Target grid width.

    Returns:
        Interpolated encoding of shape ``(tgt_h*tgt_w, D)``.
    """
    if src_grid_h == tgt_grid_h and src_grid_w == tgt_grid_w:
        return enc

    D = enc.shape[-1]
    # Reshape to (1, D, H, W) for F.interpolate
    enc_4d = enc.reshape(1, src_grid_h, src_grid_w, D).permute(0, 3, 1, 2)  # (1, D, H, W)
    enc_4d = F.interpolate(enc_4d, size=(tgt_grid_h, tgt_grid_w), mode="bilinear", align_corners=False)
    return enc_4d.permute(0, 2, 3, 1).reshape(tgt_grid_h * tgt_grid_w, D)
