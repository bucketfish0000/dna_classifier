"""Positional encoding utilities."""

from __future__ import annotations

import math

import torch


def sinusoidal_positional_encoding(
    length: int,
    dim: int,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Return a standard sinusoidal positional encoding of shape (length, dim)."""
    if dim <= 0:
        raise ValueError("dim must be positive.")
    if length <= 0:
        raise ValueError("length must be positive.")

    position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=torch.float32)
        * (-math.log(10000.0) / dim)
    )

    encoding = torch.zeros(length, dim, device=device, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(position * div_term)
    encoding[:, 1::2] = torch.cos(position * div_term[: encoding[:, 1::2].shape[1]])

    if dtype is not None:
        encoding = encoding.to(dtype=dtype)
    return encoding
