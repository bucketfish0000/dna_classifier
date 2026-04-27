"""Set encoder modules for DNA grouping."""

from __future__ import annotations

import torch
from torch import nn


class PairFeaturePatchBlock(nn.Module):
    """Build pairwise feature patches and aggregate them back to per-item features.

    Input:
    - x: (batch_size, num_sequences, dim)

    Output:
    - x': (batch_size, num_sequences, dim)
    """

    def __init__(self, dim: int, hidden_dim: int | None = None, dropout: float = 0.0) -> None:
        super().__init__()
        hidden_dim = hidden_dim or 2 * dim

        pair_dim = 4 * dim
        self.pair_norm = nn.LayerNorm(pair_dim)
        self.pair_mlp = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )
        self.output_norm = nn.LayerNorm(dim)
        # Start close to identity so the set encoder cannot immediately wash out embeddings.
        self.residual_gate = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                "PairFeaturePatchBlock expects input of shape (batch_size, num_sequences, dim)."
            )

        batch_size, num_sequences, dim = x.shape
        left = x.unsqueeze(2).expand(-1, -1, num_sequences, -1)
        right = x.unsqueeze(1).expand(-1, num_sequences, -1, -1)

        pair_features = torch.cat(
            [left, right, torch.abs(left - right), left * right],
            dim=-1,
        )
        pair_features = self.pair_norm(pair_features)
        pair_updates = self.pair_mlp(pair_features)

        off_diagonal_mask = ~torch.eye(
            num_sequences,
            dtype=torch.bool,
            device=x.device,
        )
        off_diagonal_mask = off_diagonal_mask.view(1, num_sequences, num_sequences, 1)

        masked_updates = pair_updates * off_diagonal_mask
        num_neighbors = max(num_sequences - 1, 1)
        aggregated_updates = masked_updates.sum(dim=2) / num_neighbors

        x = x + self.residual_gate * aggregated_updates
        return self.output_norm(x)


class SetEncoder(nn.Module):
    """A set encoder based on pairwise feature patches.

    Instead of full self-attention, this encoder builds an explicit NxNxF pairwise tensor
    and aggregates it back into per-item updates. This preserves the (B, N, D) interface
    expected by the rest of the model while using the feature-patch idea discussed for
    relation modeling.
    """

    def __init__(
        self,
        dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        del num_heads  # Kept for interface compatibility with the old SetEncoder API.
        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")

        self.blocks = nn.ModuleList(
            [PairFeaturePatchBlock(dim=dim, hidden_dim=2 * dim, dropout=dropout) for _ in range(num_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                "SetEncoder expects input of shape (batch_size, num_sequences, dim)."
            )

        for block in self.blocks:
            x = block(x)
        return x
