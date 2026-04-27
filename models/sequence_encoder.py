"""Sequence encoder modules for DNA grouping."""

from __future__ import annotations

import torch
from torch import nn

from utils.positional_encoding import sinusoidal_positional_encoding


class ResidualConvBlock1D(nn.Module):
    """A small residual 1D convolution block with sequence-length preservation."""

    def __init__(self, channels: int, kernel_size: int = 5, dropout: float = 0.1):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class CnnSequenceEncoder(nn.Module):
    """Encode DNA sequences into 128-dimensional embeddings.

    Expected input shape:
    - (batch_size, sequence_length)

    Output shape:
    - (batch_size, embedding_dim)
    """

    def __init__(
        self,
        vocab_size: int = 4,
        hidden_dim: int = 256,
        output_dim: int = 128,
        num_conv_layers: int = 6,
        kernel_size: int = 5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_conv_layers <= 0:
            raise ValueError("num_conv_layers must be positive.")

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.input_projection = nn.Linear(hidden_dim, hidden_dim)
        self.conv_blocks = nn.ModuleList(
            [
                ResidualConvBlock1D(
                    channels=hidden_dim,
                    kernel_size=kernel_size,
                    dropout=dropout,
                )
                for _ in range(num_conv_layers)
            ]
        )
        self.output_projection = nn.Linear(hidden_dim, output_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape (batch_size, sequence_length).")

        x = self.token_embedding(input_ids)
        pos = sinusoidal_positional_encoding(
            length=x.shape[1],
            dim=x.shape[2],
            device=x.device,
            dtype=x.dtype,
        )
        x = x + pos.unsqueeze(0)
        x = self.input_projection(x)

        x = x.transpose(1, 2)
        for block in self.conv_blocks:
            x = block(x)

        x = x.transpose(1, 2)
        x = self.layer_norm(x)
        x = x.mean(dim=1)
        x = self.output_projection(x)
        return x
