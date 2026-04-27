"""Relation head modules for DNA grouping."""

from __future__ import annotations

import torch
from torch import nn


class BilinearPairwiseFeatureBuilder(nn.Module):
    """Build pairwise features with a low-dimensional bilinear kernel bank.

    Input:
    - x: (batch_size, num_sequences, input_dim)

    Output:
    - pair_features: (batch_size, num_sequences, num_sequences, num_kernels)
    """

    def __init__(
        self,
        input_dim: int = 128,
        pair_dim: int = 32,
        num_kernels: int = 32,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.pair_dim = pair_dim
        self.num_kernels = num_kernels

        self.down_projection = nn.Linear(input_dim, pair_dim)
        self.kernel_bank = nn.Parameter(torch.empty(num_kernels, pair_dim, pair_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.down_projection.weight)
        if self.down_projection.bias is not None:
            nn.init.zeros_(self.down_projection.bias)
        nn.init.xavier_uniform_(self.kernel_bank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                "BilinearPairwiseFeatureBuilder expects input of shape "
                "(batch_size, num_sequences, input_dim)."
            )
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected last dimension {self.input_dim}, got {x.shape[-1]}."
            )

        projected = self.down_projection(x)
        pair_features = torch.einsum(
            "bid,fde,bje->bijf", projected, self.kernel_bank, projected
        )
        return pair_features


class PairwiseConvScoringHead(nn.Module):
    """Map pairwise features to NxN logits with a 1x1 Conv2d head."""

    def __init__(self, num_kernels: int = 32) -> None:
        super().__init__()
        self.scorer = nn.Conv2d(
            in_channels=num_kernels,
            out_channels=1,
            kernel_size=1,
            bias=True,
        )

    def forward(self, pair_features: torch.Tensor) -> torch.Tensor:
        if pair_features.ndim != 4:
            raise ValueError(
                "PairwiseConvScoringHead expects input of shape "
                "(batch_size, num_sequences, num_sequences, num_kernels)."
            )

        x = pair_features.permute(0, 3, 1, 2)
        logits = self.scorer(x).squeeze(1)
        return logits


class BilinearRelationHead(nn.Module):
    """Full relation head: pairwise bilinear features + 1x1 Conv2d scorer."""

    def __init__(
        self,
        input_dim: int = 128,
        pair_dim: int = 32,
        num_kernels: int = 32,
    ) -> None:
        super().__init__()
        self.feature_builder = BilinearPairwiseFeatureBuilder(
            input_dim=input_dim,
            pair_dim=pair_dim,
            num_kernels=num_kernels,
        )
        self.scoring_head = PairwiseConvScoringHead(num_kernels=num_kernels)

    def forward(
        self, x: torch.Tensor, return_pair_features: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        pair_features = self.feature_builder(x)
        logits = self.scoring_head(pair_features)
        if return_pair_features:
            return logits, pair_features
        return logits


class PairwiseMlpRelationHead(nn.Module):
    """Relation head based on explicit pair features and a 5-layer MLP."""

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        pair_feature_dim = 4 * input_dim
        self.input_dim = input_dim
        self.mlp = nn.Sequential(
            nn.Linear(pair_feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, x: torch.Tensor, return_pair_features: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(
                "PairwiseMlpRelationHead expects input of shape "
                "(batch_size, num_sequences, input_dim)."
            )
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected last dimension {self.input_dim}, got {x.shape[-1]}."
            )

        left = x.unsqueeze(2)
        right = x.unsqueeze(1)
        pair_features = torch.cat(
            [left.expand(-1, -1, x.shape[1], -1), right.expand(-1, x.shape[1], -1, -1),
             torch.abs(left - right), left * right],
            dim=-1,
        )
        logits = self.mlp(pair_features).squeeze(-1)
        if return_pair_features:
            return logits, pair_features
        return logits


class CosineRelationHead(nn.Module):
    """Relation head based on cosine similarity with a learned affine map."""

    def __init__(self, input_dim: int = 128) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(
        self, x: torch.Tensor, return_pair_features: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(
                "CosineRelationHead expects input of shape "
                "(batch_size, num_sequences, input_dim)."
            )
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected last dimension {self.input_dim}, got {x.shape[-1]}."
            )

        normalized = nn.functional.normalize(x, p=2, dim=-1)
        cosine_similarity = normalized @ normalized.transpose(1, 2)
        logits = self.scale * cosine_similarity + self.bias
        if return_pair_features:
            return logits, cosine_similarity.unsqueeze(-1)
        return logits
