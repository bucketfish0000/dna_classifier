"""Loss functions for DNA grouping relation models."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class RelationLoss:
    total_loss: torch.Tensor
    upper_loss: torch.Tensor
    diagonal_loss: torch.Tensor
    num_upper_positive: int
    num_upper_negative: int
    num_diagonal: int


def _validate_shapes(logits: torch.Tensor, targets: torch.Tensor) -> None:
    if logits.ndim != 3 or targets.ndim != 3:
        raise ValueError("logits and targets must both have shape (B, N, N).")
    if logits.shape != targets.shape:
        raise ValueError(
            f"logits and targets must have the same shape, got {logits.shape} and {targets.shape}."
        )
    if logits.shape[-1] != logits.shape[-2]:
        raise ValueError("The last two dimensions must form an NxN matrix.")


def _upper_triangle_mask(size: int, device: torch.device) -> torch.Tensor:
    return torch.triu(
        torch.ones(size, size, dtype=torch.bool, device=device),
        diagonal=1,
    )


def _diagonal_mask(size: int, device: torch.device) -> torch.Tensor:
    return torch.eye(size, dtype=torch.bool, device=device)


def weighted_relation_bce_with_diagonal_regularization(
    logits: torch.Tensor,
    targets: torch.Tensor,
    positive_weight: float = 1.0,
    negative_weight: float = 1.0,
    diagonal_weight: float = 0.0,
    reduction: str = "mean",
) -> RelationLoss:
    """Compute weighted BCE on the upper triangle plus a weak diagonal regularizer.

    Main loss:
    - uses only the strict upper triangle
    - applies different weights to positive and negative targets

    Regularizer:
    - uses the diagonal only
    - always compares against the diagonal entries in ``targets`` (typically all ones)
    - is scaled by ``diagonal_weight``
    """

    _validate_shapes(logits, targets)
    if reduction != "mean":
        raise ValueError("Only reduction='mean' is currently supported.")

    batch_size, num_sequences, _ = logits.shape
    device = logits.device
    targets = targets.to(dtype=logits.dtype)

    upper_mask = _upper_triangle_mask(num_sequences, device=device).unsqueeze(0)
    upper_mask = upper_mask.expand(batch_size, -1, -1)
    diagonal_mask = _diagonal_mask(num_sequences, device=device).unsqueeze(0)
    diagonal_mask = diagonal_mask.expand(batch_size, -1, -1)

    upper_targets = targets[upper_mask]
    upper_logits = logits[upper_mask]
    upper_losses = F.binary_cross_entropy_with_logits(
        upper_logits,
        upper_targets,
        reduction="none",
    )

    upper_weights = torch.where(
        upper_targets > 0.5,
        torch.full_like(upper_targets, fill_value=positive_weight),
        torch.full_like(upper_targets, fill_value=negative_weight),
    )
    weighted_upper_losses = upper_losses * upper_weights

    if weighted_upper_losses.numel() == 0:
        upper_loss = logits.new_tensor(0.0)
    else:
        upper_loss = weighted_upper_losses.sum() / upper_weights.sum().clamp_min(1e-8)

    diagonal_targets = targets[diagonal_mask]
    diagonal_logits = logits[diagonal_mask]
    diagonal_losses = F.binary_cross_entropy_with_logits(
        diagonal_logits,
        diagonal_targets,
        reduction="none",
    )

    if diagonal_losses.numel() == 0 or diagonal_weight == 0.0:
        diagonal_loss = logits.new_tensor(0.0)
    else:
        diagonal_loss = diagonal_weight * diagonal_losses.mean()

    total_loss = upper_loss + diagonal_loss

    num_upper_positive = int((upper_targets > 0.5).sum().item())
    num_upper_negative = int((upper_targets <= 0.5).sum().item())
    num_diagonal = batch_size * num_sequences

    return RelationLoss(
        total_loss=total_loss,
        upper_loss=upper_loss,
        diagonal_loss=diagonal_loss,
        num_upper_positive=num_upper_positive,
        num_upper_negative=num_upper_negative,
        num_diagonal=num_diagonal,
    )


class WeightedRelationBCELoss(nn.Module):
    """Module wrapper around the weighted masked relation BCE loss."""

    def __init__(
        self,
        positive_weight: float = 1.0,
        negative_weight: float = 1.0,
        diagonal_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.positive_weight = positive_weight
        self.negative_weight = negative_weight
        self.diagonal_weight = diagonal_weight

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> RelationLoss:
        return weighted_relation_bce_with_diagonal_regularization(
            logits=logits,
            targets=targets,
            positive_weight=self.positive_weight,
            negative_weight=self.negative_weight,
            diagonal_weight=self.diagonal_weight,
        )
