"""End-to-end relation model for DNA grouping."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .relation_head import CosineRelationHead
from .sequence_encoder import CnnSequenceEncoder
from .set_encoder import SetEncoder


class RelationModel(nn.Module):
    """Compose sequence encoder, set encoder, and relation head.

    Expected input shape:
    - input_ids: (batch_size, num_sequences, padded_sequence_length)

    Output shape:
    - logits: (batch_size, num_sequences, num_sequences)
    """

    def __init__(
        self,
        sequence_encoder: nn.Module | None = None,
        set_encoder: nn.Module | None = None,
        relation_head: nn.Module | None = None,
        embedding_dim: int = 128,
        use_set_encoder: bool = False,
        normalize_sequence_embeddings: bool = True,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.use_set_encoder = use_set_encoder
        self.normalize_sequence_embeddings = normalize_sequence_embeddings
        self.sequence_encoder = (
            sequence_encoder
            if sequence_encoder is not None
            else CnnSequenceEncoder(
                hidden_dim=embedding_dim,
                output_dim=embedding_dim,
                num_conv_layers=3,
                dropout=0.1,
            )
        )
        if self.use_set_encoder:
            self.set_encoder = (
                set_encoder
                if set_encoder is not None
                else SetEncoder(
                    dim=embedding_dim,
                    num_heads=4,
                    num_layers=4,
                    dropout=0.1,
                )
            )
        else:
            self.set_encoder = nn.Identity()
        self.relation_head = (
            relation_head
            if relation_head is not None
            else CosineRelationHead(input_dim=embedding_dim)
        )

    def forward(
        self, input_ids: torch.Tensor, return_pair_features: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if input_ids.ndim != 3:
            raise ValueError(
                "RelationModel expects input_ids of shape "
                "(batch_size, num_sequences, sequence_length)."
            )

        batch_size, num_sequences, sequence_length = input_ids.shape
        flat_input_ids = input_ids.reshape(batch_size * num_sequences, sequence_length)

        sequence_embeddings = self.sequence_encoder(flat_input_ids)
        if sequence_embeddings.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"Expected sequence encoder output dim {self.embedding_dim}, "
                f"got {sequence_embeddings.shape[-1]}."
            )
        if self.normalize_sequence_embeddings:
            sequence_embeddings = F.normalize(sequence_embeddings, p=2, dim=-1)

        set_input = sequence_embeddings.reshape(batch_size, num_sequences, self.embedding_dim)
        set_embeddings = self.set_encoder(set_input)
        return self.relation_head(
            set_embeddings, return_pair_features=return_pair_features
        )

    def forward_debug(self, input_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        if input_ids.ndim != 3:
            raise ValueError(
                "RelationModel expects input_ids of shape "
                "(batch_size, num_sequences, sequence_length)."
            )

        batch_size, num_sequences, sequence_length = input_ids.shape
        flat_input_ids = input_ids.reshape(batch_size * num_sequences, sequence_length)

        sequence_embeddings = self.sequence_encoder(flat_input_ids)
        if self.normalize_sequence_embeddings:
            sequence_embeddings = F.normalize(sequence_embeddings, p=2, dim=-1)
        set_input = sequence_embeddings.reshape(batch_size, num_sequences, self.embedding_dim)
        set_embeddings = self.set_encoder(set_input)
        logits, pair_features = self.relation_head(
            set_embeddings, return_pair_features=True
        )

        return {
            "sequence_embeddings": set_input,
            "set_embeddings": set_embeddings,
            "pair_features": pair_features,
            "logits": logits,
        }
