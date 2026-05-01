#!/usr/bin/env python3
"""Minimal training script to test the DNA grouping workflow."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parent
PROJECT_PARENT = PROJECT_ROOT.parent
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

from dna_grouper.models import RelationModel
from dna_grouper.models.relation_head import (
    BilinearRelationHead,
    CosineRelationHead,
    PairwiseMlpRelationHead,
)
from dna_grouper.models.sequence_encoder import CnnSequenceEncoder
from dna_grouper.models.set_encoder import SetEncoder
from dna_grouper.training.training import (
    build_loader_bundle,
    build_loss_from_loader,
    evaluate_one_epoch,
    format_epoch_stats,
    test_model,
    train_one_epoch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a minimal end-to-end training workflow on DNA grouping data."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/home/grads/tml6120/workspace/dna_grouper/data/debug"),
        help="Root directory whose subdirectories are dataset folders to mix.",
    )
    parser.add_argument(
        "--episode-size",
        type=int,
        default=6,
        help="Number of sequences per episode (fixed N for this run).",
    )
    parser.add_argument(
        "--train-episodes-per-epoch",
        type=int,
        default=64,
        help="Number of train episodes to sample per epoch.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=24,
        help="Number of validation/test episodes to sample.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Number of episodes per batch.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Fallback number of epochs when stage-specific epochs are not set.",
    )
    parser.add_argument(
        "--encoder-only-epochs",
        type=int,
        default=None,
        help="Number of epochs for stage 1: freeze relation head and train only the sequence encoder.",
    )
    parser.add_argument(
        "--set-only-epochs",
        type=int,
        default=None,
        help="Number of epochs for stage 2: freeze sequence encoder and relation head, and train only the set encoder.",
    )
    parser.add_argument(
        "--head-only-epochs",
        type=int,
        default=None,
        help="Number of epochs for stage 3: freeze sequence encoder and set encoder, and train only the relation head.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Optimizer learning rate.",
    )
    parser.add_argument(
        "--positive-weight-scale",
        type=float,
        default=2.0,
        help="Multiply the estimated positive class weight by this factor.",
    )
    parser.add_argument(
        "--diagonal-weight",
        type=float,
        default=0.0,
        help="Diagonal regularization weight in the relation loss.",
    )
    parser.add_argument(
        "--families-per-episode-min",
        type=int,
        default=2,
        help="Minimum number of families in one episode.",
    )
    parser.add_argument(
        "--families-per-episode-max",
        type=int,
        default=None,
        help="Maximum number of families in one episode.",
    )
    parser.add_argument(
        "--length-sampling",
        choices=["uniform", "proportional"],
        default="uniform",
        help="How to sample across available sequence lengths.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed.",
    )
    parser.add_argument(
        "--no-set-encoder",
        action="store_true",
        help="If set, bypass the set encoder with identity.",
    )
    parser.add_argument(
        "--disable-phase-training",
        action="store_true",
        help="If set, disable staged freezing and train all enabled modules end-to-end.",
    )
    parser.add_argument(
        "--alternate-epochs",
        action="store_true",
        help="If set, alternate phase-only epochs instead of running each phase in one contiguous block.",
    )
    parser.add_argument(
        "--set-phase-last",
        action="store_true",
        help="If set, move set-only training after encoder/head phases instead of before head-only.",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=128,
        help="Shared embedding dimension used between encoder, set encoder, and relation head.",
    )
    parser.add_argument(
        "--normalize-sequence-embeddings",
        action="store_true",
        default=True,
        help="Apply L2 normalization to sequence embeddings before the set encoder/relation head.",
    )
    parser.add_argument(
        "--no-normalize-sequence-embeddings",
        action="store_false",
        dest="normalize_sequence_embeddings",
        help="Disable L2 normalization on sequence embeddings.",
    )
    parser.add_argument(
        "--sequence-vocab-size",
        type=int,
        default=5,
        help="Vocabulary size for the DNA token embedding, including PAD.",
    )
    parser.add_argument(
        "--sequence-hidden-dim",
        type=int,
        default=128,
        help="Internal hidden dimension used inside the CNN sequence encoder.",
    )
    parser.add_argument(
        "--sequence-num-conv-layers",
        type=int,
        default=3,
        help="Number of residual 1D convolution blocks in the sequence encoder.",
    )
    parser.add_argument(
        "--sequence-kernel-size",
        type=int,
        default=5,
        help="Kernel size for 1D convolution blocks in the sequence encoder.",
    )
    parser.add_argument(
        "--sequence-dropout",
        type=float,
        default=0.0,
        help="Dropout used inside the sequence encoder convolution blocks.",
    )
    parser.add_argument(
        "--set-num-heads",
        type=int,
        default=4,
        help="Set encoder num_heads argument. Kept for compatibility even if the active implementation ignores it.",
    )
    parser.add_argument(
        "--set-num-layers",
        type=int,
        default=4,
        help="Number of layers/blocks in the set encoder.",
    )
    parser.add_argument(
        "--set-dropout",
        type=float,
        default=0.0,
        help="Dropout used inside the set encoder.",
    )
    parser.add_argument(
        "--relation-head-type",
        choices=["cosine", "mlp", "bilinear"],
        default="cosine",
        help="Which relation head implementation to use.",
    )
    parser.add_argument(
        "--relation-hidden-dim",
        type=int,
        default=256,
        help="Hidden dimension for the MLP relation head.",
    )
    parser.add_argument(
        "--relation-pair-dim",
        type=int,
        default=32,
        help="Low-dimensional pair space for the bilinear relation head.",
    )
    parser.add_argument(
        "--relation-num-kernels",
        type=int,
        default=32,
        help="Number of kernels/channels for the bilinear relation head.",
    )
    return parser.parse_args()


def discover_data_dirs(data_root: Path) -> list[str]:
    split_files = ["train.jsonl", "val.jsonl", "test.jsonl"]
    if data_root.is_dir() and all((data_root / name).exists() for name in split_files):
        return [str(data_root)]

    data_dirs = sorted(str(path) for path in data_root.iterdir() if path.is_dir())
    if not data_dirs:
        raise ValueError(f"No dataset directories found under {data_root}")
    return data_dirs


def build_relation_head_from_args(args: argparse.Namespace) -> nn.Module:
    if args.relation_head_type == "cosine":
        return CosineRelationHead(input_dim=args.embedding_dim)
    if args.relation_head_type == "mlp":
        return PairwiseMlpRelationHead(
            input_dim=args.embedding_dim,
            hidden_dim=args.relation_hidden_dim,
        )
    if args.relation_head_type == "bilinear":
        return BilinearRelationHead(
            input_dim=args.embedding_dim,
            pair_dim=args.relation_pair_dim,
            num_kernels=args.relation_num_kernels,
        )
    raise ValueError(f"Unsupported relation head type: {args.relation_head_type}")


def build_model_from_args(args: argparse.Namespace) -> RelationModel:
    sequence_encoder = CnnSequenceEncoder(
        vocab_size=args.sequence_vocab_size,
        hidden_dim=args.sequence_hidden_dim,
        output_dim=args.embedding_dim,
        num_conv_layers=args.sequence_num_conv_layers,
        kernel_size=args.sequence_kernel_size,
        dropout=args.sequence_dropout,
    )

    set_encoder = SetEncoder(
        dim=args.embedding_dim,
        num_heads=args.set_num_heads,
        num_layers=args.set_num_layers,
        dropout=args.set_dropout,
    )

    relation_head = build_relation_head_from_args(args)

    return RelationModel(
        sequence_encoder=sequence_encoder,
        set_encoder=set_encoder,
        relation_head=relation_head,
        embedding_dim=args.embedding_dim,
        use_set_encoder=not args.no_set_encoder,
        normalize_sequence_embeddings=args.normalize_sequence_embeddings,
    )


def set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = requires_grad


def count_trainable_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def configure_training_phase(model: RelationModel, phase_name: str) -> None:
    if phase_name == "encoder_only":
        set_requires_grad(model.sequence_encoder, True)
        set_requires_grad(model.relation_head, False)
        set_requires_grad(model.set_encoder, False)
        return

    if phase_name == "set_only":
        set_requires_grad(model.sequence_encoder, False)
        set_requires_grad(model.relation_head, False)
        set_requires_grad(model.set_encoder, True)
        return

    if phase_name == "head_only":
        set_requires_grad(model.sequence_encoder, False)
        set_requires_grad(model.relation_head, True)
        set_requires_grad(model.set_encoder, False)
        return

    raise ValueError(f"Unsupported training phase: {phase_name}")


def enable_all_relevant_parameters(model: RelationModel) -> None:
    set_requires_grad(model.sequence_encoder, True)
    set_requires_grad(model.relation_head, True)
    set_requires_grad(model.set_encoder, True)


def make_optimizer(model: RelationModel, learning_rate: float) -> torch.optim.Optimizer:
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise ValueError("No trainable parameters are enabled for the current phase.")
    return torch.optim.AdamW(trainable_parameters, lr=learning_rate)


def run_training_phase(
    *,
    model: RelationModel,
    loaders: dict,
    criterion,
    device: torch.device,
    learning_rate: float,
    num_epochs: int,
    phase_name: str,
) -> None:
    configure_training_phase(model, phase_name)
    optimizer = make_optimizer(model, learning_rate)

    print(
        f"Starting phase '{phase_name}' with "
        f"{count_trainable_parameters(model):,} trainable parameters."
    )
    print(
        "  trainable breakdown:",
        {
            "sequence_encoder": count_trainable_parameters(model.sequence_encoder),
            "set_encoder": count_trainable_parameters(model.set_encoder),
            "relation_head": count_trainable_parameters(model.relation_head),
        },
    )

    for epoch in range(1, num_epochs + 1):
        train_stats = train_one_epoch(
            model=model,
            loader=loaders["train_loader"],
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )
        val_stats = evaluate_one_epoch(
            model=model,
            loader=loaders["val_loader"],
            criterion=criterion,
            device=device,
        )
        print(f"Phase {phase_name} | Epoch {epoch}")
        print("  train:", format_epoch_stats(train_stats))
        print("  val:  ", format_epoch_stats(val_stats))


def run_one_epoch_for_phase(
    *,
    model: RelationModel,
    loaders: dict,
    criterion,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    phase_name: str,
    epoch_index: int,
) -> None:
    train_stats = train_one_epoch(
        model=model,
        loader=loaders["train_loader"],
        optimizer=optimizer,
        criterion=criterion,
        device=device,
    )
    val_stats = evaluate_one_epoch(
        model=model,
        loader=loaders["val_loader"],
        criterion=criterion,
        device=device,
    )
    print(f"Phase {phase_name} | Epoch {epoch_index}")
    print("  train:", format_epoch_stats(train_stats))
    print("  val:  ", format_epoch_stats(val_stats))


def build_phase_optimizer(
    *,
    model: RelationModel,
    learning_rate: float,
    phase_name: str,
) -> torch.optim.Optimizer:
    configure_training_phase(model, phase_name)
    optimizer = make_optimizer(model, learning_rate)
    print(
        f"Starting phase '{phase_name}' with "
        f"{count_trainable_parameters(model):,} trainable parameters."
    )
    print(
        "  trainable breakdown:",
        {
            "sequence_encoder": count_trainable_parameters(model.sequence_encoder),
            "set_encoder": count_trainable_parameters(model.set_encoder),
            "relation_head": count_trainable_parameters(model.relation_head),
        },
    )
    return optimizer


def run_staged_training(
    *,
    model: RelationModel,
    loaders: dict,
    criterion,
    device: torch.device,
    learning_rate: float,
    encoder_only_epochs: int,
    set_only_epochs: int,
    head_only_epochs: int,
    alternate_epochs: bool,
    set_phase_last: bool,
) -> None:
    if alternate_epochs:
        phase_epochs_remaining = {
            "encoder_only": encoder_only_epochs,
            "set_only": set_only_epochs,
            "head_only": head_only_epochs,
        }
        optimizers: dict[str, torch.optim.Optimizer] = {}
        for phase_name, num_epochs in phase_epochs_remaining.items():
            if num_epochs > 0:
                optimizers[phase_name] = build_phase_optimizer(
                    model=model,
                    learning_rate=learning_rate,
                    phase_name=phase_name,
                )

        if set_phase_last:
            alternating_phase_order = ["encoder_only", "head_only"]
        else:
            alternating_phase_order = ["encoder_only", "set_only", "head_only"]

        phase_epoch_indices = {
            "encoder_only": 0,
            "set_only": 0,
            "head_only": 0,
        }

        while any(
            phase_epochs_remaining[phase_name] > 0
            for phase_name in alternating_phase_order
        ):
            for phase_name in alternating_phase_order:
                if phase_epochs_remaining[phase_name] <= 0:
                    continue
                phase_epoch_indices[phase_name] += 1
                configure_training_phase(model, phase_name)
                run_one_epoch_for_phase(
                    model=model,
                    loaders=loaders,
                    criterion=criterion,
                    device=device,
                    optimizer=optimizers[phase_name],
                    phase_name=phase_name,
                    epoch_index=phase_epoch_indices[phase_name],
                )
                phase_epochs_remaining[phase_name] -= 1

        if set_phase_last and phase_epochs_remaining["set_only"] > 0:
            run_training_phase(
                model=model,
                loaders=loaders,
                criterion=criterion,
                device=device,
                learning_rate=learning_rate,
                num_epochs=phase_epochs_remaining["set_only"],
                phase_name="set_only",
            )
        return

    ordered_phases: list[tuple[str, int]]
    if set_phase_last:
        ordered_phases = [
            ("encoder_only", encoder_only_epochs),
            ("head_only", head_only_epochs),
            ("set_only", set_only_epochs),
        ]
    else:
        ordered_phases = [
            ("encoder_only", encoder_only_epochs),
            ("set_only", set_only_epochs),
            ("head_only", head_only_epochs),
        ]

    for phase_name, num_epochs in ordered_phases:
        if num_epochs > 0:
            run_training_phase(
                model=model,
                loaders=loaders,
                criterion=criterion,
                device=device,
                learning_rate=learning_rate,
                num_epochs=num_epochs,
                phase_name=phase_name,
            )


def run_end_to_end_training(
    *,
    model: RelationModel,
    loaders: dict,
    criterion,
    device: torch.device,
    learning_rate: float,
    num_epochs: int,
) -> None:
    enable_all_relevant_parameters(model)
    optimizer = make_optimizer(model, learning_rate)

    print(
        f"Starting end_to_end training with "
        f"{count_trainable_parameters(model):,} trainable parameters."
    )
    print(
        "  trainable breakdown:",
        {
            "sequence_encoder": count_trainable_parameters(model.sequence_encoder),
            "set_encoder": count_trainable_parameters(model.set_encoder),
            "relation_head": count_trainable_parameters(model.relation_head),
        },
    )

    for epoch in range(1, num_epochs + 1):
        train_stats = train_one_epoch(
            model=model,
            loader=loaders["train_loader"],
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )
        val_stats = evaluate_one_epoch(
            model=model,
            loader=loaders["val_loader"],
            criterion=criterion,
            device=device,
        )
        print(f"End-to-end | Epoch {epoch}")
        print("  train:", format_epoch_stats(train_stats))
        print("  val:  ", format_epoch_stats(val_stats))


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    data_dirs = discover_data_dirs(args.data_root)
    print("Using data directories:")
    for data_dir in data_dirs:
        print(f"  - {data_dir}")

    loaders = build_loader_bundle(
        data_dirs=data_dirs,
        episode_size=args.episode_size,
        train_episodes_per_epoch=args.train_episodes_per_epoch,
        eval_episodes=args.eval_episodes,
        batch_size=args.batch_size,
        seed=args.seed,
        families_per_episode_min=args.families_per_episode_min,
        families_per_episode_max=args.families_per_episode_max,
        length_sampling=args.length_sampling,
    )

    criterion, weight_info = build_loss_from_loader(
        loaders["train_loader"],
        diagonal_weight=args.diagonal_weight,
    )
    weight_info["positive_weight"] = float(weight_info["positive_weight"]) * args.positive_weight_scale
    criterion.positive_weight = float(weight_info["positive_weight"])
    print("Loss weights:")
    print(weight_info)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = build_model_from_args(args).to(device)
    encoder_only_epochs = (
        args.encoder_only_epochs if args.encoder_only_epochs is not None else args.epochs
    )
    set_only_epochs = (
        args.set_only_epochs if args.set_only_epochs is not None else args.epochs
    )
    head_only_epochs = (
        args.head_only_epochs if args.head_only_epochs is not None else args.epochs
    )

    if args.disable_phase_training:
        run_end_to_end_training(
            model=model,
            loaders=loaders,
            criterion=criterion,
            device=device,
            learning_rate=args.learning_rate,
            num_epochs=args.epochs,
        )
    else:
        run_staged_training(
            model=model,
            loaders=loaders,
            criterion=criterion,
            device=device,
            learning_rate=args.learning_rate,
            encoder_only_epochs=encoder_only_epochs,
            set_only_epochs=set_only_epochs,
            head_only_epochs=head_only_epochs,
            alternate_epochs=args.alternate_epochs,
            set_phase_last=args.set_phase_last,
        )

    test_stats = test_model(
        model=model,
        loader=loaders["test_loader"],
        criterion=criterion,
        device=device,
    )
    print("test:", format_epoch_stats(test_stats))


if __name__ == "__main__":
    main()
