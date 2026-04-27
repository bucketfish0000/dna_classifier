"""Function-only training utilities for DNA grouping models."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Sequence

import torch
from torch.utils.data import DataLoader

from dna_grouper.data.episode_dataset import (
    DnaEpisodeDataset,
    LengthBucketBatchSampler,
    episode_collate_fn,
)
from dna_grouper.training.loss import WeightedRelationBCELoss


def make_loader(
    dataset: DnaEpisodeDataset,
    batch_size: int,
    shuffle: bool,
    drop_last: bool,
    seed: int,
) -> DataLoader:
    batch_sampler = LengthBucketBatchSampler(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        seed=seed,
    )
    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=episode_collate_fn,
    )


def build_loader_bundle(
    data_dirs: Sequence[str],
    episode_size: int,
    train_episodes_per_epoch: int,
    eval_episodes: int,
    batch_size: int,
    seed: int = 0,
    families_per_episode_min: int = 2,
    families_per_episode_max: int | None = None,
    length_sampling: str = "uniform",
    positive_fraction_min: float = 0.25,
    positive_fraction_max: float = 0.75,
) -> Dict[str, Any]:
    """Build train/val/test datasets and loaders for one fixed episode size N."""

    common_kwargs = dict(
        data_dirs=data_dirs,
        episode_size=episode_size,
        families_per_episode_min=families_per_episode_min,
        families_per_episode_max=families_per_episode_max,
        length_sampling=length_sampling,
        positive_fraction_min=positive_fraction_min,
        positive_fraction_max=positive_fraction_max,
    )
    train_dataset = DnaEpisodeDataset(
        split="train",
        episodes_per_epoch=train_episodes_per_epoch,
        seed=seed,
        **common_kwargs,
    )
    val_dataset = DnaEpisodeDataset(
        split="val",
        episodes_per_epoch=eval_episodes,
        seed=seed + 1,
        **common_kwargs,
    )
    test_dataset = DnaEpisodeDataset(
        split="test",
        episodes_per_epoch=eval_episodes,
        seed=seed + 2,
        **common_kwargs,
    )

    return {
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
        "train_loader": make_loader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            seed=seed,
        ),
        "val_loader": make_loader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            seed=seed + 1,
        ),
        "test_loader": make_loader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            seed=seed + 2,
        ),
    }


def upper_triangle_mask(size: int, device: torch.device) -> torch.Tensor:
    return torch.triu(torch.ones(size, size, device=device, dtype=torch.bool), diagonal=1)


def estimate_class_weights_from_loader(
    loader: Iterable[dict],
    max_batches: int | None = None,
    clamp_positive_weight: float | None = None,
) -> Dict[str, float | int]:
    """Estimate upper-triangle positive/negative weights from sampled episodes."""

    num_positive = 0
    num_negative = 0
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        labels = batch["label_matrix"]
        mask = upper_triangle_mask(labels.shape[-1], labels.device).unsqueeze(0)
        mask = mask.expand(labels.shape[0], -1, -1)
        upper = labels[mask]
        num_positive += int((upper > 0.5).sum().item())
        num_negative += int((upper <= 0.5).sum().item())

    if num_positive == 0:
        positive_weight = 1.0
    else:
        positive_weight = num_negative / num_positive
    if clamp_positive_weight is not None:
        positive_weight = min(positive_weight, clamp_positive_weight)

    return {
        "positive_weight": float(positive_weight),
        "negative_weight": 1.0,
        "num_positive": num_positive,
        "num_negative": num_negative,
    }


def build_loss_from_loader(
    loader: Iterable[dict],
    diagonal_weight: float = 0.0,
    max_batches_for_weight_estimate: int | None = None,
    clamp_positive_weight: float | None = None,
) -> tuple[WeightedRelationBCELoss, Dict[str, float | int]]:
    weight_estimate = estimate_class_weights_from_loader(
        loader=loader,
        max_batches=max_batches_for_weight_estimate,
        clamp_positive_weight=clamp_positive_weight,
    )
    criterion = WeightedRelationBCELoss(
        positive_weight=float(weight_estimate["positive_weight"]),
        negative_weight=float(weight_estimate["negative_weight"]),
        diagonal_weight=diagonal_weight,
    )
    return criterion, weight_estimate


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def cosine_distance_matrix(embeddings: torch.Tensor) -> torch.Tensor:
    normalized = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
    cosine_similarity = normalized @ normalized.transpose(0, 1)
    return 1.0 - cosine_similarity


def collect_pair_metrics(logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, int]:
    return collect_pair_metrics_at_threshold(logits, targets, threshold=0.5)


def collect_pair_metrics_at_threshold(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float,
) -> Dict[str, int]:
    mask = upper_triangle_mask(logits.shape[-1], logits.device).unsqueeze(0)
    mask = mask.expand(logits.shape[0], -1, -1)

    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).to(targets.dtype)

    target_upper = targets[mask]
    pred_upper = preds[mask]

    true_positive = int(((pred_upper > 0.5) & (target_upper > 0.5)).sum().item())
    true_negative = int(((pred_upper <= 0.5) & (target_upper <= 0.5)).sum().item())
    false_positive = int(((pred_upper > 0.5) & (target_upper <= 0.5)).sum().item())
    false_negative = int(((pred_upper <= 0.5) & (target_upper > 0.5)).sum().item())

    return {
        "tp": true_positive,
        "tn": true_negative,
        "fp": false_positive,
        "fn": false_negative,
        "num_pairs": int(target_upper.numel()),
        "num_positive_pairs": int((target_upper > 0.5).sum().item()),
        "num_negative_pairs": int((target_upper <= 0.5).sum().item()),
    }


@torch.no_grad()
def collect_upper_triangle_probs_and_targets(
    model: torch.nn.Module,
    loader: Iterable[dict],
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = torch.device(device)
    model.eval()

    all_probs = []
    all_targets = []
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        logits = model(batch["input_ids"])
        probs = torch.sigmoid(logits)
        mask = upper_triangle_mask(logits.shape[-1], logits.device).unsqueeze(0)
        mask = mask.expand(logits.shape[0], -1, -1)
        all_probs.append(probs[mask].detach().cpu())
        all_targets.append(batch["label_matrix"][mask].detach().cpu())

    if not all_probs:
        raise ValueError("Loader produced no batches for threshold sweep.")

    return torch.cat(all_probs, dim=0), torch.cat(all_targets, dim=0)


def summarize_binary_predictions(
    probs: torch.Tensor,
    targets: torch.Tensor,
    threshold: float,
) -> Dict[str, float | int]:
    preds = (probs >= threshold).to(targets.dtype)

    tp = int(((preds > 0.5) & (targets > 0.5)).sum().item())
    tn = int(((preds <= 0.5) & (targets <= 0.5)).sum().item())
    fp = int(((preds > 0.5) & (targets <= 0.5)).sum().item())
    fn = int(((preds <= 0.5) & (targets > 0.5)).sum().item())
    num_pairs = int(targets.numel())

    accuracy = (tp + tn) / max(num_pairs, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 0.0 if precision + recall == 0.0 else 2 * precision * recall / (precision + recall)

    return {
        "threshold": float(threshold),
        "pair_accuracy": accuracy,
        "pair_precision": precision,
        "pair_recall": recall,
        "pair_f1": f1,
        "num_pairs": num_pairs,
        "num_positive_pairs": int((targets > 0.5).sum().item()),
        "num_negative_pairs": int((targets <= 0.5).sum().item()),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


@torch.no_grad()
def sweep_thresholds(
    model: torch.nn.Module,
    loader: Iterable[dict],
    device: torch.device | str,
    thresholds: Sequence[float] | None = None,
    selection_metric: str = "pair_f1",
) -> Dict[str, Any]:
    if thresholds is None:
        thresholds = [i / 20 for i in range(1, 20)]
    if selection_metric not in {"pair_f1", "pair_accuracy", "pair_precision", "pair_recall"}:
        raise ValueError("Unsupported selection_metric for threshold sweep.")

    probs, targets = collect_upper_triangle_probs_and_targets(model, loader, device)
    candidates = [
        summarize_binary_predictions(probs, targets, threshold)
        for threshold in thresholds
    ]
    best = max(candidates, key=lambda item: (float(item[selection_metric]), -abs(item["threshold"] - 0.5)))
    return {"best": best, "candidates": candidates}


def empty_metric_counts() -> Dict[str, int]:
    return {
        "tp": 0,
        "tn": 0,
        "fp": 0,
        "fn": 0,
        "num_pairs": 0,
        "num_positive_pairs": 0,
        "num_negative_pairs": 0,
    }


def update_metric_counts(total: Dict[str, int], batch_counts: Dict[str, int]) -> None:
    for key in total:
        total[key] += batch_counts[key]


def finalize_epoch_stats(
    total_loss_sum: float,
    upper_loss_sum: float,
    diagonal_loss_sum: float,
    num_batches: int,
    metric_counts: Dict[str, int],
) -> Dict[str, float | int]:
    if num_batches == 0:
        raise ValueError("Cannot finalize epoch stats with zero batches.")

    tp = metric_counts["tp"]
    tn = metric_counts["tn"]
    fp = metric_counts["fp"]
    fn = metric_counts["fn"]
    num_pairs = metric_counts["num_pairs"]

    accuracy = (tp + tn) / max(num_pairs, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)

    return {
        "loss": total_loss_sum / num_batches,
        "upper_loss": upper_loss_sum / num_batches,
        "diagonal_loss": diagonal_loss_sum / num_batches,
        "pair_accuracy": accuracy,
        "pair_precision": precision,
        "pair_recall": recall,
        "num_pairs": num_pairs,
        "num_positive_pairs": metric_counts["num_positive_pairs"],
        "num_negative_pairs": metric_counts["num_negative_pairs"],
    }


def train_one_epoch(
    model: torch.nn.Module,
    loader: Iterable[dict],
    optimizer: torch.optim.Optimizer,
    criterion: WeightedRelationBCELoss,
    device: torch.device | str,
) -> Dict[str, float | int]:
    device = torch.device(device)
    model.train()

    total_loss_sum = 0.0
    upper_loss_sum = 0.0
    diagonal_loss_sum = 0.0
    num_batches = 0
    metric_counts = empty_metric_counts()

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)

        logits = model(batch["input_ids"])
        loss_breakdown = criterion(logits, batch["label_matrix"])
        loss_breakdown.total_loss.backward()
        optimizer.step()

        update_metric_counts(
            metric_counts, collect_pair_metrics(logits.detach(), batch["label_matrix"])
        )
        total_loss_sum += float(loss_breakdown.total_loss.detach().item())
        upper_loss_sum += float(loss_breakdown.upper_loss.detach().item())
        diagonal_loss_sum += float(loss_breakdown.diagonal_loss.detach().item())
        num_batches += 1

    return finalize_epoch_stats(
        total_loss_sum=total_loss_sum,
        upper_loss_sum=upper_loss_sum,
        diagonal_loss_sum=diagonal_loss_sum,
        num_batches=num_batches,
        metric_counts=metric_counts,
    )


@torch.no_grad()
def evaluate_one_epoch(
    model: torch.nn.Module,
    loader: Iterable[dict],
    criterion: WeightedRelationBCELoss,
    device: torch.device | str,
) -> Dict[str, float | int]:
    device = torch.device(device)
    model.eval()

    total_loss_sum = 0.0
    upper_loss_sum = 0.0
    diagonal_loss_sum = 0.0
    num_batches = 0
    metric_counts = empty_metric_counts()

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        logits = model(batch["input_ids"])
        loss_breakdown = criterion(logits, batch["label_matrix"])

        update_metric_counts(metric_counts, collect_pair_metrics(logits, batch["label_matrix"]))
        total_loss_sum += float(loss_breakdown.total_loss.item())
        upper_loss_sum += float(loss_breakdown.upper_loss.item())
        diagonal_loss_sum += float(loss_breakdown.diagonal_loss.item())
        num_batches += 1

    return finalize_epoch_stats(
        total_loss_sum=total_loss_sum,
        upper_loss_sum=upper_loss_sum,
        diagonal_loss_sum=diagonal_loss_sum,
        num_batches=num_batches,
        metric_counts=metric_counts,
    )


def test_model(
    model: torch.nn.Module,
    loader: Iterable[dict],
    criterion: WeightedRelationBCELoss,
    device: torch.device | str,
) -> Dict[str, float | int]:
    return evaluate_one_epoch(
        model=model,
        loader=loader,
        criterion=criterion,
        device=device,
    )


def format_epoch_stats(stats: Dict[str, float | int]) -> Dict[str, float | int]:
    return {
        "loss": round(float(stats["loss"]), 6),
        "upper_loss": round(float(stats["upper_loss"]), 6),
        "diagonal_loss": round(float(stats["diagonal_loss"]), 6),
        "pair_accuracy": round(float(stats["pair_accuracy"]), 6),
        "pair_precision": round(float(stats["pair_precision"]), 6),
        "pair_recall": round(float(stats["pair_recall"]), 6),
        "num_pairs": int(stats["num_pairs"]),
        "num_positive_pairs": int(stats["num_positive_pairs"]),
        "num_negative_pairs": int(stats["num_negative_pairs"]),
    }


def format_threshold_stats(stats: Dict[str, float | int]) -> Dict[str, float | int]:
    return {
        "threshold": round(float(stats["threshold"]), 4),
        "pair_accuracy": round(float(stats["pair_accuracy"]), 6),
        "pair_precision": round(float(stats["pair_precision"]), 6),
        "pair_recall": round(float(stats["pair_recall"]), 6),
        "pair_f1": round(float(stats["pair_f1"]), 6),
        "num_pairs": int(stats["num_pairs"]),
        "num_positive_pairs": int(stats["num_positive_pairs"]),
        "num_negative_pairs": int(stats["num_negative_pairs"]),
    }


@torch.no_grad()
def get_first_logit_matrix(
    model: torch.nn.Module,
    loader: Iterable[dict],
    device: torch.device | str,
) -> torch.Tensor:
    device = torch.device(device)
    model.eval()

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        logits = model(batch["input_ids"])
        return logits[0].detach().cpu()

    raise ValueError("Loader produced no batches, so no logit matrix could be retrieved.")


@torch.no_grad()
def get_first_debug_snapshot(
    model: torch.nn.Module,
    loader: Iterable[dict],
    device: torch.device | str,
) -> Dict[str, torch.Tensor]:
    device = torch.device(device)
    model.eval()

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        if not hasattr(model, "forward_debug"):
            raise ValueError("Model does not implement forward_debug for debugging snapshots.")

        debug_outputs = model.forward_debug(batch["input_ids"])
        sequence_embeddings = debug_outputs["sequence_embeddings"][0].detach().cpu()
        set_embeddings = debug_outputs["set_embeddings"][0].detach().cpu()
        logits = debug_outputs["logits"][0].detach().cpu()
        label_matrix = batch["label_matrix"][0].detach().cpu()

        return {
            "sequence_embeddings": sequence_embeddings,
            "set_embeddings": set_embeddings,
            "sequence_distance_matrix": cosine_distance_matrix(sequence_embeddings),
            "set_distance_matrix": cosine_distance_matrix(set_embeddings),
            "logits": logits,
            "label_matrix": label_matrix,
        }

    raise ValueError("Loader produced no batches, so no debug snapshot could be retrieved.")
