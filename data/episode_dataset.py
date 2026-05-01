#!/usr/bin/env python3
"""PyTorch episode dataset for synthetic DNA grouping.

Each episode contains N padded DNA sequences and an NxN relation matrix
whose entries are 1 when the two sequences come from the same family.

The dataset can mix multiple source directories (for example uniform-GC and
segmented-GC datasets) as long as they share the same JSONL schema.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import torch
from torch.utils.data import BatchSampler, Dataset, DataLoader


DNA_VOCAB = {"A": 0, "C": 1, "G": 2, "T": 3}
PAD_TOKEN_ID = 4


@dataclass(frozen=True)
class SequenceRecord:
    sequence: str
    family_id: int
    length: int
    split: str
    seq_id: int
    source_name: str
    uses_segmented_gc: bool


@dataclass(frozen=True)
class EpisodeSpec:
    length: int
    sequence_lengths: tuple[int, ...]
    record_indices: tuple[int, ...]
    family_ids: tuple[int, ...]
    source_names: tuple[str, ...]
    segmented_flags: tuple[bool, ...]


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def encode_sequence(sequence: str) -> List[int]:
    return [DNA_VOCAB[base] for base in sequence]


def pad_encoded_sequences(
    encoded_sequences: Sequence[Sequence[int]], max_length: int | None = None
) -> torch.Tensor:
    if not encoded_sequences:
        raise ValueError("Cannot pad an empty sequence list.")
    padded_length = max_length
    if padded_length is None:
        padded_length = max(len(sequence) for sequence in encoded_sequences)

    padded = torch.full(
        (len(encoded_sequences), padded_length),
        fill_value=PAD_TOKEN_ID,
        dtype=torch.long,
    )
    for row, sequence in enumerate(encoded_sequences):
        if len(sequence) > padded_length:
            raise ValueError("Encoded sequence is longer than the requested padding length.")
        padded[row, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
    return padded


def build_relation_matrix(family_ids: Sequence[int]) -> torch.Tensor:
    family_tensor = torch.tensor(family_ids, dtype=torch.long)
    return (family_tensor[:, None] == family_tensor[None, :]).to(torch.float32)


def integer_composition(
    rng: random.Random, total: int, parts: int, min_value: int = 1
) -> List[int]:
    if parts * min_value > total:
        raise ValueError("Cannot compose total with the requested minimum value.")
    if parts == 1:
        return [total]

    adjusted_total = total - parts * min_value
    cut_points = sorted(rng.sample(range(adjusted_total + parts - 1), parts - 1))
    pieces: List[int] = []
    start = -1
    for cut in cut_points + [adjusted_total + parts - 1]:
        pieces.append(cut - start - 1)
        start = cut
    return [piece + min_value for piece in pieces]


def count_positive_pairs_from_family_counts(counts: Sequence[int]) -> int:
    return sum(count * (count - 1) // 2 for count in counts)


def count_total_pairs(num_items: int) -> int:
    return num_items * (num_items - 1) // 2


class DnaEpisodeDataset(Dataset):
    """Samples fixed-size same-length episodes from one or more dataset roots."""

    def __init__(
        self,
        data_dirs: Sequence[str | Path],
        split: str,
        episode_size: int,
        episodes_per_epoch: int,
        seed: int = 0,
        families_per_episode_min: int = 2,
        families_per_episode_max: int | None = None,
        length_sampling: str = "uniform",
        positive_fraction_min: float = 0.25,
        positive_fraction_max: float = 0.75,
        max_episode_sampling_attempts: int = 1000,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be one of train/val/test.")
        if episode_size < 2:
            raise ValueError("episode_size must be at least 2.")
        if episodes_per_epoch <= 0:
            raise ValueError("episodes_per_epoch must be positive.")
        if families_per_episode_min <= 0:
            raise ValueError("families_per_episode_min must be positive.")
        if length_sampling not in {"uniform", "proportional"}:
            raise ValueError("length_sampling must be 'uniform' or 'proportional'.")
        if not 0.0 <= positive_fraction_min <= 1.0:
            raise ValueError("positive_fraction_min must lie in [0, 1].")
        if not 0.0 <= positive_fraction_max <= 1.0:
            raise ValueError("positive_fraction_max must lie in [0, 1].")
        if positive_fraction_min > positive_fraction_max:
            raise ValueError("positive_fraction_min cannot exceed positive_fraction_max.")
        if max_episode_sampling_attempts <= 0:
            raise ValueError("max_episode_sampling_attempts must be positive.")

        self.split = split
        self.episode_size = episode_size
        self.episodes_per_epoch = episodes_per_epoch
        self.seed = seed
        self.length_sampling = length_sampling
        self.families_per_episode_min = families_per_episode_min
        self.families_per_episode_max = (
            families_per_episode_max
            if families_per_episode_max is not None
            else episode_size
        )
        self.positive_fraction_min = positive_fraction_min
        self.positive_fraction_max = positive_fraction_max
        self.max_episode_sampling_attempts = max_episode_sampling_attempts
        if self.families_per_episode_min > self.families_per_episode_max:
            raise ValueError(
                "families_per_episode_min cannot exceed families_per_episode_max."
            )

        self.data_dirs = [Path(data_dir) for data_dir in data_dirs]
        self.records: List[SequenceRecord] = []
        self.indices_by_family: Dict[int, List[int]] = defaultdict(list)
        self.lengths: List[int] = []

        self._load_records()
        self.length_weights = self._build_length_weights()
        self.episode_specs = self._build_episode_specs()

    def _load_records(self) -> None:
        for data_dir in self.data_dirs:
            split_path = data_dir / f"{self.split}.jsonl"
            if not split_path.exists():
                raise FileNotFoundError(f"Missing split file: {split_path}")

            source_name = data_dir.name
            for row in read_jsonl(split_path):
                record = SequenceRecord(
                    sequence=row["sequence"],
                    family_id=int(row["family_id"]),
                    length=int(row["length"]),
                    split=row["split"],
                    seq_id=int(row["seq_id"]),
                    source_name=source_name,
                    uses_segmented_gc=bool(row.get("uses_segmented_gc", False)),
                )
                record_index = len(self.records)
                self.records.append(record)
                self.indices_by_family[record.family_id].append(record_index)

        eligible_families = [
            family_id
            for family_id, indices in self.indices_by_family.items()
            if len(indices) >= 1
        ]
        if len(eligible_families) < min(
            self.episode_size, self.families_per_episode_min
        ):
            raise ValueError("No eligible families found for the requested split.")

        self.lengths = sorted({record.length for record in self.records})

    def _build_length_weights(self) -> List[float]:
        if self.length_sampling == "uniform":
            return [1.0 for _ in self.lengths]
        length_counts = defaultdict(int)
        for record in self.records:
            length_counts[record.length] += 1
        return [float(length_counts[length]) for length in self.lengths]

    def _choose_length(self, rng: random.Random) -> int:
        return rng.choices(self.lengths, weights=self.length_weights, k=1)[0]

    def _is_balanced_episode(self, counts: Sequence[int]) -> bool:
        total_pairs = count_total_pairs(self.episode_size)
        if total_pairs == 0:
            return False
        positive_pairs = count_positive_pairs_from_family_counts(counts)
        positive_fraction = positive_pairs / total_pairs
        return self.positive_fraction_min <= positive_fraction <= self.positive_fraction_max

    def _sample_episode_spec_once(self, rng: random.Random) -> EpisodeSpec:
        family_ids = sorted(self.indices_by_family.keys())

        max_families = min(
            self.families_per_episode_max, self.episode_size, len(family_ids)
        )
        min_families = min(self.families_per_episode_min, max_families)
        num_families = rng.randint(min_families, max_families)
        chosen_families = rng.sample(family_ids, num_families)

        counts = integer_composition(rng, self.episode_size, num_families, min_value=1)
        if not self._is_balanced_episode(counts):
            raise ValueError("Sampled episode does not satisfy the positive/negative balance constraint.")

        sampled_indices: List[int] = []
        sampled_family_ids: List[int] = []
        sampled_source_names: List[str] = []
        sampled_segmented_flags: List[bool] = []

        for family_id, count in zip(chosen_families, counts):
            candidates = self.indices_by_family[family_id]
            if len(candidates) >= count:
                selected = rng.sample(candidates, count)
            else:
                selected = [rng.choice(candidates) for _ in range(count)]
            for record_index in selected:
                record = self.records[record_index]
                sampled_indices.append(record_index)
                sampled_family_ids.append(record.family_id)
                sampled_source_names.append(record.source_name)
                sampled_segmented_flags.append(record.uses_segmented_gc)

        permutation = list(range(self.episode_size))
        rng.shuffle(permutation)
        shuffled_indices = tuple(sampled_indices[i] for i in permutation)
        shuffled_family_ids = tuple(sampled_family_ids[i] for i in permutation)
        shuffled_source_names = tuple(sampled_source_names[i] for i in permutation)
        shuffled_segmented_flags = tuple(sampled_segmented_flags[i] for i in permutation)
        shuffled_lengths = tuple(
            self.records[record_index].length for record_index in shuffled_indices
        )

        return EpisodeSpec(
            length=max(shuffled_lengths),
            sequence_lengths=shuffled_lengths,
            record_indices=shuffled_indices,
            family_ids=shuffled_family_ids,
            source_names=shuffled_source_names,
            segmented_flags=shuffled_segmented_flags,
        )

    def _sample_episode_spec(self, rng: random.Random) -> EpisodeSpec:
        for _ in range(self.max_episode_sampling_attempts):
            try:
                return self._sample_episode_spec_once(rng)
            except ValueError:
                continue
        raise RuntimeError(
            "Failed to sample an episode satisfying the positive/negative balance constraint. "
            "Try relaxing the family-count or positive-fraction settings."
        )

    def _build_episode_specs(self) -> List[EpisodeSpec]:
        rng = random.Random(self.seed)
        return [self._sample_episode_spec(rng) for _ in range(self.episodes_per_epoch)]

    def __len__(self) -> int:
        return len(self.episode_specs)

    def __getitem__(self, index: int) -> dict:
        spec = self.episode_specs[index]
        records = [self.records[record_index] for record_index in spec.record_indices]
        sequences = [record.sequence for record in records]
        encoded_sequences = [encode_sequence(sequence) for sequence in sequences]
        input_ids = pad_encoded_sequences(encoded_sequences, max_length=spec.length)
        family_ids = torch.tensor(spec.family_ids, dtype=torch.long)
        label_matrix = build_relation_matrix(spec.family_ids)

        return {
            "input_ids": input_ids,
            "sequences": sequences,
            "family_ids": family_ids,
            "label_matrix": label_matrix,
            "length": spec.length,
            "sequence_lengths": torch.tensor(spec.sequence_lengths, dtype=torch.long),
            "source_names": list(spec.source_names),
            "uses_segmented_gc": torch.tensor(spec.segmented_flags, dtype=torch.bool),
        }


class LengthBucketBatchSampler(BatchSampler):
    """Groups pre-sampled episodes by padded episode length."""

    def __init__(
        self,
        dataset: DnaEpisodeDataset,
        batch_size: int,
        drop_last: bool = False,
        shuffle: bool = True,
        seed: int = 0,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        self._batches = self._build_batches()

    def _build_batches(self) -> List[List[int]]:
        rng = random.Random(self.seed)
        indices_by_length: Dict[int, List[int]] = defaultdict(list)
        for index, spec in enumerate(self.dataset.episode_specs):
            indices_by_length[spec.length].append(index)

        all_batches: List[List[int]] = []
        for length in sorted(indices_by_length):
            indices = list(indices_by_length[length])
            if self.shuffle:
                rng.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start : start + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                all_batches.append(batch)

        if self.shuffle:
            rng.shuffle(all_batches)
        return all_batches

    def __iter__(self):
        yield from self._batches

    def __len__(self) -> int:
        return len(self._batches)


def episode_collate_fn(batch: Sequence[dict]) -> dict:
    if not batch:
        raise ValueError("Cannot collate an empty batch.")

    max_length = max(int(item["length"]) for item in batch)
    padded_input_ids = []
    for item in batch:
        input_ids = item["input_ids"]
        if input_ids.shape[-1] == max_length:
            padded_input_ids.append(input_ids)
            continue
        padded = torch.full(
            (input_ids.shape[0], max_length),
            fill_value=PAD_TOKEN_ID,
            dtype=input_ids.dtype,
        )
        padded[:, : input_ids.shape[-1]] = input_ids
        padded_input_ids.append(padded)

    return {
        "input_ids": torch.stack(padded_input_ids, dim=0),
        "family_ids": torch.stack([item["family_ids"] for item in batch], dim=0),
        "label_matrix": torch.stack([item["label_matrix"] for item in batch], dim=0),
        "length": max_length,
        "sequence_lengths": torch.stack(
            [item["sequence_lengths"] for item in batch], dim=0
        ),
        "sequences": [item["sequences"] for item in batch],
        "source_names": [item["source_names"] for item in batch],
        "uses_segmented_gc": torch.stack(
            [item["uses_segmented_gc"] for item in batch], dim=0
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build episode-style PyTorch datasets for DNA grouping."
    )
    parser.add_argument(
        "--data-dirs",
        nargs="+",
        required=True,
        help="One or more dataset directories that contain train/val/test JSONL files.",
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "val", "test"],
        help="Which split to load.",
    )
    parser.add_argument(
        "--episode-size",
        type=int,
        required=True,
        help="Number of sequences per episode (N).",
    )
    parser.add_argument(
        "--episodes-per-epoch",
        type=int,
        default=128,
        help="Number of sampled episodes to expose in one epoch.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Number of episodes per batch.",
    )
    parser.add_argument(
        "--families-per-episode-min",
        type=int,
        default=2,
        help="Minimum number of families that appear in one episode.",
    )
    parser.add_argument(
        "--families-per-episode-max",
        type=int,
        default=None,
        help="Maximum number of families that appear in one episode.",
    )
    parser.add_argument(
        "--length-sampling",
        default="uniform",
        choices=["uniform", "proportional"],
        help="How to sample sequence lengths across dataset roots.",
    )
    parser.add_argument(
        "--positive-fraction-min",
        type=float,
        default=0.25,
        help="Minimum fraction of upper-triangle same-family pairs in one episode.",
    )
    parser.add_argument(
        "--positive-fraction-max",
        type=float,
        default=0.75,
        help="Maximum fraction of upper-triangle same-family pairs in one episode.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for episode sampling and batch construction.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = DnaEpisodeDataset(
        data_dirs=args.data_dirs,
        split=args.split,
        episode_size=args.episode_size,
        episodes_per_epoch=args.episodes_per_epoch,
        seed=args.seed,
        families_per_episode_min=args.families_per_episode_min,
        families_per_episode_max=args.families_per_episode_max,
        length_sampling=args.length_sampling,
        positive_fraction_min=args.positive_fraction_min,
        positive_fraction_max=args.positive_fraction_max,
    )
    batch_sampler = LengthBucketBatchSampler(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        seed=args.seed,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=episode_collate_fn,
    )

    first_batch = next(iter(loader))
    print(f"num_records={len(dataset.records)}")
    print(f"num_episodes={len(dataset)}")
    print(f"available_lengths={dataset.lengths}")
    print(f"batch_input_shape={tuple(first_batch['input_ids'].shape)}")
    print(f"batch_label_shape={tuple(first_batch['label_matrix'].shape)}")
    print(f"batch_length={first_batch['length']}")
    print(f"batch_sequence_lengths={first_batch['sequence_lengths'].tolist()}")
    print(
        "segmented_counts="
        f"{int(first_batch['uses_segmented_gc'].sum().item())}/"
        f"{first_batch['uses_segmented_gc'].numel()}"
    )


if __name__ == "__main__":
    main()
