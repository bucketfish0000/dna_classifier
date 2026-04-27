#!/usr/bin/env python3
"""Generate synthetic DNA families for grouping experiments.

This script builds a family-level train/val/test split from synthetic DNA
templates. Each family starts from a template sequence, and member sequences are
created by independent substitution mutations with a per-sample mutation rate.

Outputs are flat JSONL files so downstream PyTorch code can build pairwise,
triplet, or episodic datasets without needing to reshape the raw data first.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


DNA_ALPHABET: Tuple[str, ...] = ("A", "C", "G", "T")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic DNA families with substitution mutations."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where JSONL and metadata outputs will be written.",
    )
    parser.add_argument(
        "--num-families",
        type=int,
        default=300,
        help="Number of template families to generate.",
    )
    parser.add_argument(
        "--family-size-min",
        type=int,
        default=32,
        help="Minimum number of mutated sequences per family.",
    )
    parser.add_argument(
        "--family-size-max",
        type=int,
        default=64,
        help="Maximum number of mutated sequences per family.",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=200,
        help="Minimum template sequence length.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=301,
        help="Maximum template sequence length.",
    )
    parser.add_argument(
        "--fixed-length",
        type=int,
        default=None,
        help="If set, generate all templates at exactly this sequence length.",
    )
    parser.add_argument(
        "--gc-min",
        type=float,
        default=0.40,
        help="Minimum per-template GC content.",
    )
    parser.add_argument(
        "--gc-max",
        type=float,
        default=0.60,
        help="Maximum per-template GC content.",
    )
    parser.add_argument(
        "--theta-min",
        type=float,
        default=0.02,
        help="Minimum per-sample substitution rate.",
    )
    parser.add_argument(
        "--theta-max",
        type=float,
        default=0.10,
        help="Maximum per-sample substitution rate.",
    )
    parser.add_argument(
        "--train-frac",
        type=float,
        default=0.8,
        help="Fraction of families assigned to train.",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.1,
        help="Fraction of families assigned to val.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed.",
    )
    parser.add_argument(
        "--max-homopolymer-run",
        type=int,
        default=8,
        help="Reject template sequences with a longer homopolymer run.",
    )
    parser.add_argument(
        "--template-distance-ratio",
        type=float,
        default=0.12,
        help=(
            "Minimum normalized Hamming distance for templates with the same length. "
            "Set to 0 to disable."
        ),
    )
    parser.add_argument(
        "--use-segmented-gc",
        action="store_true",
        help="Generate each template with 3-5 GC-content segments instead of one global GC.",
    )
    parser.add_argument(
        "--segment-count-min",
        type=int,
        default=3,
        help="Minimum number of GC segments per template when segmented GC is enabled.",
    )
    parser.add_argument(
        "--segment-count-max",
        type=int,
        default=5,
        help="Maximum number of GC segments per template when segmented GC is enabled.",
    )
    parser.add_argument(
        "--segment-min-length",
        type=int,
        default=40,
        help="Minimum segment length when segmented GC is enabled.",
    )
    parser.add_argument(
        "--segment-gc-delta",
        type=float,
        default=0.10,
        help="Maximum absolute GC perturbation around the template base GC for each segment.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.num_families <= 0:
        raise ValueError("--num-families must be positive.")
    if args.family_size_min <= 0 or args.family_size_max <= 0:
        raise ValueError("Family sizes must be positive.")
    if args.family_size_min > args.family_size_max:
        raise ValueError("--family-size-min cannot exceed --family-size-max.")
    if args.min_length <= 0 or args.max_length <= 0:
        raise ValueError("Sequence lengths must be positive.")
    if args.min_length > args.max_length:
        raise ValueError("--min-length cannot exceed --max-length.")
    if args.fixed_length is not None:
        if args.fixed_length <= 0:
            raise ValueError("--fixed-length must be positive.")
        if args.fixed_length < args.min_length or args.fixed_length > args.max_length:
            raise ValueError(
                "--fixed-length must lie within [--min-length, --max-length]."
            )
    if not 0.0 <= args.gc_min <= 1.0 or not 0.0 <= args.gc_max <= 1.0:
        raise ValueError("GC content bounds must lie in [0, 1].")
    if args.gc_min > args.gc_max:
        raise ValueError("--gc-min cannot exceed --gc-max.")
    if not 0.0 <= args.theta_min <= 1.0 or not 0.0 <= args.theta_max <= 1.0:
        raise ValueError("Theta bounds must lie in [0, 1].")
    if args.theta_min > args.theta_max:
        raise ValueError("--theta-min cannot exceed --theta-max.")
    if args.max_homopolymer_run <= 0:
        raise ValueError("--max-homopolymer-run must be positive.")
    if args.template_distance_ratio < 0.0 or args.template_distance_ratio > 1.0:
        raise ValueError("--template-distance-ratio must lie in [0, 1].")
    if args.segment_count_min <= 0 or args.segment_count_max <= 0:
        raise ValueError("Segment counts must be positive.")
    if args.segment_count_min > args.segment_count_max:
        raise ValueError("--segment-count-min cannot exceed --segment-count-max.")
    if args.segment_min_length <= 0:
        raise ValueError("--segment-min-length must be positive.")
    if args.segment_gc_delta < 0.0 or args.segment_gc_delta > 1.0:
        raise ValueError("--segment-gc-delta must lie in [0, 1].")
    if args.use_segmented_gc:
        effective_min_length = (
            args.fixed_length if args.fixed_length is not None else args.min_length
        )
        required_length = args.segment_count_min * args.segment_min_length
        if effective_min_length < required_length:
            raise ValueError(
                "Segmented GC requires the effective sequence length to satisfy "
                "--segment-count-min * --segment-min-length."
            )
    test_frac = 1.0 - args.train_frac - args.val_frac
    if args.train_frac <= 0.0 or args.val_frac < 0.0 or test_frac < 0.0:
        raise ValueError("Split fractions must produce a non-negative test fraction.")
    if not math.isclose(args.train_frac + args.val_frac + test_frac, 1.0):
        raise ValueError("Split fractions must sum to 1.0.")


def random_length(rng: random.Random, min_length: int, max_length: int) -> int:
    return rng.randint(min_length, max_length)


def random_gc_content(rng: random.Random, gc_min: float, gc_max: float) -> float:
    return rng.uniform(gc_min, gc_max)


def make_base_distribution(gc_content: float) -> Dict[str, float]:
    gc_half = gc_content / 2.0
    at_half = (1.0 - gc_content) / 2.0
    return {"A": at_half, "C": gc_half, "G": gc_half, "T": at_half}


def sample_sequence(
    rng: random.Random, length: int, base_distribution: Dict[str, float]
) -> str:
    weights = [base_distribution[base] for base in DNA_ALPHABET]
    return "".join(rng.choices(DNA_ALPHABET, weights=weights, k=length))


def clip_gc_content(gc_content: float, gc_min: float, gc_max: float) -> float:
    return max(gc_min, min(gc_max, gc_content))


def longest_homopolymer_run(sequence: str) -> int:
    longest = 0
    current = 0
    prev = ""
    for base in sequence:
        if base == prev:
            current += 1
        else:
            prev = base
            current = 1
        longest = max(longest, current)
    return longest


def hamming_distance(left: str, right: str) -> int:
    if len(left) != len(right):
        raise ValueError("Hamming distance requires equal-length strings.")
    return sum(a != b for a, b in zip(left, right))


def gc_content_of(sequence: str) -> float:
    gc_count = sequence.count("G") + sequence.count("C")
    return gc_count / len(sequence)


def mutate_sequence(
    rng: random.Random, template: str, theta: float
) -> Tuple[str, List[Dict[str, str]]]:
    sequence_chars = list(template)
    mutations: List[Dict[str, str]] = []
    for index, base in enumerate(template):
        if rng.random() >= theta:
            continue
        candidates = [candidate for candidate in DNA_ALPHABET if candidate != base]
        mutated = rng.choice(candidates)
        sequence_chars[index] = mutated
        mutations.append({"position": index, "from": base, "to": mutated})
    return "".join(sequence_chars), mutations


def choose_family_size(rng: random.Random, min_size: int, max_size: int) -> int:
    return rng.randint(min_size, max_size)


def choose_theta(rng: random.Random, theta_min: float, theta_max: float) -> float:
    return rng.uniform(theta_min, theta_max)


def sample_segment_lengths(
    rng: random.Random, total_length: int, segment_count: int, segment_min_length: int
) -> List[int]:
    lengths = [segment_min_length] * segment_count
    remaining = total_length - segment_count * segment_min_length
    for _ in range(remaining):
        lengths[rng.randrange(segment_count)] += 1
    return lengths


def build_segmented_template(
    rng: random.Random,
    length: int,
    base_gc: float,
    gc_min: float,
    gc_max: float,
    segment_count_min: int,
    segment_count_max: int,
    segment_min_length: int,
    segment_gc_delta: float,
) -> Tuple[str, List[Dict[str, object]]]:
    max_segments_by_length = length // segment_min_length
    segment_count = min(segment_count_max, max_segments_by_length)
    segment_count = rng.randint(segment_count_min, segment_count)
    segment_lengths = sample_segment_lengths(
        rng=rng,
        total_length=length,
        segment_count=segment_count,
        segment_min_length=segment_min_length,
    )

    template_parts: List[str] = []
    segments: List[Dict[str, object]] = []
    start = 0
    for segment_index, segment_length in enumerate(segment_lengths):
        delta = rng.uniform(-segment_gc_delta, segment_gc_delta)
        segment_gc = clip_gc_content(base_gc + delta, gc_min, gc_max)
        segment_distribution = make_base_distribution(segment_gc)
        segment_sequence = sample_sequence(rng, segment_length, segment_distribution)
        end = start + segment_length
        template_parts.append(segment_sequence)
        segments.append(
            {
                "segment_index": segment_index,
                "start": start,
                "end": end,
                "length": segment_length,
                "target_gc_content": segment_gc,
                "realized_gc_content": gc_content_of(segment_sequence),
            }
        )
        start = end

    return "".join(template_parts), segments


def pick_split_counts(
    num_families: int, train_frac: float, val_frac: float
) -> Dict[str, int]:
    requested = {
        "train": train_frac,
        "val": val_frac,
        "test": max(0.0, 1.0 - train_frac - val_frac),
    }
    counts = {split: 0 for split in requested}

    positive_splits = [split for split, frac in requested.items() if frac > 0.0]
    if num_families >= len(positive_splits):
        for split in positive_splits:
            counts[split] = 1
        remaining = num_families - len(positive_splits)
    else:
        remaining = num_families

    raw_targets = {
        split: max(0.0, num_families * frac - counts[split])
        for split, frac in requested.items()
    }
    for split, target in raw_targets.items():
        extra = int(math.floor(target))
        counts[split] += extra
        remaining -= extra

    remainders = sorted(
        (
            raw_targets[split] - math.floor(raw_targets[split]),
            split,
        )
        for split in requested
    )
    for _, split in reversed(remainders):
        if remaining <= 0:
            break
        counts[split] += 1
        remaining -= 1

    return counts


def assign_family_splits(
    rng: random.Random, family_ids: Sequence[int], train_frac: float, val_frac: float
) -> Dict[int, str]:
    shuffled = list(family_ids)
    rng.shuffle(shuffled)
    counts = pick_split_counts(len(shuffled), train_frac, val_frac)

    split_map: Dict[int, str] = {}
    cursor = 0
    for split_name in ("train", "val", "test"):
        split_count = counts[split_name]
        for family_id in shuffled[cursor : cursor + split_count]:
            split_map[family_id] = split_name
        cursor += split_count
    return split_map


def build_template(
    rng: random.Random,
    family_id: int,
    min_length: int,
    max_length: int,
    fixed_length: int | None,
    gc_min: float,
    gc_max: float,
    use_segmented_gc: bool,
    segment_count_min: int,
    segment_count_max: int,
    segment_min_length: int,
    segment_gc_delta: float,
    max_homopolymer_run: int,
    template_distance_ratio: float,
    existing_templates_by_length: Dict[int, List[str]],
    max_attempts: int = 1000,
) -> Dict[str, object]:
    for _ in range(max_attempts):
        length = (
            fixed_length
            if fixed_length is not None
            else random_length(rng, min_length, max_length)
        )
        target_gc = random_gc_content(rng, gc_min, gc_max)
        gc_segments: List[Dict[str, object]]
        if use_segmented_gc:
            template, gc_segments = build_segmented_template(
                rng=rng,
                length=length,
                base_gc=target_gc,
                gc_min=gc_min,
                gc_max=gc_max,
                segment_count_min=segment_count_min,
                segment_count_max=segment_count_max,
                segment_min_length=segment_min_length,
                segment_gc_delta=segment_gc_delta,
            )
        else:
            base_distribution = make_base_distribution(target_gc)
            template = sample_sequence(rng, length, base_distribution)
            gc_segments = [
                {
                    "segment_index": 0,
                    "start": 0,
                    "end": length,
                    "length": length,
                    "target_gc_content": target_gc,
                    "realized_gc_content": gc_content_of(template),
                }
            ]

        if longest_homopolymer_run(template) > max_homopolymer_run:
            continue

        if template_distance_ratio > 0.0:
            same_length_templates = existing_templates_by_length.get(length, [])
            min_distance = math.ceil(length * template_distance_ratio)
            too_close = any(
                hamming_distance(template, other) < min_distance
                for other in same_length_templates
            )
            if too_close:
                continue

        existing_templates_by_length.setdefault(length, []).append(template)
        return {
            "family_id": family_id,
            "template_id": family_id,
            "template_sequence": template,
            "length": length,
            "target_gc_content": target_gc,
            "realized_gc_content": gc_content_of(template),
            "gc_segments": gc_segments,
            "uses_segmented_gc": use_segmented_gc,
            "max_homopolymer_run": longest_homopolymer_run(template),
        }

    raise RuntimeError(
        "Failed to generate a valid template. Try relaxing template filters."
    )


def jsonl_dump(path: Path, records: Sequence[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True))
            handle.write("\n")


def summarize_records(records: Sequence[Dict[str, object]]) -> Dict[str, object]:
    split_counts = Counter(record["split"] for record in records)
    family_counts = Counter(record["family_id"] for record in records)
    lengths = [int(record["length"]) for record in records]
    thetas = [float(record["theta"]) for record in records]
    substitutions = [int(record["num_substitutions"]) for record in records]

    return {
        "num_sequences": len(records),
        "num_families": len(family_counts),
        "split_counts": dict(split_counts),
        "family_size_min": min(family_counts.values()) if family_counts else 0,
        "family_size_max": max(family_counts.values()) if family_counts else 0,
        "length_min": min(lengths) if lengths else 0,
        "length_max": max(lengths) if lengths else 0,
        "theta_min": min(thetas) if thetas else 0.0,
        "theta_max": max(thetas) if thetas else 0.0,
        "substitutions_min": min(substitutions) if substitutions else 0,
        "substitutions_max": max(substitutions) if substitutions else 0,
    }


def serialize_config(args: argparse.Namespace) -> Dict[str, object]:
    config = vars(args).copy()
    config["output_dir"] = str(args.output_dir)
    return config


def main() -> None:
    args = parse_args()
    validate_args(args)

    rng = random.Random(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    existing_templates_by_length: Dict[int, List[str]] = {}
    templates: List[Dict[str, object]] = []
    for family_id in range(args.num_families):
        templates.append(
            build_template(
                rng=rng,
                family_id=family_id,
                min_length=args.min_length,
                max_length=args.max_length,
                fixed_length=args.fixed_length,
                gc_min=args.gc_min,
                gc_max=args.gc_max,
                use_segmented_gc=args.use_segmented_gc,
                segment_count_min=args.segment_count_min,
                segment_count_max=args.segment_count_max,
                segment_min_length=args.segment_min_length,
                segment_gc_delta=args.segment_gc_delta,
                max_homopolymer_run=args.max_homopolymer_run,
                template_distance_ratio=args.template_distance_ratio,
                existing_templates_by_length=existing_templates_by_length,
            )
        )

    family_ids = [int(template["family_id"]) for template in templates]
    family_split_map = assign_family_splits(
        rng=rng,
        family_ids=family_ids,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
    )

    records: List[Dict[str, object]] = []
    seq_id = 0
    for template in templates:
        family_id = int(template["family_id"])
        template_sequence = str(template["template_sequence"])
        split = family_split_map[family_id]
        family_size = choose_family_size(
            rng, args.family_size_min, args.family_size_max
        )
        for family_index in range(family_size):
            theta = choose_theta(rng, args.theta_min, args.theta_max)
            sequence, mutations = mutate_sequence(rng, template_sequence, theta)
            records.append(
                {
                    "seq_id": seq_id,
                    "split": split,
                    "family_id": family_id,
                    "template_id": int(template["template_id"]),
                    "family_member_index": family_index,
                    "sequence": sequence,
                    "length": len(sequence),
                    "theta": theta,
                    "num_substitutions": len(mutations),
                    "mutations": mutations,
                    "template_sequence": template_sequence,
                    "target_gc_content": template["target_gc_content"],
                    "realized_template_gc_content": template["realized_gc_content"],
                    "gc_segments": template["gc_segments"],
                    "uses_segmented_gc": template["uses_segmented_gc"],
                }
            )
            seq_id += 1

    split_records = {
        split_name: [record for record in records if record["split"] == split_name]
        for split_name in ("train", "val", "test")
    }

    jsonl_dump(args.output_dir / "templates.jsonl", templates)
    jsonl_dump(args.output_dir / "all_sequences.jsonl", records)
    for split_name, split_data in split_records.items():
        jsonl_dump(args.output_dir / f"{split_name}.jsonl", split_data)

    metadata = {
        "generator": "generate_synthetic_dna_dataset.py",
        "seed": args.seed,
        "config": serialize_config(args),
        "template_summary": {
            "num_templates": len(templates),
            "length_min": min(int(template["length"]) for template in templates),
            "length_max": max(int(template["length"]) for template in templates),
            "gc_content_min": min(
                float(template["realized_gc_content"]) for template in templates
            ),
            "gc_content_max": max(
                float(template["realized_gc_content"]) for template in templates
            ),
        },
        "sequence_summary": summarize_records(records),
    }

    metadata_path = args.output_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"Wrote synthetic DNA dataset to {args.output_dir}")
    print(f"Templates: {len(templates)}")
    print(f"Sequences: {len(records)}")


if __name__ == "__main__":
    main()
