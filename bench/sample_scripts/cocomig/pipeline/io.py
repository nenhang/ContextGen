"""Load benchmark JSON in the two shapes used across the COCO-MIG scripts."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


def resolve_segment_mask_path(
    mask_dir: str | Path,
    cat_idx: int,
    sample_idx: int,
    caption_underscore: str,
) -> Path:
    """
    Return ``*_mask.png`` (packed 1-bit) or ``*_mask.jpg`` if present under ``mask_dir``.

    Raises ``FileNotFoundError`` if neither exists.
    """
    d = Path(mask_dir)
    base = f"{cat_idx:04d}_{sample_idx:02d}_{caption_underscore}_mask"
    for ext in (".png", ".jpg"):
        p = d / f"{base}{ext}"
        if p.is_file():
            return p
    raise FileNotFoundError(f"No mask file {base}.png or {base}.jpg under {d}")


def masked_overlay_path_for_mask(mask_path: str | Path) -> Path:
    """Peer ``*_masked.jpg`` for a ``*_mask.png`` or ``*_mask.jpg`` path."""
    p = Path(mask_path)
    stem = p.stem
    if not stem.endswith("_mask"):
        raise ValueError(f"Expected mask filename stem ending with _mask, got {p.name}")
    return p.with_name(f"{stem.removesuffix('_mask')}_masked.jpg")


def read_benchmark_segments(
    benchmark_path: str | Path,
    num_samples: int | None = None,
    sample_method: str = "random",
) -> list[tuple[Any, Any, int, int]]:
    """
    Flatten all segments as (label, bbox, image_index, segment_index).
    Sampling uses torch.randperm when ``sample_method`` is ``random`` (legacy behavior).
    """
    benchmark_path = Path(benchmark_path)
    with open(benchmark_path, "r") as f:
        benchmark = json.load(f)

    assert sample_method in ("random", "sequential"), "sample_method must be 'random' or 'sequential'."
    if num_samples is not None:
        import torch

        assert num_samples <= len(benchmark), "num_samples must be <= len(benchmark)."
        if sample_method == "random":
            indices = torch.randperm(len(benchmark))[:num_samples]
        else:
            indices = range(num_samples)
        if isinstance(benchmark, dict):
            benchmark = {str(i): benchmark[str(i)] for i in indices}
        elif isinstance(benchmark, list):
            benchmark = [benchmark[i] for i in indices]

    segment_items: list[tuple[Any, Any, int, int]] = []
    for i in range(len(benchmark)):
        item = benchmark[str(i)]
        for j, segment in enumerate(item["segment"]):
            segment_items.append((segment["label"], segment["bbox"], i, j))

    return segment_items


def read_benchmark_items(
    benchmark_path: str | Path,
    num_samples: int | None = None,
    sample_method: str = "random",
) -> list[dict]:
    """
    One dict per benchmark image, with original index preserved in ``item['index']``.
    Uses ``random.sample`` for random sampling (legacy behavior).
    """
    benchmark_path = Path(benchmark_path)
    with open(benchmark_path, "r") as f:
        benchmark = json.load(f)

    assert sample_method in ("random", "sequential"), "sample_method must be 'random' or 'sequential'."
    if num_samples is not None:
        assert num_samples <= len(benchmark), "num_samples must be <= len(benchmark)."
        if sample_method == "random":
            indices = random.sample(range(len(benchmark)), num_samples)
        else:
            indices = range(num_samples)
        if isinstance(benchmark, dict):
            benchmark = [benchmark[str(i)] for i in indices]
        elif isinstance(benchmark, list):
            benchmark = [benchmark[i] for i in indices]
    else:
        indices = list(range(len(benchmark)))
        if isinstance(benchmark, dict):
            benchmark = list(benchmark.values())
        elif isinstance(benchmark, list):
            benchmark = benchmark

    segment_items: list[dict] = []
    for i in range(len(benchmark)):
        item = benchmark[i]
        item["index"] = indices[i]
        segment_items.append(item)

    return segment_items
