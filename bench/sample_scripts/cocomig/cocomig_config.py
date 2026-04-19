"""
COCO-MIG path configuration.

Edit the **USER CONFIG** block at the top, then use ``get_paths()`` / ``CocomigPaths``.
CLI flags (see ``run_cocomig.py``) override individual fields without editing this file.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

# =============================================================================
# USER CONFIG — edit below
# =============================================================================

# ContextGen repo root (parent of ``bench/``)
REPO_ROOT = Path(__file__).resolve().parents[3]

# Benchmark assets: JSON, segments, masks, generated images
DATA_DIR = REPO_ROOT / "bench" / "cocomig_data"

# Flux weights for ``gen-reference`` (Hugging Face cache dir or local folder)
FLUX_MODEL_PATH = "black-forest-labs/FLUX.1-dev"

# Hugging Face id or local folder for ``crop-reference`` (transformers Grounding DINO)
GROUNDING_DINO_MODEL = "IDEA-Research/grounding-dino-base"

# Project root that contains ``src/`` (``sample-bench`` imports ``src.*`` from here)
GENERATION_REPO = REPO_ROOT

# Optional training YAML (not used by ``sample-bench``; kept for other tooling / overrides)
TRAIN_CONFIG_YAML = GENERATION_REPO / "train" / "config" / "mig_1024.yaml"

# =============================================================================
# Implementation
# =============================================================================


def data_paths_under(root: Path) -> dict[str, Path]:
    """Standard filenames under a data root (used by ``--data-dir``)."""
    r = root.resolve()
    return {
        "data_dir": r,
        "benchmark_json": r / "mig_bench.json",
        "instance_categories_json": r / "bench_instance_categories.json",
        "instance_categories_filtered_json": r / "bench_instance_categories_filtered.json",
        "segments_dir": r / "mig_segments",
        "reference_images_dir": r / "reference_images",
        "mask_dir": r / "reference_masks",
        "mask_dir_full": r / "reference_masks_full",
        "generated_images_dir": REPO_ROOT / "bench" / "output" / "cocomig",
    }


@dataclass
class CocomigPaths:
    """Resolved paths for the COCO-MIG pipeline."""

    repo_root: Path
    data_dir: Path
    benchmark_json: Path
    instance_categories_json: Path
    instance_categories_filtered_json: Path
    segments_dir: Path
    reference_images_dir: Path
    mask_dir: Path
    mask_dir_full: Path
    generated_images_dir: Path
    flux_model_path: Path
    grounding_dino_model: str
    generation_repo: Path
    train_config_yaml: Path

    def ensure_generation_repo_on_syspath(self) -> None:
        """Insert ``generation_repo`` so ``src.*`` imports work for full-scene generation."""
        import sys

        root = Path(self.generation_repo)
        if not (root / "src").is_dir():
            raise RuntimeError(
                f"Expected a Python project with a src/ directory; got generation_repo={root!s}. "
                "Fix GENERATION_REPO in cocomig_config.py (USER CONFIG) or pass --generation-repo."
            )
        rs = str(root)
        if rs not in sys.path:
            sys.path.insert(0, rs)


def get_paths(data_dir: str | Path | None = None) -> CocomigPaths:
    """Build paths from USER CONFIG; if ``data_dir`` is set, only rebases benchmark asset paths."""
    data = data_paths_under(Path(data_dir)) if data_dir is not None else data_paths_under(DATA_DIR)
    return CocomigPaths(
        repo_root=Path(REPO_ROOT),
        **data,
        flux_model_path=Path(FLUX_MODEL_PATH),
        grounding_dino_model=str(GROUNDING_DINO_MODEL),
        generation_repo=Path(GENERATION_REPO),
        train_config_yaml=Path(TRAIN_CONFIG_YAML),
    )


_STR_FIELDS = frozenset({"grounding_dino_model"})


def merge_cli_overrides(paths: CocomigPaths, **kwargs) -> CocomigPaths:
    names = {f.name for f in fields(CocomigPaths)}
    d = {f.name: getattr(paths, f.name) for f in fields(CocomigPaths)}
    for k, v in kwargs.items():
        if v is None or k not in names:
            continue
        if k in _STR_FIELDS:
            d[k] = str(v)
        else:
            d[k] = Path(v) if not isinstance(v, Path) else v
    return CocomigPaths(**d)
