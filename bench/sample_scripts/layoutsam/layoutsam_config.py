"""
LayoutSAM benchmark path configuration (aligned with ``cocomig`` style).

Edit the **USER CONFIG** block, then ``get_layoutsam_paths()`` / ``LayoutsamPaths``.
CLI ``--data-dir`` rebases all paths under ``DATA_DIR``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path

# =============================================================================
# USER CONFIG
# =============================================================================

REPO_ROOT = Path(__file__).resolve().parents[3]

# Root folder for LayoutSAM JSON + segments + masks + outputs (mirror your disk layout here)
DATA_DIR = REPO_ROOT / "bench" / "layoutsam_data"

# Flux + Turbo LoRA for ``gen-segments`` (HF id or local dir)
FLUX_MODEL_PATH = os.environ.get("FLUX_MODEL_PATH", "black-forest-labs/FLUX.1-dev")
FLUX_TURBO_LORA_PATH = os.environ.get("FLUX_TURBO_LORA_PATH", "alimama-creative/FLUX.1-Turbo-Alpha")

# Hugging Face id or folder for ``valid-bbox`` (transformers Grounding DINO, same as cocomig ``crop-reference``)
GROUNDING_DINO_MODEL = os.environ.get("GROUNDING_DINO_MODEL", "IDEA-Research/grounding-dino-base")

# Repo whose ``src/`` is used for ``sample-bench`` (ContextGen root)
GENERATION_REPO = REPO_ROOT

# Kontext + LoRA (same as ``inference.py`` / cocomig ``sample-bench``)
KONTEXT_MODEL_PATH = os.environ.get("KONTEXT_MODEL_PATH", "black-forest-labs/FLUX.1-Kontext-dev")
ADAPTER_PATH = os.environ.get("ADAPTER_PATH", "/root/autodl-tmp/ckpt/ContextGen")

# =============================================================================
# Implementation
# =============================================================================


def layoutsam_paths_under(root: Path) -> dict[str, Path]:
    r = root.resolve()
    return {
        "data_dir": r,
        "layoutsam_benchmark_json": r / "layoutsam_benchmark.json",
        "segments_dir": r / "segments",
        "benchmark_with_bbox_json": r / "layoutsam_benchmark_with_bbox.json",
        "benchmark_filtered_json": r / "layoutsam_benchmark_filtered.json",
        "generated_images_dir": REPO_ROOT / "bench" / "output" / "layoutsam",
    }


@dataclass
class LayoutsamPaths:
    repo_root: Path
    data_dir: Path
    layoutsam_benchmark_json: Path
    segments_dir: Path
    benchmark_with_bbox_json: Path
    benchmark_filtered_json: Path
    generated_images_dir: Path
    flux_model_path: str
    flux_turbo_lora_path: str
    grounding_dino_model: str
    generation_repo: Path
    kontext_model_path: str
    adapter_path: str

    def ensure_generation_repo_on_syspath(self) -> None:
        import sys

        root = Path(self.generation_repo)
        if not (root / "src").is_dir():
            raise RuntimeError(
                f"Expected project root with src/; got generation_repo={root!s}. Set GENERATION_REPO in layoutsam_config.py."
            )
        rs = str(root)
        if rs not in sys.path:
            sys.path.insert(0, rs)

    def normalize_asset_path(self, p: str | Path) -> str:
        """Return resolved path if ``p`` points to an existing file; otherwise the string form of ``p``."""
        pth = Path(p)
        if pth.is_file():
            return str(pth.resolve())
        return str(p)


def get_layoutsam_paths(data_dir: str | Path | None = None) -> LayoutsamPaths:
    base = Path(data_dir).resolve() if data_dir is not None else DATA_DIR.resolve()
    d = layoutsam_paths_under(base)
    return LayoutsamPaths(
        repo_root=REPO_ROOT,
        data_dir=d["data_dir"],
        layoutsam_benchmark_json=d["layoutsam_benchmark_json"],
        segments_dir=d["segments_dir"],
        benchmark_with_bbox_json=d["benchmark_with_bbox_json"],
        benchmark_filtered_json=d["benchmark_filtered_json"],
        generated_images_dir=d["generated_images_dir"],
        flux_model_path=str(FLUX_MODEL_PATH),
        flux_turbo_lora_path=str(FLUX_TURBO_LORA_PATH),
        grounding_dino_model=str(GROUNDING_DINO_MODEL),
        generation_repo=Path(GENERATION_REPO),
        kontext_model_path=str(KONTEXT_MODEL_PATH),
        adapter_path=str(ADAPTER_PATH),
    )


_STR_FIELDS = frozenset(
    {
        "flux_model_path",
        "flux_turbo_lora_path",
        "grounding_dino_model",
        "kontext_model_path",
        "adapter_path",
    }
)


def merge_layoutsam_cli(paths: LayoutsamPaths, **kwargs) -> LayoutsamPaths:
    names = {f.name for f in fields(LayoutsamPaths)}
    d = {f.name: getattr(paths, f.name) for f in fields(LayoutsamPaths)}
    for k, v in kwargs.items():
        if v is None or k not in names:
            continue
        if k in _STR_FIELDS:
            d[k] = str(v)
        else:
            d[k] = Path(v) if not isinstance(v, Path) else v
    return LayoutsamPaths(**d)
