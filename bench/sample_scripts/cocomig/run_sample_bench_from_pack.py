#!/usr/bin/env python3
"""
Run ``sample-bench`` using an unpacked **reference pack** (segment JPGs + masks + category JSON).

The pack must contain::

    bench_instance_categories_filtered.json
    reference_images/
    reference_masks/
    mig_bench.json   (recommended; otherwise set --benchmark-json)

``cocomig_config.py`` supplies ``GENERATION_REPO`` (repo root on ``PYTHONPATH`` for ``src.*``) and
``generated_images_dir``. The model is always loaded like ``inference.py``: Kontext base +
LoRA from ``--kontext-model-path`` and ``--adapter-path`` (defaults: env ``KONTEXT_MODEL_PATH``,
``ADAPTER_PATH``).

Example::

    python run_sample_bench_from_pack.py --pack /path/to/unpacked_pack
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_env_file = _REPO_ROOT / ".env"
if _env_file.is_file():
    from dotenv import load_dotenv

    load_dotenv(_env_file)

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from cocomig_config import get_paths, merge_cli_overrides
from pipeline.sample_bench import run_sample_bench

DEFAULT_PACK = _SCRIPT_DIR / "cocomig_reference_pack"


def main() -> None:
    p = argparse.ArgumentParser(description="sample-bench using a packaged reference directory")
    p.add_argument(
        "--pack",
        type=Path,
        default=DEFAULT_PACK,
        help="Root of the unpacked reference pack (default: ./cocomig_reference_pack beside this script).",
    )
    p.add_argument("--benchmark-json", type=Path, default=None, help="Override mig_bench.json if not inside pack")
    p.add_argument(
        "--generated-dir",
        type=Path,
        default=None,
        help="Override output root (default: cocomig_config GENERATED_IMAGES_DIR)",
    )
    p.add_argument("--image-width", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument(
        "--resize-pattern",
        type=str,
        default="instance-fit-distort",
        help="Instance placement: instance-fit | instance-fit-distort | image-fit (passed to src.model.generate).",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Before sampling, scan benchmark and skip indices whose images/<index>.jpg already exists.",
    )
    p.add_argument("--generation-repo", type=Path, default=None)
    p.add_argument(
        "--kontext-model-path",
        type=str,
        default=os.environ.get("KONTEXT_MODEL_PATH", "black-forest-labs/FLUX.1-Kontext-dev"),
        help="Kontext base (HF id or local folder). Default: env KONTEXT_MODEL_PATH or FLUX.1-Kontext-dev.",
    )
    p.add_argument(
        "--adapter-path",
        type=str,
        default=os.environ.get("ADAPTER_PATH", "/root/autodl-tmp/ckpt/ContextGen"),
        help="Directory containing LoRA .safetensors (same as inference.py). Default: env ADAPTER_PATH.",
    )
    args = p.parse_args()

    pack = args.pack.resolve()
    if not pack.is_dir():
        raise SystemExit(f"Not a directory: {pack}")

    filtered = pack / "bench_instance_categories_filtered.json"
    if not filtered.is_file():
        raise SystemExit(f"Missing {filtered}")

    bench_in_pack = pack / "mig_bench.json"
    bench_path = args.benchmark_json
    if bench_path is None:
        bench_path = bench_in_pack if bench_in_pack.is_file() else None
    if bench_path is None:
        raise SystemExit("Provide mig_bench.json inside the pack or pass --benchmark-json")

    base = get_paths()
    kwargs = {
        "reference_images_dir": pack / "reference_images",
        "mask_dir": pack / "reference_masks",
        "instance_categories_filtered_json": filtered,
        "benchmark_json": Path(bench_path),
    }
    if args.generated_dir is not None:
        kwargs["generated_images_dir"] = args.generated_dir
    if args.generation_repo is not None:
        kwargs["generation_repo"] = args.generation_repo

    paths = merge_cli_overrides(base, **kwargs)

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    km = args.kontext_model_path.strip() or None
    ad = args.adapter_path.strip() or None
    if not km or not ad:
        raise SystemExit(
            "Need non-empty --kontext-model-path and --adapter-path (or set KONTEXT_MODEL_PATH / ADAPTER_PATH)."
        )

    run_sample_bench(
        paths,
        image_width=args.image_width,
        batch_size=args.batch_size,
        resize_pattern=args.resize_pattern,
        kontext_model_path=km,
        adapter_path=ad,
        skip_existing=args.skip_existing,
    )


if __name__ == "__main__":
    main()
