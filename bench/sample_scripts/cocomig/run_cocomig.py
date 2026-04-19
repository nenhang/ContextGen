#!/usr/bin/env python3
"""
Unified CLI for COCO-MIG sample pipeline tasks.

Run from this directory (or ensure ``bench/sample_scripts/cocomig`` is on ``PYTHONPATH``):

    python run_cocomig.py --help
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import fields
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from cocomig_config import CocomigPaths, get_paths, merge_cli_overrides


def _paths_from_args(args: argparse.Namespace) -> CocomigPaths:
    base = get_paths(getattr(args, "data_dir", None))
    kwargs: dict = {}
    for f in fields(CocomigPaths):
        name = f.name
        if hasattr(args, name):
            v = getattr(args, name)
            if v is not None:
                kwargs[name] = v
    return merge_cli_overrides(base, **kwargs)


def _add_path_overrides(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Override DATA_DIR in cocomig_config.py (rebases benchmark/mask paths).",
    )
    p.add_argument("--benchmark-json", type=Path, dest="benchmark_json", default=None)
    p.add_argument("--instance-categories", type=Path, dest="instance_categories_json", default=None)
    p.add_argument(
        "--instance-categories-filtered",
        type=Path,
        dest="instance_categories_filtered_json",
        default=None,
    )
    p.add_argument("--segments-dir", type=Path, dest="segments_dir", default=None)
    p.add_argument("--reference-images-dir", type=Path, dest="reference_images_dir", default=None)
    p.add_argument("--mask-dir", type=Path, dest="mask_dir", default=None)
    p.add_argument("--mask-dir-full", type=Path, dest="mask_dir_full", default=None)
    p.add_argument("--generated-dir", type=Path, dest="generated_images_dir", default=None)
    p.add_argument("--flux-path", type=Path, dest="flux_model_path", default=None)
    p.add_argument(
        "--grounding-dino-model",
        type=str,
        dest="grounding_dino_model",
        default=None,
        help="Override GROUNDING_DINO_MODEL in cocomig_config.py.",
    )
    p.add_argument(
        "--generation-repo",
        type=Path,
        dest="generation_repo",
        default=None,
        help="Override GENERATION_REPO in cocomig_config.py (project with src/).",
    )
    p.add_argument(
        "--train-config-yaml",
        type=Path,
        dest="train_config_yaml",
        default=None,
        help="Override TRAIN_CONFIG_YAML in cocomig_config.py.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="COCO-MIG benchmark sample scripts")
    _add_path_overrides(parser)
    sub = parser.add_subparsers(dest="task", required=True)

    p_cat = sub.add_parser("build-categories", help="Aggregate segment labels into bench_instance_categories.json")
    p_cat.set_defaults(handler=_cmd_build_categories)

    p_ref = sub.add_parser("gen-reference", help="Flux: generate per-category reference images into segments-dir")
    p_ref.add_argument("--target-size", type=int, nargs=2, default=[512, 512], metavar=("W", "H"))
    p_ref.add_argument("--batch-size", type=int, default=4)
    p_ref.add_argument("--num-samples", type=int, default=32)
    p_ref.set_defaults(handler=_cmd_gen_reference)

    p_crop = sub.add_parser(
        "crop-reference",
        help="transformers Grounding DINO: crop segments-dir -> reference-images-dir",
    )
    p_crop.add_argument(
        "--num-processes",
        type=int,
        default=None,
        help="Defaults to number of visible CUDA devices.",
    )
    p_crop.set_defaults(handler=_cmd_crop)

    p_f = sub.add_parser("filter", help="Mask coverage filter + sync mask files")
    p_f.add_argument(
        "filter_mode",
        choices=("mask", "sync"),
        help="mask: coverage on instance_categories JSON and write filtered JSON | "
        "sync: hardlink from mask-dir-full to mask-dir",
    )
    p_f.set_defaults(handler=_cmd_filter)

    p_gen = sub.add_parser("sample-bench", help="Full-scene sampling via src.model.generate + disk LoRA (multi-GPU)")
    p_gen.add_argument("--image-width", type=int, default=512)
    p_gen.add_argument("--batch-size", type=int, default=4)
    p_gen.add_argument(
        "--resize-pattern",
        type=str,
        default="instance-fit-distort",
        help="Instance placement resize strategy (instance-fit | instance-fit-distort | image-fit).",
    )
    p_gen.add_argument(
        "--kontext-model-path",
        type=str,
        default=os.environ.get("KONTEXT_MODEL_PATH", "black-forest-labs/FLUX.1-Kontext-dev"),
        help="Kontext base id or folder. Default: env KONTEXT_MODEL_PATH or FLUX.1-Kontext-dev.",
    )
    p_gen.add_argument(
        "--adapter-path",
        type=str,
        default=os.environ.get("ADAPTER_PATH", "/root/autodl-tmp/ckpt/ContextGen"),
        help="LoRA .safetensors directory (same as inference.py). Default: env ADAPTER_PATH.",
    )
    p_gen.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip items when images/<index>.jpg already exists (scan once before sampling).",
    )
    p_gen.set_defaults(handler=_cmd_sample_bench)

    args = parser.parse_args()
    paths = _paths_from_args(args)
    args.handler(paths, args)


def _cmd_build_categories(paths: CocomigPaths, args: argparse.Namespace) -> None:
    from pipeline.reference import run_build_instance_categories

    run_build_instance_categories(paths)


def _cmd_gen_reference(paths: CocomigPaths, args: argparse.Namespace) -> None:
    from pipeline.reference import run_generate_reference_flux

    w, h = args.target_size
    run_generate_reference_flux(paths, target_size=(w, h), batch_size=args.batch_size, num_samples=args.num_samples)


def _cmd_crop(paths: CocomigPaths, args: argparse.Namespace) -> None:
    from pipeline.reference import run_crop_references

    run_crop_references(paths, num_processes=args.num_processes)


def _cmd_filter(paths: CocomigPaths, args: argparse.Namespace) -> None:
    from pipeline.reference import filter_with_mask, remove_invalid_images

    mode = args.filter_mode
    if mode == "mask":
        filter_with_mask(paths)
    elif mode == "sync":
        remove_invalid_images(paths)
    else:
        raise AssertionError(mode)


def _cmd_sample_bench(paths: CocomigPaths, args: argparse.Namespace) -> None:
    from pipeline.sample_bench import run_sample_bench

    try:
        import multiprocessing as mp

        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    km = args.kontext_model_path.strip() or None
    ad = args.adapter_path.strip() or None
    if not km or not ad:
        raise SystemExit("sample-bench needs --kontext-model-path and --adapter-path (or KONTEXT_MODEL_PATH / ADAPTER_PATH).")
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
