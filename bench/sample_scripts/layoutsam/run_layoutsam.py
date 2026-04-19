#!/usr/bin/env python3
"""
Unified CLI for LayoutSAM prep + sampling (paths in ``layoutsam_config.py``).

Run from this directory::

    python run_layoutsam.py --help
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import fields
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[3]
for _p in (_SCRIPT_DIR, _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from layoutsam_config import LayoutsamPaths, get_layoutsam_paths, merge_layoutsam_cli


def _paths_from_args(args: argparse.Namespace) -> LayoutsamPaths:
    base = get_layoutsam_paths(getattr(args, "data_dir", None))
    kwargs: dict = {}
    for f in fields(LayoutsamPaths):
        name = f.name
        if hasattr(args, name):
            v = getattr(args, name)
            if v is not None:
                kwargs[name] = v
    return merge_layoutsam_cli(base, **kwargs)


def _add_common_data(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Override DATA_DIR (rebase layoutsam_benchmark.json, segments/, …).",
    )


def _add_model_overrides(p: argparse.ArgumentParser) -> None:
    p.add_argument("--flux-model-path", type=str, dest="flux_model_path", default=None)
    p.add_argument("--flux-turbo-lora-path", type=str, dest="flux_turbo_lora_path", default=None)
    p.add_argument(
        "--grounding-dino-model",
        type=str,
        dest="grounding_dino_model",
        default=None,
        help="HF id or local dir (transformers Grounding DINO). Default: layoutsam_config / GROUNDING_DINO_MODEL.",
    )
    p.add_argument("--generation-repo", type=Path, dest="generation_repo", default=None)
    p.add_argument("--kontext-model-path", type=str, dest="kontext_model_path", default=None)
    p.add_argument("--adapter-path", type=str, dest="adapter_path", default=None)


def main() -> None:
    parser = argparse.ArgumentParser(description="LayoutSAM benchmark pipeline")
    _add_common_data(parser)
    _add_model_overrides(parser)
    sub = parser.add_subparsers(dest="task", required=True)

    p_seg = sub.add_parser(
        "gen-segments", help="Flux(+Turbo): segment JPGs into segments_dir from layoutsam_benchmark.json"
    )
    p_seg.add_argument("--batch-size", type=int, default=8)
    p_seg.add_argument("--num-gpus", type=int, default=None, help="Default: visible CUDA device count")
    p_seg.add_argument("--no-turbo", action="store_true", help="Disable Turbo LoRA (more steps)")
    p_seg.set_defaults(handler=_cmd_gen_segments)

    p_bbox = sub.add_parser("valid-bbox", help="Grounding DINO: write layoutsam_benchmark_with_bbox.json")
    p_bbox.add_argument(
        "--num-processes",
        type=int,
        default=None,
        help="Parallel worker count (default: one per GPU)",
    )
    p_bbox.set_defaults(handler=_cmd_valid_bbox)

    p_filt = sub.add_parser("filter-benchmark", help="Mask-quality filter → layoutsam_benchmark_filtered.json")
    p_filt.set_defaults(handler=_cmd_filter)

    p_samp = sub.add_parser("sample-bench", help="Full-scene sampling via src.model.generate + disk LoRA (multi-GPU)")
    p_samp.add_argument(
        "--benchmark-json",
        type=Path,
        default=None,
        help="Default: benchmark_filtered_json under data-dir",
    )
    p_samp.add_argument("--image-width", type=int, default=768)
    p_samp.add_argument("--batch-size", type=int, default=4)
    p_samp.add_argument("--resize-pattern", type=str, default="instance-fit")
    p_samp.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip items when images/<index>.jpg already exists (scan once before sampling).",
    )
    p_samp.set_defaults(handler=_cmd_sample_bench)

    args = parser.parse_args()
    paths = _paths_from_args(args)
    args.handler(paths, args)


def _cmd_gen_segments(paths: LayoutsamPaths, args: argparse.Namespace) -> None:
    from pipeline.reference import run_generate_segments

    run_generate_segments(
        paths,
        batch_size=args.batch_size,
        num_gpus=args.num_gpus,
        use_turbo=not args.no_turbo,
    )


def _cmd_valid_bbox(paths: LayoutsamPaths, args: argparse.Namespace) -> None:
    from pipeline.reference import run_valid_bbox

    run_valid_bbox(paths, num_processes=args.num_processes)


def _cmd_filter(paths: LayoutsamPaths, args: argparse.Namespace) -> None:
    from pipeline.reference import run_filter_benchmark

    run_filter_benchmark(paths)


def _cmd_sample_bench(paths: LayoutsamPaths, args: argparse.Namespace) -> None:
    from pipeline.sample_bench import run_sample_bench_layoutsam

    bench = args.benchmark_json if args.benchmark_json is not None else paths.benchmark_filtered_json
    km = (args.kontext_model_path or os.environ.get("KONTEXT_MODEL_PATH") or paths.kontext_model_path or "").strip()
    ad = (args.adapter_path or os.environ.get("ADAPTER_PATH") or paths.adapter_path or "").strip()
    if not km or not ad:
        raise SystemExit("sample-bench needs --kontext-model-path and --adapter-path (or env / layoutsam_config).")

    run_sample_bench_layoutsam(
        paths,
        benchmark_json=Path(bench),
        image_width=args.image_width,
        batch_size=args.batch_size,
        resize_pattern=args.resize_pattern,
        kontext_model_path=km,
        adapter_path=ad,
        skip_existing=args.skip_existing,
    )


if __name__ == "__main__":
    try:
        mp = __import__("multiprocessing")
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
