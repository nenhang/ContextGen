#!/usr/bin/env python3
"""
Run LayoutSAM ``sample-bench`` using an unpacked reference pack.

Pack should contain:
    - layoutsam_benchmark_filtered_sampled.json (or layoutsam_sampled.json)
    - reference_images/
    - reference_masks/
    - optional reference_images_1/ and reference_masks_1/
"""

from __future__ import annotations

import argparse
import json
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
for _p in (_SCRIPT_DIR, _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from layoutsam_config import get_layoutsam_paths, merge_layoutsam_cli

DEFAULT_PACK = _SCRIPT_DIR / "layoutsam_reference_pack"


def _resolve_benchmark_json(pack: Path, benchmark_json: Path | None) -> Path:
    if benchmark_json is not None:
        return benchmark_json.resolve()
    for cand in (
        pack / "layoutsam_benchmark_filtered_sampled.json",
        pack / "layoutsam_sampled.json",
    ):
        if cand.is_file():
            return cand
    raise SystemExit(
        "Provide --benchmark-json, or put layoutsam_benchmark_filtered_sampled.json/layoutsam_sampled.json in --pack."
    )


def _require_pack_dirs(pack: Path) -> None:
    required = [pack / "reference_images", pack / "reference_masks"]
    missing = [p for p in required if not p.is_dir()]
    if missing:
        raise SystemExit(f"Missing required directories in pack: {', '.join(str(p) for p in missing)}")


def _rewrite_paths_to_pack_abs(src_json: Path, pack: Path) -> Path:
    with open(src_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        for seg in item.get("segments", []):
            for vb in seg.get("valid_bbox", []):
                image_path = vb.get("image_path")
                mask_path = vb.get("mask_path")
                if isinstance(image_path, str) and image_path and not Path(image_path).is_absolute():
                    vb["image_path"] = str((pack / image_path).resolve())
                if isinstance(mask_path, str) and mask_path and not Path(mask_path).is_absolute():
                    vb["mask_path"] = str((pack / mask_path).resolve())

    out_json = pack / "_layoutsam_benchmark_filtered_sampled_abs.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return out_json


def main() -> None:
    p = argparse.ArgumentParser(description="LayoutSAM sample-bench using a packaged reference directory")
    p.add_argument(
        "--pack",
        type=Path,
        default=DEFAULT_PACK,
        help="Unpacked reference pack root (default: ./layoutsam_reference_pack beside this script).",
    )
    p.add_argument(
        "--benchmark-json",
        type=Path,
        default=None,
        help="Override benchmark JSON path; default auto-detects one inside --pack.",
    )
    p.add_argument(
        "--generated-dir",
        type=Path,
        default=None,
        help="Override output root (default: generated_images_dir from layoutsam_config).",
    )
    p.add_argument("--image-width", type=int, default=768)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--resize-pattern", type=str, default="instance-fit")
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Before sampling, scan benchmark and skip indices whose images/<index>.jpg already exists.",
    )
    p.add_argument("--generation-repo", type=Path, default=None)
    p.add_argument(
        "--kontext-model-path",
        type=str,
        default=os.environ.get("KONTEXT_MODEL_PATH", ""),
        help="Kontext base model path or HF id (default: env KONTEXT_MODEL_PATH).",
    )
    p.add_argument(
        "--adapter-path",
        type=str,
        default=os.environ.get("ADAPTER_PATH", ""),
        help="LoRA adapter directory (default: env ADAPTER_PATH).",
    )
    args = p.parse_args()

    pack = args.pack.resolve()
    if not pack.is_dir():
        raise SystemExit(f"Not a directory: {pack}")
    _require_pack_dirs(pack)

    bench_json = _resolve_benchmark_json(pack, args.benchmark_json)
    bench_json_abs = _rewrite_paths_to_pack_abs(bench_json, pack)

    base = get_layoutsam_paths()
    overrides: dict = {}
    if args.generated_dir is not None:
        overrides["generated_images_dir"] = args.generated_dir
    if args.generation_repo is not None:
        overrides["generation_repo"] = args.generation_repo
    paths = merge_layoutsam_cli(base, **overrides)

    kontext_model_path = (args.kontext_model_path or "").strip()
    adapter_path = (args.adapter_path or "").strip()
    if not kontext_model_path or not adapter_path:
        raise SystemExit(
            "Need non-empty --kontext-model-path and --adapter-path (or env KONTEXT_MODEL_PATH / ADAPTER_PATH)."
        )

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    from pipeline.sample_bench import run_sample_bench_layoutsam

    run_sample_bench_layoutsam(
        paths,
        benchmark_json=bench_json_abs,
        image_width=args.image_width,
        batch_size=args.batch_size,
        resize_pattern=args.resize_pattern,
        kontext_model_path=kontext_model_path,
        adapter_path=adapter_path,
        skip_existing=args.skip_existing,
    )


if __name__ == "__main__":
    main()
