"""Multi-GPU full-scene sampling for LayoutSAM JSON via ``src.model.generate`` (Kontext + LoRA)."""

from __future__ import annotations

import json
import multiprocessing as mp
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[4]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import torch
from layoutsam_config import LayoutsamPaths
from tqdm import tqdm


def _reference_tuples_to_dicts(reference_info: list[list[tuple]]) -> list[list[dict]]:
    return [[{"image": a, "bbox": b, "mask": c} for (a, b, c) in row] for row in reference_info]


def _build_reference_info_for_batch(
    batch: list,
    paths: LayoutsamPaths,
    image_width: int,
    device: str | torch.device,
) -> list[list[tuple]]:
    reference_info = []
    for item in batch:
        segment_info = []
        for segment in item["segments"]:
            bbox = torch.tensor(segment["bbox"], device=device) * image_width
            segment_image_path = paths.normalize_asset_path(segment["valid_bbox"][0]["image_path"])
            segment_mask_path = paths.normalize_asset_path(segment["valid_bbox"][0]["mask_path"])
            segment_info.append((segment_image_path, bbox, segment_mask_path))
        reference_info.append(segment_info)
    return reference_info


def _generate_bench_images_worker(
    flux_pipe,
    items: list,
    paths: LayoutsamPaths,
    output_dir: Path | None,
    output_ref_dir: Path | None,
    resize_pattern: str,
    image_width: int,
    batch_size: int,
    device: str,
) -> None:
    from src.model.generate import generate as model_generate  # type: ignore  # noqa: WPS433
    from src.utils.image_process import annotate  # type: ignore  # noqa: WPS433

    dev = torch.device(device) if isinstance(device, str) else device

    for i in tqdm(range(0, len(items), batch_size), desc=f"sample {device}"):
        batch = items[i : i + batch_size]
        item_index = [item["index"] for item in batch]
        captions = [item["global_caption"] for item in batch]
        reference_info = _build_reference_info_for_batch(batch, paths, image_width, device)
        ref_dicts = _reference_tuples_to_dicts(reference_info)
        seeds = [42] * len(batch)
        with torch.inference_mode():
            imgs = model_generate(
                flux_pipe=flux_pipe,
                prompts=captions,
                reference_info=ref_dicts,
                height=image_width,
                width=image_width,
                device=dev,
                seed=seeds,
                resize_pattern=resize_pattern,
                layout_image=None,
            )

        for j, img in enumerate(imgs):
            if output_dir is not None:
                img.save(output_dir / f"{item_index[j]:04d}.jpg")
            if output_ref_dir is not None:
                phrases_for_annotation = []
                bboxes_for_annotation = []
                for k in range(len(batch[j]["segments"])):
                    phrases_for_annotation.append(batch[j]["segments"][k]["caption"])
                    bboxes_for_annotation.append(
                        (torch.tensor(batch[j]["segments"][k]["bbox"], device=device) * image_width)
                        .round()
                        .to(torch.int32)
                    )
                annotated_img = annotate(img, bboxes_for_annotation, phrases_for_annotation)
                annotated_img.save(output_ref_dir / f"{item_index[j]:04d}_annotated.jpg")


def _processor(
    chunk: list,
    paths: LayoutsamPaths,
    output_dir: Path,
    output_ref_dir: Path,
    image_width: int,
    batch_size: int,
    gpu_id: int,
    resize_pattern: str,
    kontext_model_path: str,
    adapter_path: str,
) -> None:
    paths.ensure_generation_repo_on_syspath()
    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    from src.model.generate import load_model  # type: ignore  # noqa: WPS433

    print(f"CUDA {gpu_id}: load_model Kontext + LoRA")
    flux_pipe = load_model(kontext_model_path, adapter_path, device=device)
    flux_pipe.set_progress_bar_config(disable=True)
    _generate_bench_images_worker(
        flux_pipe,
        chunk,
        paths,
        output_dir,
        output_ref_dir,
        resize_pattern,
        image_width,
        batch_size,
        device,
    )


def run_sample_bench_layoutsam(
    paths: LayoutsamPaths,
    *,
    benchmark_json: Path,
    image_width: int,
    batch_size: int,
    resize_pattern: str,
    kontext_model_path: str,
    adapter_path: str,
    skip_existing: bool = False,
) -> None:
    with open(benchmark_json, "r") as f:
        benchmark = json.load(f)

    save_path = paths.generated_images_dir / "images"
    save_ref_path = paths.generated_images_dir / "annotated_images"
    save_path.mkdir(parents=True, exist_ok=True)
    save_ref_path.mkdir(parents=True, exist_ok=True)

    to_run = benchmark
    if skip_existing:
        n_total = len(benchmark)
        to_run = [
            item
            for item in benchmark
            if not (save_path / f"{int(item['index']):04d}.jpg").is_file()
        ]
        n_skip = n_total - len(to_run)
        print(
            f"[skip-existing] benchmark items: {n_total}, already have images: {n_skip}, to generate: {len(to_run)}"
        )
        if not to_run:
            print("[skip-existing] nothing to generate, exiting.")
            return

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPU available.")
    chunk_size = len(to_run) // num_gpus + 1
    chunks = [to_run[i : i + chunk_size] for i in range(0, len(to_run), chunk_size)]

    processes = []
    for gpu_id, chunk in enumerate(chunks):
        if not chunk:
            continue
        p = mp.Process(
            target=_processor,
            args=(
                chunk,
                paths,
                save_path,
                save_ref_path,
                image_width,
                batch_size,
                gpu_id,
                resize_pattern,
                kontext_model_path,
                adapter_path,
            ),
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    print(f"Outputs under {paths.generated_images_dir}")
