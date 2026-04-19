"""Full-scene benchmark sampling via ``src.model.generate`` (Kontext + LoRA on disk)."""

from __future__ import annotations

import sys
from pathlib import Path as _Path

_COCOMIG_ROOT = _Path(__file__).resolve().parent.parent
if str(_COCOMIG_ROOT) not in sys.path:
    sys.path.insert(0, str(_COCOMIG_ROOT))

import json
import multiprocessing as mp
from pathlib import Path

import torch
from cocomig_config import CocomigPaths
from pipeline.io import read_benchmark_items, resolve_segment_mask_path
from PIL import Image
from tqdm import tqdm


def _reference_tuples_to_dicts(reference_info: list[list[tuple]]) -> list[list[dict]]:
    """``src.model.generate.generate`` expects nested dicts (image / bbox / mask), not path tuples."""
    return [[{"image": a, "bbox": b, "mask": c} for (a, b, c) in row] for row in reference_info]


def build_reference_info_for_batch(
    batch: list,
    segments_category_info: list,
    segments_path: Path,
    segments_mask_path: Path,
    image_width: int,
    device: str | torch.device,
) -> list[list[tuple]]:
    reference_info = []
    for j, item in enumerate(batch):
        segment_info = []
        for segment in item["segment"]:
            bbox = torch.tensor(segment["bbox"], device=device) * image_width
            bbox_width, bbox_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
            bbox_aspect_ratio = bbox_width / bbox_height
            segment_caption = segment["label"]
            segment_item = next((x for x in segments_category_info if x["caption"] == segment_caption), None)
            assert segment_item is not None
            segment_category_index = segment_item["index"]
            segment_category_caption = segment_item["caption"].strip().replace(" ", "_")
            valid_mask_samples = segment_item.get("available_mask_image", [])
            assert valid_mask_samples, f"No valid mask samples found for caption '{segment_caption}'."

            closest_mask_sample = 0
            min_aspect_ratio_diff = float("inf")
            for mask_sample_index in valid_mask_samples:
                try:
                    seg_path = (
                        segments_path
                        / f"{segment_category_index:04d}_{mask_sample_index:02d}_{segment_category_caption}.jpg"
                    )
                    segment_image_pil = Image.open(seg_path).convert("RGB")
                    segment_image_aspect_ratio = segment_image_pil.width / segment_image_pil.height
                    aspect_ratio_diff = abs(segment_image_aspect_ratio - bbox_aspect_ratio)
                    if aspect_ratio_diff < min_aspect_ratio_diff:
                        min_aspect_ratio_diff = aspect_ratio_diff
                        closest_mask_sample = mask_sample_index
                except Exception as e:
                    print(f"Error opening segment image {seg_path}: {e}")
                    continue

            segment_mask_path = resolve_segment_mask_path(
                segments_mask_path,
                segment_category_index,
                closest_mask_sample,
                segment_category_caption,
            )
            segment_image_path = segments_path / (
                f"{segment_category_index:04d}_{closest_mask_sample:02d}_{segment_category_caption}.jpg"
            )
            assert segment_image_path.is_file(), f"Segment image {segment_image_path} does not exist."
            segment_info.append((str(segment_image_path), bbox, str(segment_mask_path)))
        reference_info.append(segment_info)
    return reference_info


def sample_bench_images_with_model_generate(
    flux_pipe,
    items: list,
    paths: CocomigPaths,
    segments_category_info: list | None,
    segments_path: Path,
    segments_mask_path: Path,
    output_dir: Path | None,
    output_ref_dir: Path | None,
    resize_pattern: str,
    file_name_prefix: str = "",
    image_width: int = 512,
    batch_size: int = 16,
    device: str | torch.device = "cuda",
) -> None:
    """Batched ``src.model.generate.generate`` (same path as ``inference.py``)."""
    paths.ensure_generation_repo_on_syspath()
    from src.model.generate import generate as model_generate  # type: ignore  # noqa: WPS433
    from src.utils.image_process import annotate  # type: ignore  # noqa: WPS433

    if segments_category_info is None:
        with open(paths.instance_categories_filtered_json, "r") as f:
            segments_category_info = json.load(f)

    dev = torch.device(device) if isinstance(device, str) else device

    for i in tqdm(range(0, len(items), batch_size), desc=f"Device {device} processing"):
        batch = items[i : i + batch_size]
        item_index = [item["index"] for item in batch]
        captions = [item["caption"] for item in batch]
        reference_info = build_reference_info_for_batch(
            batch, segments_category_info, segments_path, segments_mask_path, image_width, device
        )
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
                img.save(output_dir / f"{file_name_prefix}{item_index[j]:04d}.jpg")
            if output_ref_dir is not None:
                phrases_for_annotation = []
                bboxes_for_annotation = []
                for k in range(len(batch[j]["segment"])):
                    phrases_for_annotation.append(batch[j]["segment"][k]["label"])
                    bboxes_for_annotation.append(
                        (torch.tensor(batch[j]["segment"][k]["bbox"], device=device) * image_width)
                        .round()
                        .to(torch.int32)
                    )
                annotated_img = annotate(img, bboxes_for_annotation, phrases_for_annotation)
                annotated_img.save(output_ref_dir / f"{file_name_prefix}{item_index[j]:04d}_annotated.jpg")


def _processor(
    items: list,
    paths: CocomigPaths,
    segments_path: Path,
    segments_mask_path: Path,
    output_dir: Path | None,
    output_ref_dir: Path | None,
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

    print(f"CUDA {gpu_id}: loading Kontext + LoRA from disk")
    flux_pipe = load_model(kontext_model_path, adapter_path, device=device)
    flux_pipe.set_progress_bar_config(disable=True)
    sample_bench_images_with_model_generate(
        flux_pipe=flux_pipe,
        items=items,
        paths=paths,
        segments_category_info=None,
        segments_path=segments_path,
        segments_mask_path=segments_mask_path,
        output_dir=output_dir,
        output_ref_dir=output_ref_dir,
        resize_pattern=resize_pattern,
        image_width=image_width,
        batch_size=batch_size,
        device=device,
    )


def parallel_sample_bench_images(
    items: list,
    paths: CocomigPaths,
    segments_path: Path,
    segments_mask_path: Path,
    output_dir: Path | None,
    output_ref_dir: Path | None,
    image_width: int,
    batch_size: int,
    resize_pattern: str,
    kontext_model_path: str,
    adapter_path: str,
) -> None:
    num_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_gpus}")
    if num_gpus == 0:
        raise RuntimeError("No GPU available for parallel processing.")

    chunk_size = len(items) // num_gpus + 1
    chunks = [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]

    processes = []
    for gpu_id, chunk in enumerate(chunks):
        p = mp.Process(
            target=_processor,
            args=(
                chunk,
                paths,
                segments_path,
                segments_mask_path,
                output_dir,
                output_ref_dir,
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


def run_sample_bench(
    paths: CocomigPaths,
    image_width: int,
    batch_size: int,
    resize_pattern: str,
    kontext_model_path: str,
    adapter_path: str,
    skip_existing: bool = False,
) -> None:
    benchmark = read_benchmark_items(paths.benchmark_json)
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

    parallel_sample_bench_images(
        items=to_run,
        paths=paths,
        segments_path=paths.reference_images_dir,
        segments_mask_path=paths.mask_dir,
        output_dir=save_path,
        output_ref_dir=save_ref_path,
        image_width=image_width,
        batch_size=batch_size,
        resize_pattern=resize_pattern,
        kontext_model_path=kontext_model_path,
        adapter_path=adapter_path,
    )
