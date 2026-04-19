#!/usr/bin/env python3
"""Sample LAMICBench+ with Kontext + adapter, minimizing hardcoded paths."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import sys
from pathlib import Path

import supervision as sv
import torch
from PIL import Image
from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[2]
_env_file = _REPO_ROOT / ".env"
if _env_file.is_file():
    from dotenv import load_dotenv

    load_dotenv(_env_file)

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.model.generate import generate, load_model

DEFAULT_LAYOUT_JSON = _SCRIPT_DIR / "lamicbench_layout_pack" / "lamicbench_layout.json"
DEFAULT_LAYOUT_DIR = _SCRIPT_DIR / "lamicbench_layout_pack" / "layout_images"
DEFAULT_REF_DIR = _REPO_ROOT / "bench" / "lamicbench_plus" / "lamicbench_plus_files" / "reference_images"
DEFAULT_REF_MASK_DIR = _REPO_ROOT / "bench" / "lamicbench_plus" / "lamicbench_plus_files" / "reference_masks"
DEFAULT_OUTPUT_DIR = _REPO_ROOT / "bench" / "output" / "lamicbench_plus"


def annotate(image_source: Image.Image, boxes, phrases) -> Image.Image:
    xyxy = torch.stack(boxes, dim=0).cpu().numpy()
    detections = sv.Detections(xyxy=xyxy)
    labels = [str(phrase) for phrase in phrases]
    bbox_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
    label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX)
    annotated = image_source.copy()
    annotated = bbox_annotator.annotate(scene=annotated, detections=detections)
    annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)
    if not isinstance(annotated, Image.Image):
        annotated = Image.fromarray(annotated)
    return annotated


def _existing_count_for_item(images_dir: Path, item_index: int) -> int:
    prefix = f"{item_index:04d}_"
    return sum(
        1
        for p in images_dir.iterdir()
        if p.is_file() and p.name.startswith(prefix) and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )


def _build_reference_info(batch: list, image_width: int, ref_dir: Path, ref_mask_dir: Path, layout_dir: Path):
    item_indices = [item["index"] for item in batch]
    captions = [item["prompt"] for item in batch]
    layout_images = []
    reference_info = [[] for _ in batch]
    bboxes_for_annotation = [[] for _ in batch]
    labels_for_annotation = [[] for _ in batch]

    for j, item in enumerate(batch):
        layout_path = layout_dir / f"layout_{item_indices[j]:04d}.png"
        if not layout_path.is_file():
            raise FileNotFoundError(f"Missing layout image: {layout_path}")
        layout_images.append(
            Image.open(layout_path).convert("RGBA").resize((image_width, image_width), Image.Resampling.LANCZOS)
        )

        for instance in item["references"]:
            bbox = (torch.tensor(instance["bbox"]) * image_width).round().to(dtype=torch.int32)
            image_rel = instance["image_path"]
            image_path = ref_dir / image_rel
            mask_path = ref_mask_dir / image_rel.replace(".jpg", "_mask.png")
            if not image_path.is_file():
                raise FileNotFoundError(f"Missing reference image: {image_path}")
            if not mask_path.is_file():
                raise FileNotFoundError(f"Missing reference mask: {mask_path}")
            reference_info[j].append({"image": str(image_path), "bbox": bbox, "mask": str(mask_path)})
            bboxes_for_annotation[j].append(bbox)
            labels_for_annotation[j].append(instance["phrase"])

    return item_indices, captions, layout_images, reference_info, bboxes_for_annotation, labels_for_annotation


def generate_bench_images(
    items: list,
    layout_dir: Path,
    ref_dir: Path,
    ref_mask_dir: Path,
    output_dir: Path,
    annotated_dir: Path,
    image_width: int,
    batch_size: int,
    num_samples_per_image: int,
    gpu_id: int,
    seeds: list[int] | None,
    kontext_model_path: str,
    adapter_path: str,
) -> None:
    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    print(f"GPU {gpu_id}: loading model on {device}")
    flux_pipe = load_model(kontext_model_path, adapter_path, device=device)
    flux_pipe.set_progress_bar_config(disable=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir.mkdir(parents=True, exist_ok=True)

    if seeds:
        seed_passes = [(s, idx) for idx, s in enumerate(seeds)]
    else:
        seed_passes = [(None, idx) for idx in range(num_samples_per_image)]

    for seed_val, sample_idx in seed_passes:
        for i in tqdm(range(0, len(items), batch_size), desc=f"GPU {gpu_id} seed={seed_val}"):
            batch = items[i : i + batch_size]
            batch = [it for it in batch if not (output_dir / f"{it['index']:04d}_{sample_idx}.jpg").is_file()]
            if not batch:
                continue

            (
                item_indices,
                captions,
                layout_images,
                reference_info,
                bboxes_for_annotation,
                labels_for_annotation,
            ) = _build_reference_info(batch, image_width, ref_dir, ref_mask_dir, layout_dir)

            batch_seeds = [random.randint(1, 2**32 - 1) for _ in batch] if seed_val is None else [seed_val] * len(batch)

            with torch.inference_mode():
                imgs = generate(
                    flux_pipe=flux_pipe,
                    prompts=captions,
                    reference_info=reference_info,
                    layout_image=layout_images,
                    height=image_width,
                    width=image_width,
                    device=dev,
                    seed=batch_seeds,
                )

            for j, img in enumerate(imgs):
                img_pil: Image.Image
                if isinstance(img, Image.Image):
                    img_pil = img
                else:
                    img_pil = Image.fromarray(img)
                fname = f"{item_indices[j]:04d}_{sample_idx}.jpg"
                img_pil.save(output_dir / fname)
                ann = annotate(img_pil, bboxes_for_annotation[j], labels_for_annotation[j])
                if not isinstance(ann, Image.Image):
                    ann = Image.fromarray(ann)
                ann.save(annotated_dir / f"{item_indices[j]:04d}_{sample_idx}_annotated.jpg")


def parallel_generate_bench_images(
    items: list,
    layout_dir: Path,
    ref_dir: Path,
    ref_mask_dir: Path,
    output_dir: Path,
    annotated_dir: Path,
    image_width: int,
    num_samples_per_image: int,
    batch_size: int,
    seeds: list[int] | None,
    kontext_model_path: str,
    adapter_path: str,
) -> None:
    num_gpus = torch.cuda.device_count()
    if num_gpus <= 0:
        raise RuntimeError("No GPU available for parallel processing.")
    print(f"Using {num_gpus} GPUs")

    chunks = [[] for _ in range(num_gpus)]
    for idx, item in enumerate(items):
        chunks[idx % num_gpus].append(item)

    processes = []
    for gpu_id, chunk in enumerate(chunks):
        if not chunk:
            continue
        p = mp.Process(
            target=generate_bench_images,
            args=(
                chunk,
                layout_dir,
                ref_dir,
                ref_mask_dir,
                output_dir,
                annotated_dir,
                image_width,
                batch_size,
                num_samples_per_image,
                gpu_id,
                seeds,
                kontext_model_path,
                adapter_path,
            ),
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()


def _parse_seeds(raw: str | None) -> list[int] | None:
    if raw is None or not raw.strip():
        return None
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="LAMICBench+ sample-bench runner")
    parser.add_argument("--layout-json", type=Path, default=DEFAULT_LAYOUT_JSON)
    parser.add_argument("--layout-dir", type=Path, default=DEFAULT_LAYOUT_DIR)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REF_DIR)
    parser.add_argument("--reference-mask-dir", type=Path, default=DEFAULT_REF_MASK_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--image-width", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-samples-per-image", type=int, default=1)
    parser.add_argument("--seeds", type=str, default="42,43,44,45", help="Comma-separated seeds. Empty means random.")
    parser.add_argument("--kontext-model-path", type=str, default=os.environ.get("KONTEXT_MODEL_PATH", ""))
    parser.add_argument("--adapter-path", type=str, default=os.environ.get("ADAPTER_PATH", ""))
    args = parser.parse_args()

    kontext_model_path = (args.kontext_model_path or "").strip()
    adapter_path = (args.adapter_path or "").strip()
    if not kontext_model_path or not adapter_path:
        raise SystemExit("Need --kontext-model-path and --adapter-path (or env KONTEXT_MODEL_PATH / ADAPTER_PATH).")

    layout_json = args.layout_json.resolve()
    layout_dir = args.layout_dir.resolve()
    ref_dir = args.reference_dir.resolve()
    ref_mask_dir = args.reference_mask_dir.resolve()
    images_dir = (args.output_dir / "images").resolve()
    annotated_dir = (args.output_dir / "annotated_images").resolve()
    images_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir.mkdir(parents=True, exist_ok=True)

    with open(layout_json, "r", encoding="utf-8") as f:
        benchmark = json.load(f)

    seeds = _parse_seeds(args.seeds)
    expected_per_item = len(seeds) if seeds else args.num_samples_per_image
    items_to_process = []
    for item in benchmark:
        done = _existing_count_for_item(images_dir, item["index"])
        if done >= expected_per_item:
            continue
        items_to_process.append(item)

    print(f"Items to generate: {len(items_to_process)} / {len(benchmark)}")
    if not items_to_process:
        return

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    parallel_generate_bench_images(
        items=items_to_process,
        layout_dir=layout_dir,
        ref_dir=ref_dir,
        ref_mask_dir=ref_mask_dir,
        output_dir=images_dir,
        annotated_dir=annotated_dir,
        image_width=args.image_width,
        num_samples_per_image=args.num_samples_per_image,
        batch_size=args.batch_size,
        seeds=seeds,
        kontext_model_path=kontext_model_path,
        adapter_path=adapter_path,
    )


if __name__ == "__main__":
    main()
