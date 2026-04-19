"""
LayoutSAM reference pipeline: segment JPGs (Flux), valid bbox (HF Grounding DINO), mask filter.

Aligned with ``cocomig_reference`` thresholds for the DINO stage. Paths come from ``layoutsam_config.LayoutsamPaths``.
"""

from __future__ import annotations

import json
import multiprocessing
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[4]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from layoutsam_config import LayoutsamPaths

# ---------------------------------------------------------------------------
# gen-segments (Flux + optional Turbo)
# ---------------------------------------------------------------------------


def read_prompts_from_benchmark_json(file_path: Path) -> list[dict]:
    with open(file_path, "r") as file:
        prompts = json.load(file)

    processed_prompts = []
    for p in prompts:
        image_index = p["index"]
        for s in p["segments"]:
            processed_prompts.append(
                {
                    "prompt": s["caption"],
                    "image_index": image_index,
                    "segment_index": s["index"],
                }
            )

    print(f"Read {len(processed_prompts)} segment prompts from {file_path}")
    return processed_prompts


def generate_image_chunk(
    prompts: list[dict],
    flux_model,
    save_dir: str | Path,
    batch_size: int = 4,
    use_turbo: bool = True,
) -> None:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    for i in range(0, len(prompts), batch_size):
        prompt_batch = prompts[i : i + batch_size]
        prompt = [p["prompt"] for p in prompt_batch]
        with torch.no_grad():
            imgs = flux_model(
                prompt=prompt,
                height=512,
                width=512,
                num_inference_steps=8 if use_turbo else 28,
            ).images
        for j, img in enumerate(imgs):
            image_index = prompt_batch[j]["image_index"]
            segment_index = prompt_batch[j]["segment_index"]
            img_save_path = save_dir / f"{image_index:04d}_{segment_index:02d}.jpg"
            img.save(img_save_path)


def _worker_generate(
    prompts_chunk: list[dict],
    image_save_dir: Path,
    batch_size: int,
    gpu_id: int,
    flux_path: str,
    turbo_lora_path: str,
    use_turbo: bool,
) -> None:
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    from diffusers import FluxPipeline

    model = FluxPipeline.from_pretrained(flux_path, torch_dtype=torch.bfloat16).to(device)
    if use_turbo and turbo_lora_path:
        model.load_lora_weights(turbo_lora_path)
        model.fuse_lora()
    generate_image_chunk(
        prompts_chunk,
        model,
        image_save_dir,
        batch_size=batch_size,
        use_turbo=bool(use_turbo and turbo_lora_path),
    )


def run_generate_segments(
    paths: LayoutsamPaths,
    *,
    batch_size: int = 8,
    num_gpus: int | None = None,
    use_turbo: bool = True,
) -> None:
    if not paths.flux_turbo_lora_path and use_turbo:
        print("Warning: FLUX_TURBO_LORA_PATH empty; generating without Turbo (more steps).")
    paths.segments_dir.mkdir(parents=True, exist_ok=True)
    raw_prompts = read_prompts_from_benchmark_json(paths.layoutsam_benchmark_json)

    if num_gpus is None:
        num_gpus = torch.cuda.device_count()
    if num_gpus <= 0:
        raise RuntimeError("No CUDA device for gen-segments.")

    try:
        torch.multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    chunk_size = len(raw_prompts) // num_gpus + 1
    prompt_chunks = [raw_prompts[i : i + chunk_size] for i in range(0, len(raw_prompts), chunk_size)]

    processes = []
    for gpu_idx in range(num_gpus):
        if gpu_idx >= len(prompt_chunks) or not prompt_chunks[gpu_idx]:
            continue
        p = multiprocessing.Process(
            target=_worker_generate,
            args=(
                prompt_chunks[gpu_idx],
                paths.segments_dir,
                batch_size,
                gpu_idx,
                paths.flux_model_path,
                paths.flux_turbo_lora_path,
                use_turbo,
            ),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print(f"Segment images saved under {paths.segments_dir}")


# ---------------------------------------------------------------------------
# valid-bbox (transformers Grounding DINO — same as cocomig ``crop-reference``)
# ---------------------------------------------------------------------------

_BOX_THRESHOLD = 0.35
_TEXT_THRESHOLD = 0.25


def _predict_first_box_xyxy(
    image_pil: Image.Image,
    text_prompt: str,
    model,
    processor,
    device: str | torch.device,
) -> tuple[int, int, int, int] | None:
    tp = text_prompt.lower()
    if not tp.endswith("."):
        tp = tp + "."
    inputs = processor(images=image_pil, text=tp, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=_BOX_THRESHOLD,
        text_threshold=_TEXT_THRESHOLD,
        target_sizes=[image_pil.size[::-1]],
    )
    boxes = results[0]["boxes"].cpu().numpy()
    if boxes is None or len(boxes) == 0:
        return None
    x1, y1, x2, y2 = boxes[0].tolist()
    w, h = image_pil.size
    x1i = max(0, min(int(x1), w - 1))
    y1i = max(0, min(int(y1), h - 1))
    x2i = max(x1i + 1, min(int(x2), w))
    y2i = max(y1i + 1, min(int(y2), h))
    return (x1i, y1i, x2i, y2i)


def get_reference_valid_bbox(
    reference_info: list,
    segments_dir: Path,
    grounding_model_id: str,
    gpu_id: int = 0,
    process_id: int = 0,
    _model_bundle=None,
) -> list:
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"
    segments_dir = Path(segments_dir)

    if _model_bundle is None:
        processor = AutoProcessor.from_pretrained(grounding_model_id)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(grounding_model_id).to(device)
    else:
        model, processor = _model_bundle

    print(f"Process {process_id} on GPU {gpu_id}: items {reference_info[0]['index']} … {reference_info[-1]['index']}")

    for reference in tqdm(reference_info):
        image_index = reference["index"]
        for segment in reference["segments"]:
            segment_index = segment["index"]
            caption = segment["caption"]

            image_path = segments_dir / f"{image_index:04d}_{segment_index:02d}.jpg"
            image_pil = Image.open(image_path).convert("RGB")
            xyxy = _predict_first_box_xyxy(image_pil, caption, model, processor, device)
            if xyxy is not None:
                segment["valid_bbox"] = [{"image_path": str(image_path), "valid_bbox": list(xyxy)}]
            else:
                segment["valid_bbox"] = [{"image_path": str(image_path), "valid_bbox": None}]
                print(f"No bbox for {image_path} caption={caption!r}")

    return reference_info


def parallel_processor(
    reference_info: list,
    paths: LayoutsamPaths,
    num_processes: int | None,
) -> None:
    mid = (paths.grounding_dino_model or "").strip()
    if not mid:
        raise ValueError(
            "Set GROUNDING_DINO_MODEL or layoutsam_config.GROUNDING_DINO_MODEL (HF id for transformers Grounding DINO)."
        )

    gpu_count = torch.cuda.device_count()
    if num_processes is None or num_processes <= 0:
        num_processes = max(1, gpu_count)
    num_processes = min(num_processes, len(reference_info), max(1, gpu_count * 4))

    chunk_size = (len(reference_info) + num_processes - 1) // num_processes
    chunks = [reference_info[i : i + chunk_size] for i in range(0, len(reference_info), chunk_size)]

    torch.multiprocessing.set_start_method("spawn", force=True)
    with torch.multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.starmap(
            get_reference_valid_bbox,
            [
                (
                    chunk,
                    paths.segments_dir,
                    mid,
                    i % max(1, gpu_count),
                    i,
                    None,
                )
                for i, chunk in enumerate(chunks)
            ],
        )

    combined_results: list = []
    for result in results:
        combined_results.extend(result)

    combined_results.sort(key=lambda x: x["index"])

    paths.benchmark_with_bbox_json.parent.mkdir(parents=True, exist_ok=True)
    with open(paths.benchmark_with_bbox_json, "w") as f:
        json.dump(combined_results, f, indent=2)
    print(f"Wrote {paths.benchmark_with_bbox_json}")


def run_valid_bbox(paths: LayoutsamPaths, num_processes: int | None) -> None:
    with open(paths.layoutsam_benchmark_json, "r") as f:
        items = json.load(f)
    parallel_processor(items, paths, num_processes=num_processes)


# ---------------------------------------------------------------------------
# filter-benchmark (mask quality)
# ---------------------------------------------------------------------------


def _get_valid_bounds(non_zero_pixels):
    rows, cols = np.nonzero(non_zero_pixels)
    if rows.size == 0 or cols.size == 0:
        return None
    return cols.min(), cols.max(), rows.min(), rows.max()


def _determine_valid_mask(segment_mask, valid_bbox):
    if valid_bbox is None:
        return 1
    x1, y1, x2, y2 = valid_bbox
    segment_mask_cropped = segment_mask[y1:y2, x1:x2]
    height, width = segment_mask_cropped.shape
    valid_bounds = _get_valid_bounds(segment_mask_cropped)
    if valid_bounds is None:
        return 2
    x1_v, x2_v, y1_v, y2_v = valid_bounds
    valid_height = y2_v - y1_v
    valid_width = x2_v - x1_v
    return (valid_height / height, valid_width / width)


def _get_segment_mask_area(segment_mask):
    non_zero_pixels = np.count_nonzero(segment_mask)
    total_pixels = segment_mask.size
    if total_pixels == 0:
        return None
    return non_zero_pixels / total_pixels


def run_filter_benchmark(paths: LayoutsamPaths) -> None:
    with open(paths.benchmark_with_bbox_json, "r") as f:
        data = json.load(f)

    for item in tqdm(data):
        image_index = item["index"]
        for segment in item["segments"]:
            segment_index = segment["index"]
            for mask in segment["valid_bbox"]:
                segment_mask_path = mask["image_path"]
                segment_mask_path = segment_mask_path.replace(".jpg", "_mask.jpg")
                file_dir_name = segment_mask_path.split("/")[-2]
                segment_mask_path = segment_mask_path.replace(file_dir_name, file_dir_name + "_masks")
                mask["mask_path"] = segment_mask_path
                valid_bbox = mask["valid_bbox"]
                segment_mask = Image.open(segment_mask_path).convert("L")
                segment_mask_np = np.array(segment_mask)

                is_valid = _determine_valid_mask(segment_mask_np, valid_bbox)

                if isinstance(is_valid, tuple):
                    mask["mask_valid_part"] = [is_valid[0], is_valid[1]]
                else:
                    mask["mask_valid_part"] = _get_segment_mask_area(segment_mask_np)

            not_none_mask_valid_parts = [
                mask for mask in segment["valid_bbox"] if isinstance(mask["mask_valid_part"], list)
            ]
            if not_none_mask_valid_parts:
                not_none_mask_valid_parts = sorted(
                    not_none_mask_valid_parts,
                    key=lambda x: x["mask_valid_part"][0] * x["mask_valid_part"][1],
                    reverse=True,
                )
            area_mask_valid_parts = [
                mask
                for mask in segment["valid_bbox"]
                if (mask["mask_valid_part"] is not None and not isinstance(mask["mask_valid_part"], list))
            ]
            if area_mask_valid_parts:
                area_mask_valid_parts = sorted(area_mask_valid_parts, key=lambda x: x["mask_valid_part"], reverse=True)

            sorted_mask_valid_part = not_none_mask_valid_parts + area_mask_valid_parts

            filtered_mask_valid_part = [
                mask
                for mask in sorted_mask_valid_part
                if isinstance(mask["mask_valid_part"], list)
                and mask["mask_valid_part"][0] > 0.75
                and mask["mask_valid_part"][1] > 0.75
            ]

            if not filtered_mask_valid_part:
                filtered_mask_valid_part = sorted_mask_valid_part[:1]
            assert len(filtered_mask_valid_part) > 0, (
                f"Empty filter for image {image_index} segment {segment_index}: {segment['valid_bbox']}"
            )
            segment["valid_bbox"] = filtered_mask_valid_part

    paths.benchmark_filtered_json.parent.mkdir(parents=True, exist_ok=True)
    with open(paths.benchmark_filtered_json, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Filtered benchmark written to {paths.benchmark_filtered_json}")
