"""Flux reference generation, HF Grounding DINO crop, and mask coverage filtering."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_COCOMIG_DIR = Path(__file__).resolve().parent.parent
if str(_COCOMIG_DIR) not in sys.path:
    sys.path.insert(0, str(_COCOMIG_DIR))

import numpy as np
import torch
from diffusers import FluxPipeline
from PIL import Image
from tqdm import tqdm

from cocomig_config import CocomigPaths
from pipeline.io import masked_overlay_path_for_mask, resolve_segment_mask_path


def process_label(label: str) -> str:
    words = label.split()
    if len(words) > 2:
        words.insert(2, "colored")
    else:
        raise ValueError("Label must have at least two words.")
    words.append("in realistic style")
    return " ".join(words)


def generate_reference_images(
    segment_items: list,
    output_dir: str | Path,
    flux_path: str | Path,
    target_size: tuple[int, int] = (512, 512),
    batch_size: int = 64,
    num_samples: int = 32,
    flux_model: FluxPipeline | None = None,
    gpu_id: int = 0,
) -> None:
    torch.cuda.set_device(gpu_id)
    if flux_model is None:
        flux_model = FluxPipeline.from_pretrained(str(flux_path), torch_dtype=torch.bfloat16).to("cuda")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i in range(0, len(segment_items), batch_size):
        batch = segment_items[i : i + batch_size]
        raw_captions = [item["caption"] for item in batch]
        processed_captions = [process_label(caption) for caption in raw_captions]
        indexes = [item["index"] for item in batch]

        for j in range(num_samples):
            with torch.inference_mode():
                imgs = flux_model(
                    prompt=processed_captions,
                    width=target_size[0],
                    height=target_size[1],
                ).images
            for k, img in enumerate(imgs):
                index = indexes[k]
                caption_postfix = raw_captions[k].replace(" ", "_").replace(",", "").replace(".", "")
                img.save(output_dir / f"{index:04d}_{j:02d}_{caption_postfix}.jpg")


def parallel_generate_reference_images(
    segment_items: list,
    output_dir: str | Path,
    flux_path: str | Path,
    target_size: tuple[int, int] = (512, 512),
    batch_size: int = 64,
    num_samples: int = 32,
) -> None:
    device_count = torch.cuda.device_count()
    print(f"Using {device_count} GPU(s) for parallel processing.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    chunk_size = len(segment_items) // device_count + 1
    chunks = [segment_items[i : i + chunk_size] for i in range(0, len(segment_items), chunk_size)]

    import torch.multiprocessing as mp

    mp.set_start_method("spawn", force=True)

    with mp.Pool(processes=device_count) as pool:
        pool.starmap(
            generate_reference_images,
            [
                (chunk, output_dir, flux_path, target_size, batch_size, num_samples, None, i)
                for i, chunk in enumerate(chunks)
            ],
        )


# Thresholds aligned with bench/lamicbench_plus/utils/grounded_sam2.py (grounding stage only).
_BOX_THRESHOLD = 0.35
_TEXT_THRESHOLD = 0.25


def _predict_first_box_xyxy(
    image_pil: Image.Image,
    text_prompt: str,
    model,
    processor,
    device: str | torch.device,
) -> tuple[int, int, int, int] | None:
    """HF Grounding DINO zero-shot detection; returns first box as integer xyxy for PIL.crop."""
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


def crop_reference_images(
    reference_info: list,
    segments_dir: str | Path,
    save_path: str | Path,
    grounding_model_id: str,
    _model_bundle=None,
    gpu_id: int = 0,
    process_id: int = 0,
) -> None:
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"
    segments_dir = Path(segments_dir)
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    if _model_bundle is None:
        processor = AutoProcessor.from_pretrained(grounding_model_id)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(grounding_model_id).to(device)
    else:
        model, processor = _model_bundle

    print(
        f"Process {process_id} using GPU {gpu_id} is processing items from "
        f"{reference_info[0]['index']} to {reference_info[-1]['index']}"
    )

    for reference in tqdm(reference_info):
        image_path_prefix = f"{reference['index']:04d}_"
        image_files = [
            f for f in os.listdir(segments_dir) if f.startswith(image_path_prefix) and f.endswith(".jpg")
        ]
        for image_file in image_files:
            caption = reference["caption"]
            image_path = segments_dir / image_file
            image_pil = Image.open(image_path).convert("RGB")
            xyxy = _predict_first_box_xyxy(image_pil, caption, model, processor, device)
            if xyxy is not None:
                cropped_image = image_pil.crop(xyxy)
                cropped_image.save(save_path / image_file)
            else:
                print(f"Warning: No bounding boxes found for phrase '{caption}' in {image_file} - Skipping cropping.")


def parallel_crop_reference_images(
    reference_info: list,
    segments_dir: str | Path,
    save_path: str | Path,
    grounding_model_id: str,
    num_processes: int | None = None,
) -> None:
    gpu_count = torch.cuda.device_count()
    if num_processes is None:
        num_processes = gpu_count
    chunk_size = (len(reference_info) + num_processes - 1) // num_processes
    chunks = [reference_info[i : i + chunk_size] for i in range(0, len(reference_info), chunk_size)]

    torch.multiprocessing.set_start_method("spawn", force=True)
    with torch.multiprocessing.Pool(processes=num_processes) as pool:
        pool.starmap(
            crop_reference_images,
            [
                (chunk, segments_dir, save_path, grounding_model_id, None, i % gpu_count, i)
                for i, chunk in enumerate(chunks)
            ],
        )


def run_build_instance_categories(paths: CocomigPaths) -> None:
    from pipeline.io import read_benchmark_segments  # noqa: WPS433

    segments = read_benchmark_segments(paths.benchmark_json)
    caption_list: list[dict] = []
    for segment in segments:
        caption, _, image_index, segment_index = segment
        if not any(caption == item["caption"] for item in caption_list):
            caption_list.append({"caption": caption, "segment_index": [(image_index, segment_index)]})
        else:
            for item in caption_list:
                if item["caption"] == caption:
                    item["segment_index"].append((image_index, segment_index))
                    break
    for i, item in enumerate(caption_list):
        item["index"] = i
    paths.instance_categories_json.parent.mkdir(parents=True, exist_ok=True)
    with open(paths.instance_categories_json, "w") as f:
        json.dump(caption_list, f, indent=4)
    print(f"Total categories: {len(caption_list)} -> {paths.instance_categories_json}")

    paths.instance_categories_filtered_json.parent.mkdir(parents=True, exist_ok=True)
    with open(paths.instance_categories_filtered_json, "w") as f:
        json.dump(caption_list, f, indent=2)
    print(
        f"Placeholder {paths.instance_categories_filtered_json} (same as categories until `filter mask` adds mask picks)."
    )


def run_generate_reference_flux(paths: CocomigPaths, target_size: tuple[int, int], batch_size: int, num_samples: int) -> None:
    with open(paths.instance_categories_json, "r") as f:
        instance_categories = json.load(f)
    paths.segments_dir.mkdir(parents=True, exist_ok=True)
    parallel_generate_reference_images(
        instance_categories,
        paths.segments_dir,
        paths.flux_model_path,
        target_size=target_size,
        batch_size=batch_size,
        num_samples=num_samples,
    )
    print(f"Generated reference images saved to {paths.segments_dir}")


def run_crop_references(paths: CocomigPaths, num_processes: int | None) -> None:
    with open(paths.instance_categories_json, "r") as f:
        items = json.load(f)
    paths.reference_images_dir.mkdir(parents=True, exist_ok=True)
    parallel_crop_reference_images(
        items,
        paths.segments_dir,
        paths.reference_images_dir,
        paths.grounding_dino_model,
        num_processes=num_processes,
    )


# --- Mask filtering and sync -----------------------------------------------


def cal_mask_coverage(mask: str | Path | Image.Image, threshold: float = 0.75) -> bool:
    if isinstance(mask, (str, Path)):
        mask = Image.open(mask).convert("L")
    elif isinstance(mask, Image.Image):
        mask = mask.convert("L")
    mask_array = np.array(mask)
    mask_array = (mask_array > 128).astype(np.uint8)
    height, width = mask_array.shape
    valid_height = np.sum(mask_array, axis=1) > 0
    valid_width = np.sum(mask_array, axis=0) > 0
    valid_height_ratio = np.sum(valid_height) / height
    valid_width_ratio = np.sum(valid_width) / width
    return valid_height_ratio >= threshold and valid_width_ratio >= threshold


def filter_with_mask(paths: CocomigPaths) -> None:
    """Read category definitions from ``instance_categories_json``; write mask-qualified state only to ``instance_categories_filtered_json``."""
    with open(paths.instance_categories_json, "r") as f:
        instance_categories = json.load(f)

    mask_root = Path(paths.mask_dir)
    for item in instance_categories:
        item_index = item["index"]
        mask_files = [
            f
            for f in os.listdir(mask_root)
            if f.startswith(f"{item_index:04d}_") and (f.endswith("_mask.jpg") or f.endswith("_mask.png"))
        ]
        if not mask_files:
            print(f"Warning: No mask file found for item index {item_index}. Skipping.")
            continue
        for mask_file in mask_files:
            mask_path = mask_root / mask_file
            mask_sample_index = int(mask_file.split("_")[1])
            if cal_mask_coverage(mask_path):
                item.setdefault("available_mask_image", [])
                if mask_sample_index not in item["available_mask_image"]:
                    item["available_mask_image"].append(mask_sample_index)
            else:
                if "available_mask_image" in item and mask_sample_index in item["available_mask_image"]:
                    item["available_mask_image"].remove(mask_sample_index)
                    print(f"Removed {mask_file} from available_mask_image for item index {item_index}.")

    paths.instance_categories_filtered_json.parent.mkdir(parents=True, exist_ok=True)
    with open(paths.instance_categories_filtered_json, "w") as f:
        json.dump(instance_categories, f, indent=2)
    print(
        f"Mask filtering completed -> {paths.instance_categories_filtered_json} "
        f"(adds available_mask_image; {paths.instance_categories_json.name} left unchanged)."
    )


def remove_invalid_images(paths: CocomigPaths) -> None:
    input_dir = Path(paths.mask_dir_full)
    output_dir = Path(paths.mask_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(paths.instance_categories_filtered_json, "r") as f:
        instance_categories = json.load(f)

    valid_masks: set[str] = set()
    for item in tqdm(instance_categories):
        if "available_mask_image" in item:
            mask_caption = item["caption"].strip().replace(" ", "_").lower()
            for mask_index in item["available_mask_image"]:
                mask_path = resolve_segment_mask_path(input_dir, item["index"], mask_index, mask_caption)
                assert mask_path.is_file(), f"Mask file not found: {mask_path}"
                valid_masks.add(mask_path.name)
                masked_path = masked_overlay_path_for_mask(mask_path)
                assert masked_path.is_file(), f"Masked file {masked_path} does not exist."
                valid_masks.add(masked_path.name)
    for file_name in os.listdir(input_dir):
        if file_name in valid_masks:
            src_path = input_dir / file_name
            dst_path = output_dir / file_name
            if not dst_path.exists():
                os.link(src_path, dst_path)
