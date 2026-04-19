"""
LayoutSAM metrics: MiniCPM-V QA (eval) + CLIP / PickScore on generated images.

Paths default to this repo: ``bench/output/layoutsam/images`` and
``layoutsam_reference_pack/layoutsam_sampled.json`` beside this script.

Model / dataset Hugging Face ids are set as module-level constants (``MINICPM_MODEL_ID``,
``LAYOUTSAM_EVAL_DATASET``, ``CLIP_MODEL_ID``, …) near the top of this file.
"""

from __future__ import annotations

import ast
import json
import math
import os
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor, AutoTokenizer, CLIPModel, CLIPProcessor

# ---------------------------------------------------------------------------
# Paths (repo-relative defaults)
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[3]

DEFAULT_GENERATE_PATH = _REPO_ROOT / "bench" / "output" / "layoutsam" / "images"
DEFAULT_SAMPLED_BENCH = _SCRIPT_DIR / "layoutsam_reference_pack" / "layoutsam_sampled.json"
DEFAULT_CLIP_TEMP_DIR = _REPO_ROOT / "bench" / "output" / "layoutsam" / "_clip_pick_temp"

# ---------------------------------------------------------------------------
# Model / dataset ids (edit here)
# ---------------------------------------------------------------------------

MINICPM_MODEL_ID = "openbmb/MiniCPM-V-2_6"
LAYOUTSAM_EVAL_DATASET = "HuiZhang0812/LayoutSAM-eval"

CLIP_MODEL_ID = "openai/clip-vit-large-patch14"
PICKSCORE_PROCESSOR_ID = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
PICKSCORE_MODEL_ID = "yuvalkirstain/PickScore_v1"

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def chunk_evenly(items, num_chunks):
    chunks = [[] for _ in range(num_chunks)]
    for idx, item in enumerate(items):
        chunks[idx % num_chunks].append(item)
    return chunks


# ---------------------------------------------------------------------------
# Dataset (from layoutsam_dataset.py)
# ---------------------------------------------------------------------------


def adjust_and_normalize_bboxes(bboxes, orig_width, orig_height):
    normalized_bboxes = []
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        x1_norm = round(x1 / orig_width, 3)
        y1_norm = round(y1 / orig_height, 3)
        x2_norm = round(x2 / orig_width, 3)
        y2_norm = round(y2 / orig_height, 3)
        normalized_bboxes.append([x1_norm, y1_norm, x2_norm, y2_norm])

    return normalized_bboxes


class BboxDataset(Dataset):
    def __init__(self, dataset, resolution=1024):
        self.dataset = dataset
        self.resolution = resolution
        self.transform = transforms.Compose(
            [
                transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]
        image = self.transform(image)
        height = int(item["height"])
        width = int(item["width"])
        global_caption = item["global_caption"]
        region_bboxes_list = item["bbox_list"]
        detail_region_caption_list = item["detail_region_captions"]
        region_caption_list = item["region_captions"]
        file_name = item["file_name"]

        region_bboxes_list = ast.literal_eval(region_bboxes_list)
        region_bboxes_list = adjust_and_normalize_bboxes(region_bboxes_list, width, height)
        region_bboxes_list = np.array(region_bboxes_list, dtype=np.float32)

        region_caption_list = ast.literal_eval(region_caption_list)
        detail_region_caption_list = ast.literal_eval(detail_region_caption_list)

        return {
            "image": image,
            "global_caption": global_caption,
            "detail_region_caption_list": detail_region_caption_list,
            "region_bboxes_list": region_bboxes_list,
            "region_caption_list": region_caption_list,
            "file_name": file_name,
            "height": height,
            "width": width,
        }


# ---------------------------------------------------------------------------
# CLIP + PickScore
# ---------------------------------------------------------------------------


def calculate_clip_pick_score_shard(prompts, image_dir, gpu_id):
    device = torch.device(f"cuda:{gpu_id}")
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(device)

    pick_processor = AutoProcessor.from_pretrained(PICKSCORE_PROCESSOR_ID)
    pick_model = AutoModel.from_pretrained(PICKSCORE_MODEL_ID).eval().to(device)

    shard_results = {}
    shard_clip_sum = 0.0
    shard_pick_sum = 0.0

    with torch.no_grad():
        for prompt in tqdm(prompts, total=len(prompts), desc=f"GPU-{gpu_id} 计算 CLIP 和 PickScore", position=gpu_id):
            file_name = f"{prompt['index']:04d}.jpg"
            image_file = os.path.join(image_dir, file_name)
            if not os.path.exists(image_file):
                print(f"[GPU-{gpu_id}] Warning: image not found: {image_file}")
                continue
            image = Image.open(image_file).convert("RGB")
            inputs = clip_processor(
                text=[prompt["global_caption"]],
                images=image,
                max_length=77,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = clip_model(**inputs)
            image_features = outputs.image_embeds
            text_features = outputs.text_embeds

            clip_score = torch.nn.functional.cosine_similarity(image_features, text_features)

            image_inputs = pick_processor(
                images=image,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            ).to(device)

            text_inputs = pick_processor(
                text=prompt["global_caption"],
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            ).to(device)

            image_embs = pick_model.get_image_features(**image_inputs)
            image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)

            text_embs = pick_model.get_text_features(**text_inputs)
            text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)

            pick_score = pick_model.logit_scale.exp() * (text_embs @ image_embs.T)[0]

            shard_results[file_name] = {"clip_score": clip_score.item(), "pick_score": pick_score.item()}
            shard_clip_sum += clip_score.item()
            shard_pick_sum += pick_score.item()

    return shard_results, shard_clip_sum, shard_pick_sum


def clip_pick_worker(gpu_id, shard_prompts, image_dir):
    torch.cuda.set_device(gpu_id)
    return calculate_clip_pick_score_shard(shard_prompts, image_dir, gpu_id)


def run_clip_pick(
    *,
    benchmark_file: Path,
    generated_images_dir: str,
    metadata_file: str | None = None,
    metadata_file_with_scores: str | None = None,
    temp_dir: Path | None = None,
) -> None:
    temp_dir = temp_dir or DEFAULT_CLIP_TEMP_DIR
    os.makedirs(temp_dir, exist_ok=True)

    if metadata_file is None:
        metadata_file = str(Path(generated_images_dir).parent / "minicpm-vqa.json")
    if metadata_file_with_scores is None:
        metadata_file_with_scores = metadata_file.replace(".json", "-with-scores.json")

    with open(metadata_file, "r") as f:
        file_info = json.load(f)

    with open(benchmark_file, "r") as f:
        prompts = json.load(f)

    prompts = sorted(prompts, key=lambda x: x["index"])
    valid_prompts = [p for p in prompts if f"{p['index']:04d}.jpg" in file_info]

    gpu_count = torch.cuda.device_count()
    if gpu_count == 0:
        raise RuntimeError("No visible CUDA device found. Please check CUDA_VISIBLE_DEVICES.")
    print(f"[clip/pick] Using {gpu_count} visible GPU(s): {list(range(gpu_count))}")
    print(f"[clip/pick] Total prompts to score: {len(valid_prompts)}")

    prompt_shards = [s for s in chunk_evenly(valid_prompts, gpu_count) if s]
    total_clip_sum = 0.0
    total_pick_sum = 0.0
    scored_count = 0

    ctx = get_context("spawn")
    with ProcessPoolExecutor(max_workers=len(prompt_shards), mp_context=ctx) as executor:
        futures = []
        for gpu_id, shard in enumerate(prompt_shards):
            futures.append(executor.submit(clip_pick_worker, gpu_id, shard, generated_images_dir))

        for future in as_completed(futures):
            shard_results, shard_clip_sum, shard_pick_sum = future.result()
            total_clip_sum += shard_clip_sum
            total_pick_sum += shard_pick_sum
            scored_count += len(shard_results)
            for file_name, scores in shard_results.items():
                file_info[file_name]["clip_score"] = scores["clip_score"]
                file_info[file_name]["pick_score"] = scores["pick_score"]

    if scored_count > 0:
        print(f"Average CLIP Score: {total_clip_sum / scored_count:.4f}")
        print(f"Average PickScore: {total_pick_sum / scored_count:.4f}")
    else:
        print("No valid samples were scored.")

    prompts_with_scores = OrderedDict(sorted(file_info.items(), key=lambda kv: int(kv[0].split(".")[0])))

    with open(metadata_file_with_scores, "w") as f:
        json.dump(prompts_with_scores, f, indent=2)
    print(f"[clip/pick] Wrote {metadata_file_with_scores}")


# ---------------------------------------------------------------------------
# MiniCPM-V eval (layoutsam_eval.py logic unchanged)
# ---------------------------------------------------------------------------


def parse_yes_no(answer):
    return 1.0 if ("Yes" in answer or "yes" in answer) else 0.0


def _to_text_list(value):
    if not isinstance(value, (list, tuple)):
        return [str(value)]
    if len(value) == 0:
        return []
    if isinstance(value[0], (list, tuple)):
        return [str(v[0]) for v in value]
    return [str(v) for v in value]


def _to_bbox_list(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()

    if not isinstance(value, list) or len(value) == 0:
        return []

    if isinstance(value[0], list) and len(value[0]) > 0 and isinstance(value[0][0], list):
        value = value[0]

    bbox_list = []
    for bbox in value:
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            bbox_list.append([float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])])
    return bbox_list


def process_one_image(model, tokenizer, batch, generated_img, temp_save_root):
    detial_region_caption_list = _to_text_list(batch["detail_region_caption_list"])
    region_caption_list = _to_text_list(batch["region_caption_list"])
    region_bboxes_list = _to_bbox_list(batch["region_bboxes_list"])

    bbox_count = len(region_caption_list)
    img_score_spatial = 0.0
    img_score_color = 0.0
    img_score_texture = 0.0
    img_score_shape = 0.0

    img = Image.open(generated_img)
    resolution = img.size[0]

    for bbox, detial_region_caption, region_caption in zip(
        region_bboxes_list, detial_region_caption_list, region_caption_list
    ):
        x1, y1, x2, y2 = bbox
        x1 = int(x1 * resolution)
        y1 = int(y1 * resolution)
        x2 = int(x2 * resolution)
        y2 = int(y2 * resolution)

        cropped_img = img.crop((x1, y1, x2, y2))

        description = region_caption.replace("/", "")
        detail_description = detial_region_caption.replace("/", "")
        cropped_img_path = os.path.join(temp_save_root, f"{description}.jpg")
        cropped_img.save(cropped_img_path)

        question = (
            f'Is the subject "{description}" present in the image? '
            'Strictly answer with "Yes" or "No", without any irrelevant words.'
        )
        msgs = [{"role": "user", "content": [cropped_img, question]}]
        res = model.chat(image=None, msgs=msgs, tokenizer=tokenizer, seed=42)
        score_spatial = parse_yes_no(res)

        score_color, score_texture, score_shape = 0.0, 0.0, 0.0
        if score_spatial == 1.0:
            question_color = (
                f'Is the subject in "{description}" in the image consistent with the color described in '
                f'the detailed description: "{detail_description}"? Strictly answer with "Yes" or "No", '
                "without any irrelevant words. If the color is not mentioned in the detailed description, "
                'the answer is "Yes".'
            )
            msgs_color = [{"role": "user", "content": [cropped_img, question_color]}]
            color_attribute = model.chat(image=None, msgs=msgs_color, tokenizer=tokenizer, seed=42)
            score_color = parse_yes_no(color_attribute)

            question_texture = (
                f'Is the subject in "{description}" in the image consistent with the texture described in '
                f'the detailed description: "{detail_description}"? Strictly answer with "Yes" or "No", '
                "without any irrelevant words. If the texture is not mentioned in the detailed description, "
                'the answer is "Yes".'
            )
            msgs_texture = [{"role": "user", "content": [cropped_img, question_texture]}]
            texture_attribute = model.chat(image=None, msgs=msgs_texture, tokenizer=tokenizer, seed=42)
            score_texture = parse_yes_no(texture_attribute)

            question_shape = (
                f'Is the subject in "{description}" in the image consistent with the shape described in '
                f'the detailed description: "{detail_description}"? Strictly answer with "Yes" or "No", '
                "without any irrelevant words. If the shape is not mentioned in the detailed description, "
                'the answer is "Yes".'
            )
            msgs_shape = [{"role": "user", "content": [cropped_img, question_shape]}]
            shape_attribute = model.chat(image=None, msgs=msgs_shape, tokenizer=tokenizer, seed=42)
            score_shape = parse_yes_no(shape_attribute)

        img_score_spatial += score_spatial
        img_score_color += score_color
        img_score_texture += score_texture
        img_score_shape += score_shape

    return {
        "bbox_count": bbox_count,
        "score_spatial": img_score_spatial,
        "score_color": img_score_color,
        "score_texture": img_score_texture,
        "score_shape": img_score_shape,
    }


def worker_process(
    gpu_id,
    model_id,
    dataset_path,
    generate_path,
    temp_root,
    assigned_indices,
    processed_filenames,
):
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")

    model = AutoModel.from_pretrained(
        model_id,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    )
    model = model.eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    test_dataset = load_dataset(dataset_path, split="test")
    test_dataset = BboxDataset(test_dataset)

    worker_results = {}
    for idx in tqdm(assigned_indices, desc=f"GPU-{gpu_id}", position=gpu_id):
        filename = f"{idx:04d}.jpg"
        if filename in processed_filenames:
            continue

        generated_img = os.path.join(generate_path, filename)
        if not os.path.exists(generated_img):
            print(f"[GPU-{gpu_id}] Warning: {generated_img} does not exist.")
            continue

        temp_save_root = os.path.join(temp_root, filename.replace(".jpg", ""))
        os.makedirs(temp_save_root, exist_ok=True)
        batch = test_dataset[idx]

        image_stat = process_one_image(
            model=model,
            tokenizer=tokenizer,
            batch=batch,
            generated_img=generated_img,
            temp_save_root=temp_save_root,
        )
        worker_results[filename] = image_stat

    return worker_results


def filename_to_index(name):
    try:
        return int(name.split(".")[0])
    except Exception:
        return math.inf


def run_minicpm_eval(
    *,
    model_id: str,
    dataset_path: str,
    generate_path: str,
    sampled_bench_path: Path,
    save_json_path: str | None = None,
    score_save_path: str | None = None,
    temp_root: str | None = None,
) -> None:
    if save_json_path is None:
        save_json_path = str(Path(generate_path).parent / "minicpm-vqa.json")
    if score_save_path is None:
        score_save_path = str(Path(save_json_path).parent / "minicpm-vqa-score.txt")
    if temp_root is None:
        temp_root = str(Path(generate_path).parent / "images-perarea")

    os.makedirs(temp_root, exist_ok=True)

    with open(sampled_bench_path, "r") as f:
        sampled_bench = json.load(f)
    all_indices = [item["index"] for item in sampled_bench]

    print("total sampled num:", len(all_indices))
    print("processing:", generate_path)
    print("save_json_path:", save_json_path)

    image_stats = {}
    if os.path.exists(save_json_path):
        with open(save_json_path, "r") as f:
            image_stats = json.load(f)
        print(f"Read {len(image_stats)} processed images from {save_json_path}")

    processed_filenames = set(image_stats.keys())
    pending_indices = [idx for idx in all_indices if f"{idx:04d}.jpg" not in processed_filenames]
    print("pending sampled num:", len(pending_indices))

    gpu_count = torch.cuda.device_count()
    if gpu_count == 0:
        raise RuntimeError("No visible CUDA device found. Please check CUDA_VISIBLE_DEVICES.")

    print(f"Using {gpu_count} visible GPU(s): {list(range(gpu_count))}")

    shards = [s for s in chunk_evenly(pending_indices, gpu_count) if s]
    if shards:
        ctx = get_context("spawn")
        with ProcessPoolExecutor(max_workers=len(shards), mp_context=ctx) as executor:
            futures = []
            for gpu_id, shard in enumerate(shards):
                futures.append(
                    executor.submit(
                        worker_process,
                        gpu_id,
                        model_id,
                        dataset_path,
                        generate_path,
                        temp_root,
                        shard,
                        processed_filenames,
                    )
                )

            for future in as_completed(futures):
                shard_result = future.result()
                image_stats.update(shard_result)

    sorted_items = sorted(image_stats.items(), key=lambda kv: filename_to_index(kv[0]))
    sorted_image_stats = OrderedDict(sorted_items)

    with open(save_json_path, "w", encoding="utf-8") as json_file:
        json.dump(sorted_image_stats, json_file, indent=4)
    print(f"Image statistics saved to {save_json_path}")

    total_num = 0
    total_score_spatial = 0.0
    total_score_color = 0.0
    total_score_texture = 0.0
    total_score_shape = 0.0
    miss_match = 0

    for _, value in sorted_image_stats.items():
        total_num += value["bbox_count"]
        total_score_spatial += value["score_spatial"]
        total_score_color += value["score_color"]
        total_score_texture += value["score_texture"]
        total_score_shape += value["score_shape"]

        if (
            value["bbox_count"] != value["score_spatial"]
            or value["bbox_count"] != value["score_color"]
            or value["bbox_count"] != value["score_texture"]
            or value["bbox_count"] != value["score_shape"]
        ):
            miss_match += 1

    print("mismatch image num:", miss_match)
    if total_num == 0:
        raise RuntimeError("No valid bbox evaluated. Cannot compute average scores.")

    with open(score_save_path, "w") as f:
        f.write(f"Total number of bbox: {total_num}\n")
        f.write(
            f"Total score of spatial: {total_score_spatial}; Average score of spatial: {round(total_score_spatial / total_num, 4)}\n"
        )
        f.write(
            f"Total score of color: {total_score_color}; Average score of color: {round(total_score_color / total_num, 4)}\n"
        )
        f.write(
            f"Total score of texture: {total_score_texture}; Average score of texture: {round(total_score_texture / total_num, 4)}\n"
        )
        f.write(
            f"Total score of shape: {total_score_shape}; Average score of shape: {round(total_score_shape / total_num, 4)}\n"
        )


def main():
    import argparse

    p = argparse.ArgumentParser(description="LayoutSAM: MiniCPM-V QA, then CLIP/PickScore on the same bench JSON")
    p.add_argument(
        "--generate-path",
        type=Path,
        default=DEFAULT_GENERATE_PATH,
        help="Directory of generated 0000.jpg … (default: bench/output/layoutsam/images under repo root)",
    )
    p.add_argument(
        "--sampled-bench",
        type=Path,
        default=DEFAULT_SAMPLED_BENCH,
        help="layoutsam_sampled.json: indices for MiniCPM and prompts (global_caption) for CLIP/Pick",
    )
    args = p.parse_args()

    generate_path = str(args.generate_path.resolve())
    sampled_bench = args.sampled_bench.resolve()

    run_minicpm_eval(
        model_id=MINICPM_MODEL_ID,
        dataset_path=LAYOUTSAM_EVAL_DATASET,
        generate_path=generate_path,
        sampled_bench_path=sampled_bench,
    )

    metadata_file = str(Path(generate_path).parent / "minicpm-vqa.json")
    if not os.path.isfile(metadata_file):
        raise SystemExit(f"Expected {metadata_file} after MiniCPM eval.")
    run_clip_pick(
        benchmark_file=sampled_bench,
        generated_images_dir=generate_path,
        metadata_file=metadata_file,
    )


if __name__ == "__main__":
    main()
