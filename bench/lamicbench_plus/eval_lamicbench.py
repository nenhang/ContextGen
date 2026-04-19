import json
import os
from pathlib import Path

import cv2
import dotenv
import insightface
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor
from utils.aesthetic_predictor import AestheticPredictor
from utils.grounded_sam2 import GroundedSAM2
from utils.vqa_model import VQAScorer

# load dotenv
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[2]
dotenv.load_dotenv(_SCRIPT_DIR / ".env")

# config parameters here
generate_image_root = str(_REPO_ROOT / "bench" / "output" / "lamicbench_plus" / "images")
results_save_path = generate_image_root.replace("images", "metrics.json")
num_samples = 4  # the output name of the files should be like 0000_0.jpg, 0000_1.jpg, ...
max_reference_num_limit = None  # set to None to not limit the number of reference images (Some methods may only support limited number of reference images)
bench_file_dir = str(_SCRIPT_DIR / "lamicbench_plus_files")
benchmark_path = str(_SCRIPT_DIR / "lamicbench_plus_files" / "lamicbench_plus.json")
reference_image_root = os.path.join(bench_file_dir, "reference_images")
reference_mask_root = os.path.join(bench_file_dir, "reference_masks")

_REQUIRED_ENV_VARS = (
    "FACE_MODEL_ROOT",
    "CLIP_MODEL_ROOT",
    "DINO_MODEL_ROOT",
    "SAM2_CKPT",
    "GROUNDINGDINO_CKPT",
    "IMPROVED_AES_MODEL_PATH",
    "VQA_MODEL_ROOT",
)
_missing_env = [name for name in _REQUIRED_ENV_VARS if not os.getenv(name)]
if _missing_env:
    raise ValueError(
        "The following environment variables are not set in the environment (.env): " + ", ".join(_missing_env)
    )

face_model = insightface.app.FaceAnalysis(root=os.environ["FACE_MODEL_ROOT"], providers=["CUDAExecutionProvider"])
face_model.prepare(ctx_id=0, det_size=(640, 640))

clip_model = CLIPModel.from_pretrained(os.environ["CLIP_MODEL_ROOT"]).to(device="cuda")
clip_processor = CLIPProcessor.from_pretrained(os.environ["CLIP_MODEL_ROOT"])

dino_model = AutoModel.from_pretrained(os.environ["DINO_MODEL_ROOT"]).to(device="cuda")
dino_processor = AutoImageProcessor.from_pretrained(os.environ["DINO_MODEL_ROOT"])

grounded_sam2_model = GroundedSAM2(
    sam2_ckpt=os.environ["SAM2_CKPT"],
    groundingdino_ckpt=os.environ["GROUNDINGDINO_CKPT"],
    device="cuda",
)

improved_aes_model = AestheticPredictor(
    clip_model=clip_model,
    clip_processor=clip_processor,
    mlp_path=os.environ["IMPROVED_AES_MODEL_PATH"],
    clip_model_root=os.environ["CLIP_MODEL_ROOT"],
)

vqa_scorer = VQAScorer(
    vqa_model_path=os.environ["VQA_MODEL_ROOT"],
    attn_implementation="flash_attention_2",
    device="cuda",
)


def get_masked_dino_features(image_pil, mask_np, device="cuda"):
    image_tensor = dino_processor(images=image_pil, return_tensors="pt").to(device)
    with torch.inference_mode():
        features = dino_model(**image_tensor).last_hidden_state

    image_features = features[:, 1:, :]

    num_patches = image_features.shape[1]
    patch_grid_size = int(num_patches**0.5)

    if isinstance(mask_np, np.ndarray) and mask_np.dtype == bool:
        mask_np = mask_np.astype(np.float32)

    mask_resized = cv2.resize(mask_np, (patch_grid_size, patch_grid_size), interpolation=cv2.INTER_NEAREST)
    mask_tensor = torch.tensor(mask_resized, dtype=torch.bool).to(device)

    mask_expanded = mask_tensor.reshape(1, patch_grid_size * patch_grid_size, 1)
    masked_features = image_features[mask_expanded.expand_as(image_features)]
    global_feature = masked_features.view(-1, image_features.shape[-1]).mean(dim=0, keepdim=True)

    return global_feature


def cal_dino_score(image_path, label, reference_image_path, reference_mask_path, device="cuda"):
    sam_results = grounded_sam2_model.predict(
        img_path=image_path,
        text_prompt=label,
    )
    if not sam_results or len(sam_results) == 0:
        return 0.0
    instance_mask = sam_results["segmentation"]
    instance_bbox = sam_results["bbox"]

    # 从图片和掩码中 crop 出参考图像
    result_image = Image.open(image_path).convert("RGB")
    instance_bbox_int = tuple([round(coord) for coord in instance_bbox])
    image_width, image_height = result_image.size
    instance_bbox_int = (
        max(0, instance_bbox_int[0]),
        max(0, instance_bbox_int[1]),
        min(image_width, instance_bbox_int[2]),
        min(image_height, instance_bbox_int[3]),
    )

    bbox_width = instance_bbox_int[2] - instance_bbox_int[0]
    bbox_height = instance_bbox_int[3] - instance_bbox_int[1]

    if bbox_width <= 0 or bbox_height <= 0:
        print(
            f"Warning: Bounding box for {image_path} has zero or negative dimensions: width={bbox_width}, height={bbox_height}."
        )
        return 0.0  # Or some other appropriate fallback score

    instance_image = result_image.crop(instance_bbox_int)

    # instance_mask 是 np.ndarray 类型，可以直接用索引
    instance_mask = instance_mask[
        instance_bbox_int[1] : instance_bbox_int[3], instance_bbox_int[0] : instance_bbox_int[2]
    ].astype(bool)

    if np.sum(instance_mask) == 0:
        print(f"Warning: Instance mask is empty for {image_path}, using bbox instead.")
        instance_mask = np.ones_like(instance_mask, dtype=bool)

    # 读取参考图像和掩码
    ref_image = Image.open(reference_image_path).convert("RGB")
    ref_mask = Image.open(reference_mask_path).convert("L")
    ref_mask = np.array(ref_mask)
    ref_mask = ref_mask > 127
    assert np.sum(ref_mask) > 0, f"Reference mask is empty for {reference_mask_path}"

    # 获取全局特征向量
    instance_global_feature = get_masked_dino_features(instance_image, instance_mask, device=device)
    ref_global_feature = get_masked_dino_features(ref_image, ref_mask, device=device)

    # 计算两个全局特征向量之间的余弦相似度
    similarity = torch.cosine_similarity(instance_global_feature, ref_global_feature, dim=-1)
    dino_score = similarity.item()
    if np.isnan(dino_score):
        print(f"Warning: NaN score for {image_path} with reference {reference_image_path}")
        dino_score = 0.0
    return dino_score


def extract_all_faces(img_path):
    img = cv2.imread(img_path)
    faces = face_model.get(img)
    return [face.embedding for face in faces]  # 返回所有检测人脸的嵌入向量


def evaluate_group_id(original_img_path, generated_img_path):
    # 提取特征
    orig_face_list = extract_all_faces(original_img_path)
    assert len(orig_face_list) > 0, f"No faces found in original image: {original_img_path}"
    orig_emb = orig_face_list[0]
    generated_embeddings = extract_all_faces(generated_img_path)

    # 为每个原始人脸找生成图中最匹配的人
    similarities = []
    for gen_emb in generated_embeddings:
        similarity = np.dot(orig_emb, gen_emb) / (np.linalg.norm(orig_emb) * np.linalg.norm(gen_emb))
        similarities.append(float(similarity))

    # 获取最大相似度
    if similarities:
        max_similarity = max(similarities)
    else:
        max_similarity = 0.0

    return max_similarity


def cal_bench_score(benchmark_data, max_reference_num: int | None = None):
    results = []

    for item in tqdm(benchmark_data):
        sample_results = []
        for sample in range(num_samples):
            generate_image_path = f"{generate_image_root}/{item['index']:04d}_{sample}.jpg"
            if not os.path.exists(generate_image_path):
                print(f"Warning: Generated image not found: {generate_image_path}, skipping.")
                continue
            face_similarity_scores = []
            object_similarity_scores = []
            references = item["references"] if max_reference_num is None else item["references"][:max_reference_num]
            for ref in references:
                ref_image_path = f"{reference_image_root}/{ref['image_path']}"
                if "face" in ref["image_path"]:
                    if len(extract_all_faces(ref_image_path)) > 0:
                        face_similarity_scores.append(
                            {
                                "index": ref["ref_index"],
                                "score": evaluate_group_id(ref_image_path, generate_image_path),
                            }
                        )
                    else:
                        face_similarity_scores.append(
                            {
                                "index": ref["ref_index"],
                                "score": 0.0,
                            }
                        )
                if "face" not in ref["image_path"]:
                    dino_score = cal_dino_score(
                        image_path=generate_image_path,
                        label=ref["phrase"],
                        reference_image_path=ref_image_path,
                        reference_mask_path=f"{reference_mask_root}/{ref['image_path'].replace('.jpg', '_mask.png')}",
                    )
                    object_similarity_scores.append(
                        {
                            "index": ref["ref_index"],
                            "score": dino_score,
                        }
                    )
            assert len(face_similarity_scores) + len(object_similarity_scores) == len(references), (
                f"Score length mismatch for item {item['index']}"
            )
            improved_aes_score = improved_aes_model.predict(generate_image_path)
            sample_results.append(
                {
                    "sample_index": sample,
                    "face_similarity_scores": face_similarity_scores,
                    "object_similarity_scores": object_similarity_scores,
                    "improved_aes": improved_aes_score,
                }
            )

        results.append({"index": item["index"], "samples": sample_results})

    return results


def add_vqa_score(results, bench_data, max_reference_num: int | None = None):
    assert len(results) == len(bench_data), "Results and benchmark data length mismatch"
    sample_num = 0
    total_vqa_score = 0.0
    # sort result and bench_data by index
    results = sorted(results, key=lambda x: x["index"])
    bench_data = sorted(bench_data, key=lambda x: x["index"])
    for bench_item in tqdm(bench_data):
        if max_reference_num is not None:
            overflow_phrases = [ref["phrase"] for ref in bench_item["references"][max_reference_num:]]
        for sample in range(num_samples):
            # assert item["index"] == bench_item["index"], "Index mismatch between results and benchmark data"
            item = next((res for res in results if res["index"] == bench_item["index"]), None)
            if item is None:
                print(f"[Info] Skipping index {bench_item['index']}: no corresponding result found.")
                continue

            sample_entry = next((s for s in item["samples"] if s["sample_index"] == sample), None)
            if sample_entry is None:
                print(
                    f"Warning: No metrics entry for index {item['index']} sample {sample} "
                    f"(image may have been skipped in cal_bench_score); skipping VQA."
                )
                continue

            generate_image_path = f"{generate_image_root}/{item['index']:04d}_{sample}.jpg"
            if not os.path.exists(generate_image_path):
                print(f"Warning: Generated image not found: {generate_image_path}, skipping.")
                continue
            question = bench_item["vqa_prompts"]

            if max_reference_num is not None and len(bench_item["references"]) > max_reference_num:
                # 过滤掉包含 overflow_phrases 的问题
                filtered_questions = {}
                for qid, q in question.items():
                    if not any(phrase in q for phrase in overflow_phrases):
                        filtered_questions[qid] = q
                print(
                    f"Filtered out {len(question) - len(filtered_questions)} questions containing overflow phrases for index {bench_item['index']}"
                )
                question = filtered_questions
                if len(question) == 0:
                    print(f"All questions filtered out for index {bench_item['index']}, skipping.")
                    continue

            samples_vqa_score = 0.0
            for question_id, q in question.items():
                sample_vqa_score = vqa_scorer.score(generate_image_path, q)
                samples_vqa_score += float(sample_vqa_score)
            item_vqa_score = samples_vqa_score / len(question)
            sample_entry["vqa_score"] = item_vqa_score
            total_vqa_score += item_vqa_score
            sample_num += 1
    if sample_num == 0:
        print("No VQA scores were computed (sample_num=0).")
    else:
        avg_vqa_score = total_vqa_score / sample_num
        print(f"Average VQA score across all samples: {avg_vqa_score:.4f}")
    return results


if __name__ == "__main__":
    with open(benchmark_path, "r") as f:
        benchmark_data = json.load(f)
    results = cal_bench_score(benchmark_data=benchmark_data, max_reference_num=max_reference_num_limit)
    results_with_vqa = add_vqa_score(
        results=results, bench_data=benchmark_data, max_reference_num=max_reference_num_limit
    )
    with open(results_save_path, "w") as f:
        json.dump(results_with_vqa, f, indent=4)
    print(f"Updated results saved to {results_save_path}")
