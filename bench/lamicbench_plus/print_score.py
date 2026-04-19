import json
import os
from pathlib import Path

from tabulate import tabulate

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[2]

BENCH_JSON = _SCRIPT_DIR / "lamicbench_plus_files" / "lamicbench_plus.json"

models_dirs = [str(_REPO_ROOT / "bench" / "output")]
PRINT_SPLIT_INSTANCE_TABLES = True


def cal_scores(metrics, bench_data, incides: list | None = None):
    if incides is not None:
        filtered_metrics = []
        for index in incides:
            found = False
            for item in metrics:
                if item["index"] == index:
                    filtered_metrics.append(item)
                    found = True
                    break
            if not found:
                print(f"Warning: Index {index} not found in metrics.")
        metrics = filtered_metrics

        filtered_bench_data = []
        for index in incides:
            found = False
            for item in bench_data:
                if item["index"] == index:
                    filtered_bench_data.append(item)
                    found = True
                    break
            if not found:
                print(f"Warning: Index {index} not found in benchmark data.")
        bench_data = filtered_bench_data

    score_dict = {
        "all": {
            "improved_aes": 0.0,
            "vqa_score": 0.0,
            "face_similarity_score": 0.0,
            "face_count": 0,
            "object_similarity_score": 0.0,
            "object_count": 0,
            "total_samples": 0,
            "total_items": 0,
        },
        2: {
            "improved_aes": 0.0,
            "vqa_score": 0.0,
            "face_similarity_score": 0.0,
            "face_count": 0,
            "object_similarity_score": 0.0,
            "object_count": 0,
            "total_samples": 0,
            "total_items": 0,
        },
        3: {
            "improved_aes": 0.0,
            "vqa_score": 0.0,
            "face_similarity_score": 0.0,
            "face_count": 0,
            "object_similarity_score": 0.0,
            "object_count": 0,
            "total_samples": 0,
            "total_items": 0,
        },
        4: {
            "improved_aes": 0.0,
            "vqa_score": 0.0,
            "face_similarity_score": 0.0,
            "face_count": 0,
            "object_similarity_score": 0.0,
            "object_count": 0,
            "total_samples": 0,
            "total_items": 0,
        },
        5: {
            "improved_aes": 0.0,
            "vqa_score": 0.0,
            "face_similarity_score": 0.0,
            "face_count": 0,
            "object_similarity_score": 0.0,
            "object_count": 0,
            "total_samples": 0,
            "total_items": 0,
        },
        6: {
            "improved_aes": 0.0,
            "vqa_score": 0.0,
            "face_similarity_score": 0.0,
            "face_count": 0,
            "object_similarity_score": 0.0,
            "object_count": 0,
            "total_samples": 0,
            "total_items": 0,
        },
    }
    assert len(metrics) == len(bench_data), (
        f"Metrics and benchmark data length mismatch: {len(metrics)} != {len(bench_data)}"
    )
    for item, bench_item in zip(metrics, bench_data):
        assert item["index"] == bench_item["index"], (
            f"Index mismatch between metrics and benchmark data: {item['index']} != {bench_item['index']}"
        )
        num_references = len(bench_item["references"])
        key = num_references if num_references <= 6 else 6
        selected_samples = item["samples"]

        for sample in selected_samples:
            score_dict["all"]["improved_aes"] += sample["improved_aes"]
            score_dict["all"]["vqa_score"] += sample["vqa_score"]
            score_dict["all"]["face_similarity_score"] += sum(
                score["score"] for score in sample["face_similarity_scores"]
            )
            score_dict["all"]["face_count"] += len(sample["face_similarity_scores"])
            score_dict["all"]["object_similarity_score"] += sum(
                score["score"] for score in sample["object_similarity_scores"]
            )
            score_dict["all"]["object_count"] += len(sample["object_similarity_scores"])
            score_dict["all"]["total_samples"] += 1

            score_dict[key]["improved_aes"] += sample["improved_aes"]
            score_dict[key]["vqa_score"] += sample["vqa_score"]
            score_dict[key]["face_similarity_score"] += sum(
                score["score"] for score in sample["face_similarity_scores"]
            )
            score_dict[key]["face_count"] += len(sample["face_similarity_scores"])
            score_dict[key]["object_similarity_score"] += sum(
                score["score"] for score in sample["object_similarity_scores"]
            )
            score_dict[key]["object_count"] += len(sample["object_similarity_scores"])
            score_dict[key]["total_samples"] += 1

        score_dict[key]["total_items"] += 1
        score_dict["all"]["total_items"] += 1

    return score_dict


def calculate_averages(score_dict):
    avg_dict = {}
    for key, scores in score_dict.items():
        avg_dict[key] = {
            "improved_aes": scores["improved_aes"] / scores["total_samples"] if scores["total_samples"] > 0 else 0.0,
            "vqa_score": scores["vqa_score"] / scores["total_samples"] if scores["total_samples"] > 0 else 0.0,
            "face_similarity_score": scores["face_similarity_score"] / scores["face_count"]
            if scores["face_count"] > 0
            else 0.0,
            "object_similarity_score": scores["object_similarity_score"] / scores["object_count"]
            if scores["object_count"] > 0
            else 0.0,
        }
    return avg_dict


def merge_groups(score_dict, group_keys):
    merged = {
        "improved_aes": 0.0,
        "vqa_score": 0.0,
        "face_similarity_score": 0.0,
        "face_count": 0,
        "object_similarity_score": 0.0,
        "object_count": 0,
        "total_samples": 0,
    }
    for key in group_keys:
        group = score_dict[key]
        merged["improved_aes"] += group["improved_aes"]
        merged["vqa_score"] += group["vqa_score"]
        merged["face_similarity_score"] += group["face_similarity_score"]
        merged["face_count"] += group["face_count"]
        merged["object_similarity_score"] += group["object_similarity_score"]
        merged["object_count"] += group["object_count"]
        merged["total_samples"] += group["total_samples"]

    return {
        "improved_aes": merged["improved_aes"] / merged["total_samples"] if merged["total_samples"] > 0 else 0.0,
        "vqa_score": merged["vqa_score"] / merged["total_samples"] if merged["total_samples"] > 0 else 0.0,
        "face_similarity_score": merged["face_similarity_score"] / merged["face_count"]
        if merged["face_count"] > 0
        else 0.0,
        "object_similarity_score": merged["object_similarity_score"] / merged["object_count"]
        if merged["object_count"] > 0
        else 0.0,
    }


def post_process_scores(all_scores):
    for _, scores in all_scores.items():
        for key in scores:
            scores[key]["improved_aes"] *= 10
            scores[key]["vqa_score"] *= 100
            scores[key]["face_similarity_score"] *= 100
            scores[key]["object_similarity_score"] *= 100
    return all_scores


def calculate_overall_average(scores):
    return (
        scores["improved_aes"]
        + scores["vqa_score"]
        + (scores["face_similarity_score"] + scores["object_similarity_score"])
    ) / 4


def validate_metric_benchmark_alignment(metrics, bench_indices, model_name):
    metric_indices = [item.get("index") for item in metrics if isinstance(item, dict)]
    metric_index_set = set(metric_indices)
    bench_index_set = set(bench_indices)

    duplicate_count = len(metric_indices) - len(metric_index_set)
    if duplicate_count > 0:
        print(f"[{model_name}] Warning: found {duplicate_count} duplicate metric indices.")

    missing_in_metrics = sorted(bench_index_set - metric_index_set)
    if missing_in_metrics:
        print(
            f"[{model_name}] Warning: {len(missing_in_metrics)} benchmark indices are missing in metrics. "
            f"Examples: {missing_in_metrics[:10]}"
        )
    return len(missing_in_metrics) == 0 and duplicate_count == 0


def build_table_rows(sorted_models, group_key, model_name_dict):
    table_data = []
    for model_name, scores in sorted_models:
        score_group = scores[group_key]
        overall_avg = calculate_overall_average(score_group)
        row = [
            model_name_dict.get(model_name, model_name),
            f"{score_group['improved_aes']:.2f}",
            f"{score_group['vqa_score']:.2f}",
            f"{score_group['face_similarity_score']:.2f}",
            f"{score_group['object_similarity_score']:.2f}",
            f"{overall_avg:.2f}",
        ]
        table_data.append(row)
    return table_data


def load_selected_subset(path: Path):
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        bench_data = data
    elif isinstance(data, dict) and "benchmark" in data:
        bench_data = data["benchmark"]
    else:
        raise ValueError(f"Expected a JSON array or an object with key 'benchmark', got {type(data).__name__}")
    bench_data = sorted(bench_data, key=lambda x: x["index"])
    indices = [b["index"] for b in bench_data]
    return bench_data, indices


if __name__ == "__main__":
    if not BENCH_JSON.is_file():
        raise SystemExit(f"Missing {BENCH_JSON}. Run summurize_selected_indices.py first to generate the subset file.")

    bench_data, selected_indices = load_selected_subset(BENCH_JSON)
    print(f"Loaded selected subset: {len(bench_data)} items from {BENCH_JSON.name}")

    all_results = {}

    if isinstance(models_dirs, str):
        models_dir_list = [models_dirs]
    else:
        models_dir_list = models_dirs

    for models_dir in models_dir_list:
        for model_name in os.listdir(models_dir):
            model_path = os.path.join(models_dir, model_name)
            if not os.path.isdir(model_path):
                continue
            metrics_path = os.path.join(model_path, "metrics.json")
            if not os.path.exists(metrics_path):
                continue
            with open(metrics_path) as f:
                metrics = json.load(f)
            validate_metric_benchmark_alignment(metrics, selected_indices, model_name)
            raw_scores = cal_scores(metrics=metrics, bench_data=bench_data, incides=selected_indices)
            avg_scores = calculate_averages(raw_scores)
            avg_scores["2-3"] = merge_groups(raw_scores, [2, 3])
            avg_scores["4-5"] = merge_groups(raw_scores, [4, 5, 6])
            all_results[model_name] = avg_scores

    all_results = post_process_scores(all_results)
    headers = [
        "Model",
        "AES",
        "ITC",
        "IDS",
        "IPS",
        "AVG",
    ]

    model_name_dict = {}

    if PRINT_SPLIT_INSTANCE_TABLES:
        sorted_small = sorted(
            all_results.items(),
            key=lambda x: calculate_overall_average(x[1]["2-3"]),
            reverse=False,
        )
        sorted_large = sorted(
            all_results.items(),
            key=lambda x: calculate_overall_average(x[1]["4-5"]),
            reverse=False,
        )

        print("\n=== Small Instance (2-3) ===")
        table_data_small = build_table_rows(sorted_small, "2-3", model_name_dict)
        print(tabulate(table_data_small, headers=headers, tablefmt="grid", stralign="right"))

        print("\n=== Large Instance (4+) ===")
        table_data_large = build_table_rows(sorted_large, "4-5", model_name_dict)
        print(tabulate(table_data_large, headers=headers, tablefmt="grid", stralign="right"))
    else:
        sorted_models = sorted(
            all_results.items(),
            key=lambda x: calculate_overall_average(x[1]["all"]),
            reverse=False,
        )
        table_data = build_table_rows(sorted_models, "all", model_name_dict)
        print(tabulate(table_data, headers=headers, tablefmt="grid", stralign="right"))
