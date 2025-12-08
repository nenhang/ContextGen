import json
import os
import random
import time

import dotenv

if os.path.exists(".env"):
    dotenv.load_dotenv(".env")

import torch
from PIL import Image

from src.model.generate import generate, load_model

# from src.utils.image_process import annotate

KONTEXT_MODEL_PATH = os.getenv("KONTEXT_MODEL_PATH", "black-forest-labs/FLUX.1-Kontext-dev")
ADAPTER_PATH = os.getenv("ADAPTER_PATH", "/root/autodl-tmp/ckpt/ruihangxu/ContextGen")


def main(args, gpu_id=0):
    # Initialize model
    torch.cuda.set_device(gpu_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    contextgen_model = load_model(KONTEXT_MODEL_PATH, ADAPTER_PATH, device=device)
    print("Model loaded.")
    run_name = time.strftime("%Y%m%d-%H%M%S")
    if not os.path.exists(output_dir := os.path.join(args.output_dir, run_name)):
        os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(args.input_dir, "segment_info.json")) as f:
        segment_info = json.load(f)
    for i, item in enumerate(segment_info):
        for sample_id in range(args.num_samples):
            prompt = item["caption"]
            width = item["width"]
            reference_info = []
            layout_image = item.get("layout_image", None)
            if layout_image is not None:
                layout_image = os.path.join(args.input_dir, layout_image)
                assert os.path.exists(layout_image), f"Layout image {layout_image} does not exist."
                layout_image = Image.open(layout_image)
            for instance in item["instances"]:
                bbox = (torch.tensor(instance["bbox"]) * args.image_size / width).to(dtype=torch.int32)
                instance_image_path = os.path.join(args.input_dir, instance["image"])
                assert os.path.exists(instance_image_path), f"Image {instance_image_path} does not exist."
                if "mask" in instance:
                    instance_mask_path = os.path.join(args.input_dir, instance["mask"])
                    assert os.path.exists(instance_mask_path), f"Mask {instance_mask_path} does not exist."
                reference_info.append(
                    {"image": instance_image_path, "bbox": bbox, "mask": instance_mask_path}
                    if "mask" in instance
                    else {"image": instance_image_path, "bbox": bbox}
                )
            # bbox_for_annotation = [info["bbox"] for info in reference_info]
            # labels_for_annotation = [segment["label"] for segment in item["instances"]]

            seed = random.randint(0, 2**32 - 1)
            res_image = generate(
                flux_pipe=contextgen_model,
                prompts=[prompt],
                reference_info=[reference_info],
                width=args.image_size,
                height=args.image_size,
                layout_image=[layout_image] if layout_image is not None else None,
                seed=[seed],
                # debug_refs=True,
            )[0]
            res_image.save(os.path.join(output_dir, f"{i:02d}_{sample_id:02d}_{seed}.png"))
            # annotate(res_image, bbox_for_annotation, labels_for_annotation).save(
            #     os.path.join(output_dir, f"{i:02d}_{sample_id:02d}_{seed}.png")
            # )


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        default="./images/input",
        help="Directory containing input images and segment_info.json.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./images/output",
        help="Directory to save the generated samples.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=4,
        help="Number of samples to generate, if --sample_dataset is provided.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=768,
        help="The width and height of the generated images.",
    )
    args = parser.parse_args()
    main(args)
